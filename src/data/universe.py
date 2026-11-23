"""Scanner universe otomatis — pilih koin yang dianalisis berdasarkan peringkat volume / market cap.

Sumber (config/symbols.yaml -> scanner.sources, dicoba berurutan sampai ada yang berhasil):
  binance_volume     top N pasangan USDT berdasarkan quoteVolume 24 jam (Binance, tanpa API key)
  bitget_volume      top N pasangan USDT berdasarkan usdtVolume 24 jam (Bitget v2 spot, tanpa API key)
  coinmarketcap_rank peringkat CMC 1..N (market cap). Butuh API key gratis di env CMC_API_KEY.
                     Koin harus punya pasangan USDT di bursa (cmc.availability) agar datanya bisa diambil.

Prinsip validitas forward test (penting):
  * Universe di-REFRESH sekali per `refresh_hours` (default 24 jam) dan disimpan di output/universe.json
    -> daftar stabil sepanjang hari & bisa diaudit. Tiap refresh dicatat di output/universe_history.csv.
  * Peringkat memakai data SAAT INI saja (tanpa lookahead terhadap sinyal yang dicatat sesudahnya).
  * config_hash memuat ATURAN scanner (sumber, top_n, filter), bukan daftar koin hasilnya (lihat hash_basis).
  * Bila semua sumber gagal: pakai cache lama (aturan sama) -> bila tidak ada, daftar statis `symbols`.
    Fallback SELALU ditandai (status) di signals.json, laporan, dan dashboard — tidak pernah diam-diam.
  * Stablecoin, token wrapped/emas, dan token leveraged dikeluarkan (bukan aset yang relevan untuk metodologi ini).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from .fetch_ohlcv import BinanceRestProvider

log = logging.getLogger("engine.universe")

BITGET_TICKERS = "https://api.bitget.com/api/v2/spot/market/tickers"
CMC_LISTINGS = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

DEFAULT_EXCLUDE = [
    # stablecoin / fiat
    "USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "USDS", "USD1", "BUSD", "PYUSD", "BFUSD", "XUSD", "RLUSD",
    "AEUR", "EUR", "EURI", "EURC", "GBP", "TRY", "BRL", "USDD", "FRAX", "LUSD", "GUSD", "SUSD",
    # wrapped / staked turunan
    "WBTC", "WBETH", "WETH", "STETH", "WSTETH", "CBBTC", "BTCB", "WEETH", "RETH",
    # token emas
    "PAXG", "XAUT",
]

SCANNER_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "sources": ["binance_volume", "bitget_volume"],
    "top_n": 30,
    "quote": "USDT",
    "min_quote_volume_usdt": 5_000_000,   # filter likuiditas minimum (24 jam)
    "refresh_hours": 24,
    "exclude": DEFAULT_EXCLUDE,
    "always_include": [],                 # simbol yang selalu ikut (di luar peringkat), mis. ["BTC/USDT"]
    "cmc": {"limit": 200, "availability": ["binance_volume", "bitget_volume"]},
}

_LEV_SUFFIX = re.compile(r"^(?P<root>[A-Z0-9]+?)(UP|DOWN|BULL|BEAR|[235][LS])$")
HttpGet = Callable[..., Any]


class UniverseError(Exception):
    pass


@dataclass
class Universe:
    symbols: List[str]
    mode: str                     # static | scanner
    source: str                   # sumber yang dipakai: binance_volume | bitget_volume | coinmarketcap_rank | static | manual_override | demo
    status: str                   # static | fresh | cached | stale_cache | fallback_static
    fetched_at: Optional[str] = None
    ranks: Dict[str, int] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    rules_hash: str = ""
    top_n: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "source": self.source, "status": self.status, "fetched_at": self.fetched_at,
                "count": len(self.symbols), "top_n": self.top_n, "rules_hash": self.rules_hash,
                "symbols": self.symbols, "ranks": self.ranks, "volumes_24h_usdt": self.volumes, "warnings": self.warnings}

    def describe(self) -> str:
        if self.mode == "static":
            return f"Universe: daftar statis ({len(self.symbols)} simbol; {self.source})"
        name = {"binance_volume": f"Binance top {self.top_n} volume 24j", "bitget_volume": f"Bitget top {self.top_n} volume 24j",
                "coinmarketcap_rank": f"CoinMarketCap peringkat 1–{self.top_n}"}.get(self.source, self.source)
        txt = f"Universe: {name} — {len(self.symbols)} simbol, status **{self.status}**"
        if self.fetched_at:
            txt += f", di-refresh {self.fetched_at[:16].replace('T', ' ')} UTC"
        if self.status in ("stale_cache", "fallback_static"):
            txt += " ⚠️"
        return txt


# ---------------------------------------------------------------- konfigurasi
def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def scanner_cfg(symbols_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_merge(SCANNER_DEFAULTS, symbols_cfg.get("scanner") or {})


def scanner_active(cfg) -> bool:
    """Scanner aktif bila enabled di config DAN simbol tidak di-override manual (ENGINE_SYMBOLS / --symbols)."""
    return bool(scanner_cfg(cfg.symbols_cfg).get("enabled")) and not getattr(cfg, "symbols_overridden", False)


def _rules(scfg: Dict[str, Any], benchmark: Optional[str]) -> Dict[str, Any]:
    return {"sources": list(scfg["sources"]), "top_n": int(scfg["top_n"]), "quote": scfg["quote"],
            "min_quote_volume_usdt": float(scfg["min_quote_volume_usdt"]),
            "exclude": sorted({s.upper() for s in scfg["exclude"]}),
            "always_include": sorted(scfg["always_include"]), "benchmark": benchmark,
            "cmc": {"limit": int(scfg["cmc"]["limit"]), "availability": list(scfg["cmc"]["availability"])}}


def rules_hash(symbols_cfg: Dict[str, Any]) -> str:
    rules = _rules(scanner_cfg(symbols_cfg), symbols_cfg.get("benchmark"))
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:12]


def hash_basis(symbols_cfg: Dict[str, Any], active: bool) -> Dict[str, Any]:
    """Bahan config_hash. Scanner nonaktif -> IDENTIK dengan sctn/sctnv2 lama (kunci `scanner` dibuang).
    Scanner aktif -> daftar `symbols` diganti ATURAN scanner, sehingga hash tidak berubah tiap koin keluar/masuk top N."""
    d = copy.deepcopy(symbols_cfg)
    scfg = d.pop("scanner", None)
    if active:
        d["symbols"] = {"scanner_rules": _rules(scanner_cfg({"scanner": scfg or {}}), symbols_cfg.get("benchmark"))}
    return d


# ---------------------------------------------------------------- sumber data
def _ok(r, name: str):
    if getattr(r, "status_code", 200) != 200:
        raise UniverseError(f"{name}: HTTP {r.status_code}")
    return r.json()


def fetch_binance_tickers(get: HttpGet, timeout: int, quote: str = "USDT") -> Dict[str, float]:
    """{base: quoteVolume24h} dari /api/v3/ticker/24hr (multi-host seperti fetch candle; sebagian host diblokir geo)."""
    last = "tidak ada host dicoba"
    for host in BinanceRestProvider.HOSTS:
        try:
            data = _ok(get(f"{host}/api/v3/ticker/24hr", timeout=timeout), host)
            out: Dict[str, float] = {}
            for t in data:
                sym = str(t.get("symbol", ""))
                if sym.endswith(quote) and len(sym) > len(quote):
                    out[sym[: -len(quote)]] = float(t.get("quoteVolume") or 0.0)
            if out:
                return out
            last = f"{host}: respons kosong"
        except Exception as exc:  # noqa: BLE001
            last = f"{host}: {exc}"
    raise UniverseError(f"binance: {last}")


def fetch_bitget_tickers(get: HttpGet, timeout: int, quote: str = "USDT") -> Dict[str, float]:
    j = _ok(get(BITGET_TICKERS, timeout=timeout), "bitget")
    if str(j.get("code")) != "00000":
        raise UniverseError(f"bitget: code {j.get('code')} {j.get('msg')}")
    out: Dict[str, float] = {}
    for t in j.get("data") or []:
        sym = str(t.get("symbol", ""))
        if sym.endswith(quote) and len(sym) > len(quote):
            out[sym[: -len(quote)]] = float(t.get("usdtVolume") or t.get("quoteVolume") or 0.0)
    if not out:
        raise UniverseError("bitget: respons kosong")
    return out


def fetch_cmc_ranks(get: HttpGet, api_key: str, timeout: int, limit: int) -> List[Tuple[str, int]]:
    if not api_key:
        raise UniverseError("coinmarketcap_rank: env CMC_API_KEY belum diisi (buat key gratis di pro.coinmarketcap.com)")
    r = get(CMC_LISTINGS, params={"start": 1, "limit": limit, "convert": "USD"}, timeout=timeout,
            headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"})
    j = _ok(r, "coinmarketcap")
    if int((j.get("status") or {}).get("error_code", 0)) != 0:
        raise UniverseError(f"coinmarketcap: {(j.get('status') or {}).get('error_message')}")
    rows = [(str(d["symbol"]).upper(), int(d["cmc_rank"])) for d in j.get("data") or [] if d.get("cmc_rank")]
    if not rows:
        raise UniverseError("coinmarketcap: respons kosong")
    return sorted(rows, key=lambda x: x[1])


TICKER_FETCHERS = {"binance_volume": fetch_binance_tickers, "bitget_volume": fetch_bitget_tickers}


# ---------------------------------------------------------------- seleksi
def _is_leveraged(base: str, listed: Dict[str, float]) -> bool:
    m = _LEV_SUFFIX.match(base)
    return bool(m and m.group("root") in listed)  # BTCUP -> root BTC terdaftar; JUP -> root 'J' tidak -> aman


def eligible(base: str, vol: float, listed: Dict[str, float], scfg: Dict[str, Any]) -> bool:
    if not re.fullmatch(r"[A-Z0-9]{2,15}", base):
        return False
    if base in {s.upper() for s in scfg["exclude"]}:
        return False
    if _is_leveraged(base, listed):
        return False
    return vol >= float(scfg["min_quote_volume_usdt"])


def select_by_volume(tickers: Dict[str, float], scfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    rows = [(b, v) for b, v in tickers.items() if eligible(b, v, tickers, scfg)]
    rows.sort(key=lambda x: -x[1])
    return rows[: int(scfg["top_n"])]


def select_by_rank(ranks: List[Tuple[str, int]], tickers: Dict[str, float], scfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    seen = set()
    for base, _rank in ranks:
        if base in seen or base not in tickers:
            continue
        if eligible(base, tickers[base], tickers, scfg):
            seen.add(base)
            out.append((base, tickers[base]))
        if len(out) >= int(scfg["top_n"]):
            break
    return out


def _build(source: str, scfg: Dict[str, Any], get: HttpGet, timeout: int, env: Dict[str, str]) -> List[Tuple[str, float]]:
    quote = scfg["quote"]
    if source in TICKER_FETCHERS:
        rows = select_by_volume(TICKER_FETCHERS[source](get, timeout, quote), scfg)
    elif source == "coinmarketcap_rank":
        ranks = fetch_cmc_ranks(get, env.get("CMC_API_KEY", ""), timeout, int(scfg["cmc"]["limit"]))
        tickers, last = None, "tidak ada"
        for av in scfg["cmc"]["availability"]:
            try:
                tickers = TICKER_FETCHERS[av](get, timeout, quote)
                break
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
        if tickers is None:
            raise UniverseError(f"coinmarketcap_rank: ketersediaan pasangan {quote} tidak bisa diperiksa ({last})")
        rows = select_by_rank(ranks, tickers, scfg)
    else:
        raise UniverseError(f"sumber scanner tidak dikenal: {source}")
    if not rows:
        raise UniverseError(f"{source}: tidak ada koin lolos filter")
    return rows


# ---------------------------------------------------------------- resolve + cache + history
def _finish(symbols: List[str], scfg: Dict[str, Any], benchmark: Optional[str]) -> List[str]:
    out = list(symbols)
    for extra in ([benchmark] if benchmark else []) + list(scfg["always_include"]):
        if extra and extra not in out:
            out.append(extra)
    if benchmark in out:
        out = [benchmark] + [s for s in out if s != benchmark]
    return out


def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _append_history(path: Path, ts: str, source: str, rhash: str, symbols: List[str], prev: Optional[List[str]]) -> None:
    prev_set = set(prev or [])
    added = [s for s in symbols if s not in prev_set]
    removed = [s for s in (prev or []) if s not in set(symbols)]
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["refreshed_utc", "source", "rules_hash", "count", "symbols", "added", "removed"])
        w.writerow([ts, source, rhash, len(symbols), " ".join(symbols), " ".join(added), " ".join(removed)])


def resolve_universe(cfg, now: pd.Timestamp, out_dir: Path, http_get: Optional[HttpGet] = None,
                     env: Optional[Dict[str, str]] = None, force_static: bool = False) -> Universe:
    static = list(cfg.symbols)
    bench = cfg.symbols_cfg.get("benchmark")
    if getattr(cfg, "symbols_overridden", False):
        return Universe(static, "static", "manual_override", "static")
    if force_static:
        return Universe(static, "static", "demo", "static")
    scfg = scanner_cfg(cfg.symbols_cfg)
    if not scfg.get("enabled"):
        return Universe(static, "static", "symbols.yaml", "static")

    get = http_get or requests.get
    env = dict(os.environ) if env is None else env
    timeout = int(cfg.symbols_cfg.get("data", {}).get("timeout_seconds", 20))
    rhash = rules_hash(cfg.symbols_cfg)
    top_n = int(scfg["top_n"])
    cache_path, hist_path = Path(out_dir) / "universe.json", Path(out_dir) / "universe_history.csv"
    cache = _read_cache(cache_path)
    cache_ok = bool(cache and cache.get("rules_hash") == rhash and cache.get("symbols"))

    def from_cache(status: str, warnings: List[str]) -> Universe:
        return Universe(_finish(cache["symbols"], scfg, bench), "scanner", cache["source"], status, cache.get("fetched_at"),
                        cache.get("ranks", {}), cache.get("volumes_24h_usdt", {}), rhash, top_n, warnings)

    if cache_ok:
        try:
            age_h = (now - pd.Timestamp(cache["fetched_at"])).total_seconds() / 3600.0
        except (KeyError, ValueError):
            age_h = float("inf")
        if 0 <= age_h < float(scfg["refresh_hours"]):
            return from_cache("cached", [])

    warnings: List[str] = []
    for source in scfg["sources"]:
        try:
            rows = _build(source, scfg, get, timeout, env)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{source}: {exc}")
            log.warning("scanner %s gagal: %s", source, exc)
            continue
        base_syms = [f"{b}/{scfg['quote']}" for b, _ in rows]
        ts = now.isoformat()
        payload = {"rules_hash": rhash, "source": source, "fetched_at": ts, "top_n": top_n, "symbols": base_syms,
                   "ranks": {s: i + 1 for i, s in enumerate(base_syms)},
                   "volumes_24h_usdt": {f"{b}/{scfg['quote']}": round(v, 2) for b, v in rows}}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            _append_history(hist_path, ts, source, rhash, base_syms, cache["symbols"] if cache_ok else None)
        except OSError as exc:
            warnings.append(f"cache universe tidak bisa ditulis: {exc}")
        log.info("Universe di-refresh dari %s: %d simbol", source, len(base_syms))
        return Universe(_finish(base_syms, scfg, bench), "scanner", source, "fresh", ts, payload["ranks"],
                        payload["volumes_24h_usdt"], rhash, top_n, warnings)

    if cache_ok:
        warnings.append("semua sumber gagal -> memakai universe cache terakhir")
        return from_cache("stale_cache", warnings)
    warnings.append("semua sumber gagal & tidak ada cache -> memakai daftar statis symbols.yaml")
    return Universe(_finish(static, scfg, bench), "scanner", "static_fallback", "fallback_static", None, {}, {}, rhash, top_n, warnings)
