"""Pengambilan data OHLCV: Binance REST (multi-host) + ccxt fallback.
Prinsip: gagal ambil data -> raise DataFetchError (ditangani per simbol di main), tidak crash seluruh run."""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import pandas as pd
import requests

from .indicators import tf_to_timedelta

log = logging.getLogger("engine.data")

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataFetchError(Exception):
    pass


def drop_unclosed(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Buang candle yang periodenya belum selesai (candle berjalan)."""
    td = tf_to_timedelta(tf)
    return df[(df.index + td) <= now]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df["high"] >= df["low"]) & (df["volume"] >= 0) & (df["close"] > 0)]
    return df


def _rows_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts"] + COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


class BinanceRestProvider:
    name = "binance_rest"
    HOSTS = [
        "https://api.binance.com",
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self._good_host: str | None = None

    def fetch(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        """Mendukung limit > 1000 lewat paging mundur (endTime)."""
        if limit <= 1000:
            return self._fetch_once(symbol, tf, limit, None)
        chunks, end, remaining = [], None, limit
        for _ in range(60):
            want = min(1000, remaining)
            df = self._fetch_once(symbol, tf, want, end)
            chunks.append(df)
            remaining -= len(df)
            if remaining <= 0 or len(df) < want:
                break
            end = int(df.index[0].timestamp() * 1000) - 1
        return pd.concat(chunks[::-1])

    def _fetch_once(self, symbol: str, tf: str, limit: int, end_ms) -> pd.DataFrame:
        sym = symbol.replace("/", "")
        hosts = ([self._good_host] if self._good_host else []) + [h for h in self.HOSTS if h != self._good_host]
        last_err = "tidak ada host dicoba"
        for host in hosts:
            try:
                r = requests.get(f"{host}/api/v3/klines",
                                 params={"symbol": sym, "interval": tf, "limit": min(limit, 1000),
                                         **({"endTime": end_ms} if end_ms else {})},
                                 timeout=self.timeout)
                if r.status_code in (451, 403):
                    last_err = f"{host}: HTTP {r.status_code} (geo-block?)"
                    continue
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list) or not data:
                    last_err = f"{host}: respons kosong"
                    continue
                self._good_host = host
                rows = [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in data]
                return _rows_to_df(rows)
            except Exception as exc:  # noqa: BLE001
                last_err = f"{host}: {exc}"
        raise DataFetchError(f"binance_rest gagal untuk {symbol} {tf}: {last_err}")


class CcxtProvider:
    def __init__(self, exchange_id: str, timeout: int = 20):
        self.name = f"ccxt:{exchange_id}"
        try:
            import ccxt  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise DataFetchError("ccxt tidak terpasang") from exc
        if not hasattr(ccxt, exchange_id):
            raise DataFetchError(f"Exchange ccxt tidak dikenal: {exchange_id}")
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": timeout * 1000})

    def fetch(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        try:
            if limit <= 1000:
                rows = self.ex.fetch_ohlcv(symbol, tf, limit=limit)
            else:  # paging maju dari 'since'
                tf_ms = int(tf_to_timedelta(tf).total_seconds() * 1000)
                since = int(time.time() * 1000) - (limit + 5) * tf_ms
                rows = []
                for _ in range(80):
                    chunk = self.ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                    if not chunk:
                        break
                    rows += chunk
                    nxt = chunk[-1][0] + tf_ms
                    if nxt <= since or len(chunk) < 2:
                        break
                    since = nxt
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"{self.name} gagal untuk {symbol} {tf}: {exc}") from exc
        if not rows:
            raise DataFetchError(f"{self.name}: data kosong {symbol} {tf}")
        return _rows_to_df([r[:6] for r in rows])


def build_provider(spec: str, timeout: int):
    if spec == "binance_rest":
        return BinanceRestProvider(timeout)
    if spec.startswith("ccxt:"):
        return CcxtProvider(spec.split(":", 1)[1], timeout)
    raise DataFetchError(f"Provider tidak dikenal: {spec}")


class OHLCVFetcher:
    def __init__(self, providers: List[str], retries: int = 3, backoff: float = 2.0,
                 timeout: int = 20, drop_unclosed_candle: bool = True):
        self.specs = providers
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.drop_unclosed_candle = drop_unclosed_candle
        self._providers: Dict[str, object] = {}

    def _get_provider(self, spec: str):
        if spec not in self._providers:
            self._providers[spec] = build_provider(spec, self.timeout)
        return self._providers[spec]

    def _fetch_one(self, prov, symbol: str, tf: str, limit: int, now: pd.Timestamp) -> pd.DataFrame:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                df = _clean(prov.fetch(symbol, tf, limit))
                if self.drop_unclosed_candle:
                    df = drop_unclosed(df, tf, now)
                if df.empty:
                    raise DataFetchError("data kosong setelah pembersihan")
                return df
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("%s %s %s percobaan %d/%d gagal: %s", getattr(prov, "name", "?"), symbol, tf,
                            attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise DataFetchError(str(last))

    def fetch_symbol(self, symbol: str, candles: Dict[str, int], now: pd.Timestamp | None = None,
                     min_candles: Dict[str, int] | None = None) -> Dict[str, pd.DataFrame]:
        now = now or pd.Timestamp.now(tz="UTC")
        min_candles = min_candles or {}
        errors: List[str] = []
        for spec in self.specs:
            try:
                prov = self._get_provider(spec)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec}: {exc}")
                continue
            try:
                out: Dict[str, pd.DataFrame] = {}
                for tf, limit in candles.items():
                    df = self._fetch_one(prov, symbol, tf, limit, now)
                    need = int(min_candles.get(tf, 0))
                    if len(df) < need:
                        raise DataFetchError(f"{tf}: hanya {len(df)} candle (< {need})")
                    out[tf] = df
                log.info("%s: data lengkap dari %s", symbol, spec)
                return out
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec}: {exc}")
                log.warning("%s: provider %s gagal -> coba berikutnya", symbol, spec)
        raise DataFetchError(f"Semua provider gagal untuk {symbol}: " + " | ".join(errors))
