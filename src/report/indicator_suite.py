"""Suite indikator untuk GRAFIK & ringkasan dashboard (sctnv2).

MACD · OBV · RSI · EMA 12/26/50/100/200 · Supertrend · UCTS (proksi) · Spaceman BTC Key Levels · Volume (+ profile).

PENTING: modul ini TAMPILAN/INFORMASI SAJA. Tidak ada angka di sini yang masuk skor, veto, risk plan,
atau config_hash — jadi logika signal & forward test tetap identik dengan sctn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.indicators import ema, macd, obv, rsi, sma, supertrend, volume_profile
from ..data.key_levels import compute_key_levels
from ..models import SymbolAnalysis

log = logging.getLogger("engine.indicators")

NOTE = ("Informasi saja — indikator TIDAK masuk skor/veto/signal. UCTS≈ adalah proksi terbuka, "
        "bukan sinyal UCTS asli (skrip invite-only).")


@dataclass
class Suite:
    tf: str
    df: pd.DataFrame
    ema: Dict[int, pd.Series] = field(default_factory=dict)
    st: Optional[pd.Series] = None
    st_dir: Optional[pd.Series] = None
    rsi: Optional[pd.Series] = None
    macd: Optional[pd.Series] = None
    macd_sig: Optional[pd.Series] = None
    macd_hist: Optional[pd.Series] = None
    obv: Optional[pd.Series] = None
    vol_sma: Optional[pd.Series] = None
    ucts_buy: Optional[pd.Series] = None
    ucts_sell: Optional[pd.Series] = None


def compute_suite(df: pd.DataFrame, ci: Dict[str, Any], tf: str = "") -> Suite:
    """Hitung semua indikator di SELURUH frame (agar EMA200 dkk sudah 'hangat'); pemotongan jendela dilakukan saat menggambar."""
    s = Suite(tf=tf, df=df)
    c = df["close"]
    if ci.get("ema", {}).get("enabled", True):
        for p in ci["ema"]["periods"]:
            s.ema[int(p)] = ema(c, int(p))
    st_cfg = ci.get("supertrend", {})
    if st_cfg.get("enabled", True):
        s.st, s.st_dir = supertrend(df, int(st_cfg.get("period", 10)), float(st_cfg.get("multiplier", 3.0)))
    s.rsi = rsi(c, int(ci["rsi"]["period"]))
    s.macd, s.macd_sig, s.macd_hist = macd(c, int(ci["macd"]["fast"]), int(ci["macd"]["slow"]), int(ci["macd"]["signal"]))
    s.obv = obv(df)
    s.vol_sma = sma(df["volume"], int(ci.get("volume", {}).get("sma", 20)))
    u = ci.get("ucts_proxy", {})
    if u.get("enabled", True) and s.st_dir is not None:
        d = s.st_dir
        flip_up = (d == 1) & (d.shift(1) == -1)
        flip_dn = (d == -1) & (d.shift(1) == 1)
        rising = s.macd_hist > s.macd_hist.shift(1)
        falling = s.macd_hist < s.macd_hist.shift(1)
        buy, sell = flip_up & rising, flip_dn & falling
        if u.get("use_ema_filter", False):
            f = ema(c, int(u.get("ema_filter_period", 50)))
            buy, sell = buy & (c > f), sell & (c < f)
        s.ucts_buy, s.ucts_sell = buy.fillna(False), sell.fillna(False)
    return s


def build_suites(an: SymbolAnalysis, cfg) -> Dict[str, Suite]:
    ci = cfg.chart_ind
    need = set(ci.get("summary_timeframes", [])) | set(ci.get("mtf_timeframes", [])) | set(ci.get("extra_timeframes", []))
    need.add(cfg.params.get("chart", {}).get("timeframe", "1h"))
    need.add(cfg.tfs["primary_trigger"])
    out: Dict[str, Suite] = {}
    for tf in need:
        df = an.frames.get(tf)
        if df is None or len(df) < 30:
            continue
        try:
            out[tf] = compute_suite(df, ci, tf)
        except Exception as exc:  # noqa: BLE001 - indikator tampilan tidak boleh mematikan analisis
            log.warning("%s %s: indikator gagal: %s", an.symbol, tf, exc)
    return out


# ------------------------------------------------------------------ ringkasan pembacaan
def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _macd_state(s: Suite) -> str:
    m, g = s.macd, s.macd_sig
    now = m.iloc[-1] - g.iloc[-1]
    for k in (1, 2, 3):
        if len(m) > k + 1:
            prev = m.iloc[-1 - k] - g.iloc[-1 - k]
            if now > 0 >= prev:
                return f"cross naik ({k} bar lalu)"
            if now < 0 <= prev:
                return f"cross turun ({k} bar lalu)"
    return "di atas signal" if now > 0 else "di bawah signal"


def _ema_stack(vals: Dict[int, float]) -> str:
    ps = sorted(vals)
    if len(ps) < 2:
        return "-"
    seq = [vals[p] for p in ps]
    if all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
        return "bullish (EMA pendek di atas EMA panjang)"
    if all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
        return "bearish (EMA pendek di bawah EMA panjang)"
    return "campuran"


def summarize_tf(s: Suite, ci: Dict[str, Any], vp_bars: int = 160) -> Dict[str, Any]:
    df = s.df
    close = float(df["close"].iloc[-1])
    out: Dict[str, Any] = {"close": close}
    r = _f(s.rsi.iloc[-1])
    ob, os_ = ci["rsi"]["overbought"], ci["rsi"]["oversold"]
    out["rsi"] = {"value": r, "zone": None if r is None else ("overbought" if r >= ob else "oversold" if r <= os_ else "netral")}
    out["macd"] = {"line": _f(s.macd.iloc[-1]), "signal": _f(s.macd_sig.iloc[-1]), "hist": _f(s.macd_hist.iloc[-1]),
                   "state": _macd_state(s), "zero": "di atas 0" if s.macd.iloc[-1] > 0 else "di bawah 0"}
    if s.ema:
        vals = {p: _f(v.iloc[-1]) for p, v in s.ema.items()}
        vals = {p: v for p, v in vals.items() if v is not None}
        out["ema"] = {"values": {str(p): v for p, v in vals.items()}, "stack": _ema_stack(vals)}
        if 200 in vals:
            out["ema"]["vs_200"] = "di atas EMA200" if close > vals[200] else "di bawah EMA200"
    if s.st is not None and s.st_dir is not None:
        d = int(s.st_dir.iloc[-1])
        stv = _f(s.st.iloc[-1])
        flips = np.flatnonzero((s.st_dir.to_numpy()[1:] != s.st_dir.to_numpy()[:-1]) & (s.st_dir.to_numpy()[:-1] != 0))
        ago = int(len(df) - 2 - flips[-1]) if len(flips) else None
        out["supertrend"] = {"dir": d, "value": stv, "dist_pct": None if stv is None else (close - stv) / close * 100.0,
                             "last_flip_bars_ago": ago}
    o = s.obv.to_numpy()
    if len(o) > 25:
        avgv = float(df["volume"].tail(50).mean())
        delta = float(o[-1] - o[-21])
        out["obv"] = {"trend": "naik" if delta > 0 else "turun", "delta_vs_avg_volume": None if avgv <= 0 else delta / avgv}
    vs = _f(s.vol_sma.iloc[-1])
    lastv = _f(df["volume"].iloc[-1])
    if vs and lastv is not None:
        ratio = lastv / vs
        out["volume"] = {"ratio_vs_sma": ratio, "state": "tinggi" if ratio >= 1.5 else "rendah" if ratio < 0.5 else "normal"}
    if s.ucts_buy is not None:
        last = None
        for side, ser in (("BUY", s.ucts_buy), ("SELL", s.ucts_sell)):
            idx = np.flatnonzero(ser.to_numpy())
            if len(idx) and (last is None or idx[-1] > last["i"]):
                last = {"i": int(idx[-1]), "side": side}
        out["ucts_proxy"] = None if last is None else {
            "side": last["side"], "bars_ago": int(len(df) - 1 - last["i"]), "time": df.index[last["i"]].isoformat()}
    vcfg = ci.get("volume_profile", {})
    if vcfg.get("enabled", True):
        vp = volume_profile(df.iloc[-vp_bars:], int(vcfg.get("bins", 48)), float(vcfg.get("value_area", 0.70)))
        if vp:
            pos = "di atas VAH" if close > vp["vah"] else "di bawah VAL" if close < vp["val"] else "di dalam value area"
            out["volume_profile"] = {"poc": vp["poc"], "vah": vp["vah"], "val": vp["val"], "position": pos,
                                     "dist_poc_pct": (close - vp["poc"]) / close * 100.0, "bars": min(vp_bars, len(df))}
    return out


def summarize_symbol(an: SymbolAnalysis, cfg, suites: Dict[str, Suite]) -> Dict[str, Any]:
    ci = cfg.chart_ind
    vp_bars = int(cfg.params.get("chart", {}).get("candles", 160))
    tfs: Dict[str, Any] = {}
    for tf in ci.get("summary_timeframes", []):
        if tf in suites:
            try:
                tfs[tf] = summarize_tf(suites[tf], ci, vp_bars)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s %s: ringkasan indikator gagal: %s", an.symbol, tf, exc)
    mtf = {}
    for tf in ci.get("mtf_timeframes", []):
        s = suites.get(tf)
        if s is not None and s.st_dir is not None and int(s.st_dir.iloc[-1]) != 0:
            mtf[tf] = int(s.st_dir.iloc[-1])
    kl: List[Dict[str, Any]] = []
    if ci.get("key_levels", {}).get("enabled", True):
        for k in compute_key_levels(an.frames, an.asof, an.monday, ci["key_levels"].get("show")):
            kl.append({"group": k.group, "kind": k.kind, "label": k.label, "name": k.name, "price": k.price,
                       "dist_pct": (k.price - an.price) / an.price * 100.0})
        kl.sort(key=lambda x: -x["price"])
    return {"timeframes": tfs, "mtf_supertrend": mtf, "key_levels": kl, "primary_tf": cfg.tfs["primary_trigger"], "note": NOTE}
