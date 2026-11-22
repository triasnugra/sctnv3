"""Orkestrasi analisis satu simbol: data multi-TF -> SymbolAnalysis (semua temuan, belum jadi signal)."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from ..config_loader import Config
from ..data.indicators import last_atr, volume_ratio
from ..models import Finding, FibSetup, FVGZone, Level, RangeBox, SymbolAnalysis
from .breakout_retest import detect_breakout_retests
from .fibonacci import detect_fib
from .fvg import detect_fvgs
from .liquidity_deviation import detect_deviations
from .monday_range import compute_monday_range, monday_finding, monday_levels
from .multi_tf_bias import compute_bias
from .obv_confirmation import detect_obv
from .range_detector import detect_range, range_finding
from .three_tap_wolf import detect_three_tap, detect_wolf

log = logging.getLogger("engine.pipeline")


def _range_levels(rb: RangeBox) -> List[Level]:
    return [Level(f"Range High ({rb.tf})", rb.high, "range_high", rb.tf, ("up",)),
            Level(f"Range Low ({rb.tf})", rb.low, "range_low", rb.tf, ("down",))]


def analyze_symbol(symbol: str, data: Dict[str, pd.DataFrame], cfg: Config, now: pd.Timestamp) -> SymbolAnalysis:
    tfs = cfg.tfs
    ind = cfg.params.get("indicators", {})
    atr_p = int(ind.get("atr_period", 14))
    notes: List[str] = []

    trig_tfs: List[str] = list(tfs["trigger"])
    finest = min(trig_tfs, key=lambda t: pd.Timedelta(t.replace("m", "min")) if t.endswith("m") else pd.Timedelta(t))
    price = float(data[finest]["close"].iloc[-1])

    bias = compute_bias(data["1w"], data["1d"], price, cfg.params["bias"])
    monday = compute_monday_range(data["1d"], now, cfg.params.get("monday_range", {}))
    if monday is None:
        notes.append("Monday range tidak tersedia (data daily kurang).")

    atrs = {tf: last_atr(df, atr_p) for tf, df in data.items()}
    vr_cfg = cfg.risk.get("filters", {}).get("low_volume", {})
    vol_ratios = {tf: volume_ratio(df, int(vr_cfg.get("recent_candles", 3)), int(vr_cfg.get("baseline_candles", 50)))
                  for tf, df in data.items()}

    ranges: Dict[str, RangeBox] = {}
    for tf in cfg.params["range"].get("timeframes", []):
        if tf in data:
            rb = detect_range(data[tf], cfg.tf_params("range", tf), tf)
            if rb:
                ranges[tf] = rb
    if not ranges:
        notes.append("Tidak ada range valid pada TF terkonfigurasi -> tidak ada level range (default: NO TRADE).")

    levels: List[Level] = []
    for rb in ranges.values():
        levels += _range_levels(rb)
    levels += monday_levels(monday)

    findings: List[Finding] = []
    for tf, rb in ranges.items():
        findings.append(range_finding(rb, price))
    mf = monday_finding(monday, price)
    if mf:
        findings.append(mf)

    fvgs: List[FVGZone] = []
    fibs: List[FibSetup] = []
    for tf in trig_tfs:
        df = data[tf]
        try:
            findings += detect_deviations(df, levels, cfg.tf_params("deviation", tf), tf)
            findings += detect_breakout_retests(df, levels, cfg.tf_params("breakout_retest", tf), atrs[tf], tf)
            fs, ff = detect_fib(df, cfg.tf_params("fibonacci", tf), tf)
            if fs:
                fibs.append(fs)
            if ff:
                findings.append(ff)
            findings += detect_three_tap(df, cfg.tf_params("three_tap", tf), tf)
            findings += detect_wolf(df, cfg.tf_params("wolf", tf), tf)
            zs, fz = detect_fvgs(df, cfg.tf_params("fvg", tf), atrs[tf], tf)
            fvgs += zs
            findings += fz
            findings += detect_obv(df, cfg.tf_params("obv", tf), tf)
        except Exception as exc:  # noqa: BLE001 - satu detektor gagal tidak boleh mematikan simbol
            log.exception("%s %s: detektor gagal: %s", symbol, tf, exc)
            notes.append(f"Detektor {tf} error: {exc}")

    return SymbolAnalysis(symbol=symbol, asof=now, price=price, bias=bias, monday=monday, ranges=ranges,
                          levels=levels, findings=findings, fvgs=fvgs, fibs=fibs, atr=atrs,
                          vol_ratio=vol_ratios, frames=data, notes=notes)
