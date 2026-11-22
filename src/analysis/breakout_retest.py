"""Breakout + Retest / Flip.
Definisi operasional (hipotesis):
 1. Candle breakout: close melewati level >= break_pct, body >= min_body_atr x ATR,
    volume >= min_volume_mult x SMA volume.
 2. Akseptasi: accept_bars candle berikut tetap close di sisi baru (kalau tidak -> itu deviation, bukan breakout).
 3. Retest: harga kembali ke zona level (retest_zone_pct) tanpa close menembus balik > invalid_pct.
 4. Entry di retest (level flip), stop di sisi lain level (di bawah retest low / level)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.indicators import vol_sma_prior
from ..models import Finding, Level
from ..utils import fmt_price, fmt_ts
from .common import classify_state


def _scan(df: pd.DataFrame, lv: Level, orient: str, p: dict, atr_val: float, tf: str) -> Optional[Finding]:
    n = len(df)
    if n < 30:
        return None
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    vsma = vol_sma_prior(v)
    up = orient == "up"
    sgn = 1 if up else -1
    L = lv.price
    bpct = L * p["break_pct"] / 100
    best = None
    for b in range(max(1, n - int(p["scan_bars"])), n):
        if (c[b] - L) * sgn < bpct:
            continue
        if (c[b - 1] - L) * sgn > bpct / 2:
            continue  # sudah di luar sebelumnya
        if abs(c[b] - o[b]) < p["min_body_atr"] * atr_val:
            continue
        ref = vsma[b]
        if np.isnan(ref) or v[b] < p["min_volume_mult"] * ref:
            continue
        end = min(n, b + 1 + int(p["accept_bars"]))
        if any((c[m] - L) * sgn <= 0 for m in range(b + 1, end)):
            continue  # gagal bertahan -> kandidat deviation
        best = b
    if best is None:
        return None
    b = best
    inv_d = L * p["invalid_pct"] / 100
    invalid = any((c[m] - L) * sgn < -inv_d for m in range(b + 1, n))
    z = L * p["entry_zone_pct"] / 100
    zone = (L - z, L + z)
    rz = L * p["retest_zone_pct"] / 100
    retest_idx = None
    for m in range(b + 1, n):
        touched = (l[m] <= L + rz) if up else (h[m] >= L - rz)
        if touched and (c[m] - L) * sgn >= -inv_d:
            retest_idx = m
            break
    direction = "LONG" if up else "SHORT"
    if n - 1 > b:
        stop_ref = min(L, float(l[b + 1:].min())) if up else max(L, float(h[b + 1:].max()))
    else:
        stop_ref = L
    state = "invalidated" if invalid else classify_state(direction, float(c[-1]), zone, retest_idx is not None)
    vol_x = v[b] / ref
    strength = 0.5 + 0.2 * (vol_x >= 2.0) + 0.15 * (retest_idx is not None) + 0.15 * (abs(c[b] - o[b]) >= atr_val)
    side = "atas" if up else "bawah"
    detail = (f"Breakout {side} {lv.name} {fmt_price(L)} @ {fmt_ts(df.index[b])} "
              f"(vol {vol_x:.1f}x, body {abs(c[b] - o[b]) / atr_val:.1f} ATR, bertahan {int(p['accept_bars'])} candle)"
              f"{'; retest ' + fmt_ts(df.index[retest_idx]) if retest_idx is not None else '; retest belum terjadi'}. "
              f"Flip level jadi {'support' if up else 'resistance'}; entry {fmt_price(L)}, invalidasi {fmt_price(stop_ref)}. "
              f"Status: {state}")
    return Finding(kind="breakout_retest", role="trigger", direction=direction, tf=tf, strength=float(strength),
                   detail=detail, level=L, entry=L, entry_zone=zone, stop_ref=float(stop_ref), state=state,
                   time=df.index[retest_idx if retest_idx is not None else b],
                   meta={"level_name": lv.name, "level_kind": lv.kind, "level_tf": lv.tf,
                         "vol_spike": bool(vol_x >= 1.5), "breakout_time": df.index[b],
                         "retested": retest_idx is not None})


def detect_breakout_retests(df: pd.DataFrame, levels: List[Level], p: dict, atr_val: float, tf: str) -> List[Finding]:
    out: List[Finding] = []
    for lv in levels:
        for orient in lv.orientations:
            f = _scan(df, lv, orient, p, atr_val, tf)
            if f:
                out.append(f)
    return out
