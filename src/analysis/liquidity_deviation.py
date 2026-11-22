"""Liquidity / Stop-run -> Deviation + Reclaim (konfirmasi terkuat).
Definisi operasional (hipotesis):
 1. Wick menembus level >= min_dev_pct, dan close candle sebelumnya masih di sisi 'dalam'.
 2. Dalam <= reclaim_bars candle ada close kembali di sisi dalam (reclaim).
 3. Penetrasi maksimum <= max_dev_pct (lebih dalam = breakout asli, bukan deviasi).
 4. Entry = retest level (zona +/- entry_zone_pct); stop = di luar titik ekstrem deviasi.
 5. Invalid bila ada close melewati titik ekstrem deviasi setelah reclaim."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.indicators import vol_sma_prior
from ..models import Finding, Level
from ..utils import fmt_price, fmt_ts
from .common import classify_state


def _scan(df: pd.DataFrame, lv: Level, orient: str, p: dict, tf: str) -> Optional[Finding]:
    n = len(df)
    if n < 30:
        return None
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    vsma = vol_sma_prior(v)
    up = orient == "up"
    sgn = 1 if up else -1
    L = lv.price
    min_dev, max_dev = L * p["min_dev_pct"] / 100, L * p["max_dev_pct"] / 100
    found = None
    for j in range(max(1, n - int(p["scan_bars"])), n):
        ext_j = h[j] if up else l[j]
        if (ext_j - L) * sgn < min_dev:
            continue
        if (c[j - 1] - L) * sgn > 0:
            continue  # candle sebelumnya sudah close di luar -> bukan deviasi segar
        extreme, k = ext_j, None
        for m in range(j, min(n, j + int(p["reclaim_bars"]) + 1)):
            e = h[m] if up else l[m]
            extreme = max(extreme, e) if up else min(extreme, e)
            if (c[m] - L) * sgn < 0:
                k = m
                break
        if k is None or (extreme - L) * sgn > max_dev:
            continue
        found = (j, k, extreme)
    if found is None:
        return None
    j, k, extreme = found
    z = L * p["entry_zone_pct"] / 100
    zone = (L - z, L + z)
    direction = "SHORT" if up else "LONG"
    invalid = any((c[m] - extreme) * sgn > 0 for m in range(k + 1, n))
    retested = any((h[m] >= zone[0]) if up else (l[m] <= zone[1]) for m in range(k + 1, n))
    state = "invalidated" if invalid else classify_state(direction, float(c[-1]), zone, retested)

    ref = vsma[j]
    vol_spike = bool((not np.isnan(ref)) and ref > 0 and v[j:k + 1].max() >= p["vol_spike_mult"] * ref)
    rng_k = h[k] - l[k]
    cpos = (c[k] - l[k]) / rng_k if rng_k > 0 else 0.5
    good_close = cpos <= 0.4 if up else cpos >= 0.6
    strength = 0.5 + 0.2 * vol_spike + 0.15 * good_close + 0.15 * (k == j)
    dev_pct = abs(extreme - L) / L * 100
    side = "atas" if up else "bawah"
    detail = (f"Deviation di {side} {lv.name} {fmt_price(L)}: wick ke {fmt_price(extreme)} (+{dev_pct:.2f}%), "
              f"reclaim close {fmt_price(float(c[k]))} @ {fmt_ts(df.index[k])} "
              f"({k - j + 1} candle di luar){' + volume spike' if vol_spike else ''}. "
              f"Entry di retest {fmt_price(L)}, invalidasi {fmt_price(extreme)}. Status: {state}")
    return Finding(kind="deviation_reclaim", role="trigger", direction=direction, tf=tf, strength=float(strength),
                   detail=detail, level=L, entry=L, entry_zone=zone, stop_ref=float(extreme), state=state,
                   time=df.index[k],
                   meta={"level_name": lv.name, "level_kind": lv.kind, "level_tf": lv.tf, "extreme": float(extreme),
                         "dev_pct": float(dev_pct), "vol_spike": vol_spike, "retested": retested,
                         "deviation_time": df.index[j]})


def detect_deviations(df: pd.DataFrame, levels: List[Level], p: dict, tf: str) -> List[Finding]:
    out: List[Finding] = []
    for lv in levels:
        for orient in lv.orientations:
            f = _scan(df, lv, orient, p, tf)
            if f:
                out.append(f)
    return out
