"""Fair Value Gap (3 candle). Hanya KONFIRMASI tambahan, bukan trigger.
Bullish FVG: low[i] > high[i-2]; Bearish FVG: high[i] < low[i-2]; ukuran >= min_gap_atr_mult x ATR.
Zona 'open' bila belum terisi penuh; terisi sebagian menyempitkan zona."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ..models import FVGZone, Finding
from ..utils import fmt_price, fmt_ts


def detect_fvgs(df: pd.DataFrame, p: dict, atr_val: float, tf: str) -> Tuple[List[FVGZone], List[Finding]]:
    n = len(df)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    tol = p["touch_tol_pct"] / 100
    tl = int(p["touch_lookback"])
    min_gap = p["min_gap_atr_mult"] * atr_val
    zones: List[FVGZone] = []
    for i in range(max(2, n - int(p["max_age_bars"])), n):
        # bullish
        gap = l[i] - h[i - 2]
        if gap > 0 and gap >= min_gap:
            lo, hi, filled, last_touch = h[i - 2], l[i], False, -1
            for m in range(i + 1, n):
                if l[m] <= lo:
                    filled = True
                    break
                if l[m] < hi:
                    hi, last_touch = l[m], m
            if not filled:
                zones.append(FVGZone(tf, "LONG", float(lo), float(hi), df.index[i], last_touch >= n - tl))
        # bearish
        gap = l[i - 2] - h[i]
        if gap > 0 and gap >= min_gap:
            lo, hi, filled, last_touch = h[i], l[i - 2], False, -1
            for m in range(i + 1, n):
                if h[m] >= hi:
                    filled = True
                    break
                if h[m] > lo:
                    lo, last_touch = h[m], m
            if not filled:
                zones.append(FVGZone(tf, "SHORT", float(lo), float(hi), df.index[i], last_touch >= n - tl))
    findings: List[Finding] = []
    price = float(c[-1])
    for d in ("LONG", "SHORT"):
        for z in sorted([z for z in zones if z.direction == d], key=lambda z: z.time, reverse=True):
            near = z.low * (1 - tol) <= price <= z.high * (1 + tol)
            reacted = z.touched_recent and ((price >= z.low) if d == "LONG" else (price <= z.high))
            if near or reacted:
                findings.append(Finding("fvg", "confirmation", d, tf, 0.4,
                                        f"{'Bullish' if d == 'LONG' else 'Bearish'} FVG {tf} {fmt_price(z.low)}–{fmt_price(z.high)} "
                                        f"(terbentuk {fmt_ts(z.time)}) belum terisi penuh; harga bereaksi di zona",
                                        level=(z.low + z.high) / 2, time=z.time,
                                        meta={"low": z.low, "high": z.high}))
                break
    return zones, findings
