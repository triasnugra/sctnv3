"""Range / Box: tepi range (supply/demand edge) dari klaster pivot.
Hipotesis: range valid bila >= min_touches pivot menyentuh masing-masing tepi (toleransi %),
lebar dalam batas, harga tidak trending (drift kecil), dan hampir semua close di dalam box."""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from ..data.indicators import find_pivots
from ..models import Finding, RangeBox
from ..utils import fmt_price

log = logging.getLogger("engine.range")


def _best_cluster(prices: List[float], tol: float, high: bool) -> List[float]:
    best: List[float] = []
    best_key = None
    for t in prices:
        cl = [x for x in prices if abs(x - t) <= t * tol]
        key = (len(cl), t if high else -t)
        if best_key is None or key > best_key:
            best_key, best = key, cl
    return best


def detect_range(df: pd.DataFrame, p: dict, tf: str) -> Optional[RangeBox]:
    ex = int(p.get("exclude_recent", 0))
    base = df.iloc[:-ex] if ex > 0 else df
    tol = float(p["touch_tol_pct"]) / 100
    n = int(p["pivot_n"])
    best, best_key = None, None
    for w in p["windows"]:
        if len(base) < w:
            continue
        seg = base.iloc[-w:]
        piv = find_pivots(seg, n, n)
        hs = [x.price for x in piv if x.kind == "H"]
        ls = [x.price for x in piv if x.kind == "L"]
        if len(hs) < p["min_touches"] or len(ls) < p["min_touches"]:
            continue
        ch, cl = _best_cluster(hs, tol, True), _best_cluster(ls, tol, False)
        if len(ch) < p["min_touches"] or len(cl) < p["min_touches"]:
            continue
        top, bot = max(ch), min(cl)
        if top <= bot or bot <= 0:
            continue
        width = (top - bot) / bot * 100
        if not (p["min_width_pct"] <= width <= p["max_width_pct"]):
            continue
        closes = seg["close"]
        outside = ((closes > top * (1 + tol)) | (closes < bot * (1 - tol))).mean()
        if outside > p.get("max_outside_close_frac", 0.03):
            continue
        drift = abs(closes.iloc[-1] - closes.iloc[0]) / (top - bot)
        if drift > p["max_drift_ratio"]:
            continue
        key = (len(ch) + len(cl), w)
        if best_key is None or key > best_key:
            best_key = key
            best = RangeBox(tf=tf, high=float(top), low=float(bot), touches_high=len(ch), touches_low=len(cl),
                            start_time=seg.index[0], end_time=df.index[-1],
                            edge_zone_frac=float(p.get("edge_zone_frac", 0.2)))
    if best is None:
        log.debug("Range %s: tidak ada range valid", tf)
    return best


def range_finding(rb: RangeBox, price: float) -> Finding:
    pos = rb.position(price)
    txt = f"Range {rb.tf}: {fmt_price(rb.low)} – {fmt_price(rb.high)} (lebar {rb.width_pct:.1f}%, " \
          f"sentuhan {rb.touches_low}L/{rb.touches_high}H)"
    if pos == "demand_edge":
        return Finding("range_edge", "context", "LONG", rb.tf, 0.5, txt + " — harga di DEMAND edge", state=pos)
    if pos == "supply_edge":
        return Finding("range_edge", "context", "SHORT", rb.tf, 0.5, txt + " — harga di SUPPLY edge", state=pos)
    if pos == "mid":
        return Finding("range_edge", "context", "NEUTRAL", rb.tf, 0.0, txt + " — harga di TENGAH range (no-trade zone)", state=pos)
    return Finding("range_edge", "context", "NEUTRAL", rb.tf, 0.0,
                   txt + f" — harga DI LUAR range ({pos}); pantau deviation/breakout-retest", state=pos)
