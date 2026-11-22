"""Fibonacci retracement fokus 0.75 dari impulse terakhir.
Fib = LEVEL, bukan trigger. Trigger hanya jika harga di zona (0.71–0.79) DAN ada reaksi candle
(candle terakhir searah trade). Tanpa reaksi -> hanya konteks (limit di fib = prediksi, tidak dipakai)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.indicators import find_pivots
from ..models import FibSetup, Finding
from ..utils import fmt_price, fmt_ts
from .common import classify_state


def detect_fib(df: pd.DataFrame, p: dict, tf: str) -> Tuple[Optional[FibSetup], Optional[Finding]]:
    seg = df.iloc[-int(p["pivot_lookback"]):]
    n = len(seg)
    if n < 30:
        return None, None
    h, l, c, o = (seg[k].to_numpy() for k in ("high", "low", "close", "open"))
    pn = int(p["pivot_n"])
    piv = find_pivots(seg, pn, pn)
    cands = []
    for pv in [x for x in piv if x.kind == "L"][-6:]:  # impulse naik: low -> high
        a = pv.idx
        if n - a < 3:
            continue
        b = a + 1 + int(np.argmax(h[a + 1:]))
        leg = h[b] - pv.price
        if leg / pv.price * 100 < p["min_impulse_pct"] or (c[a + 1:] < pv.price).any():
            continue
        cands.append(("LONG", a, float(pv.price), b, float(h[b])))
    for pv in [x for x in piv if x.kind == "H"][-6:]:  # impulse turun: high -> low
        a = pv.idx
        if n - a < 3:
            continue
        b = a + 1 + int(np.argmin(l[a + 1:]))
        leg = pv.price - l[b]
        if leg / pv.price * 100 < p["min_impulse_pct"] or (c[a + 1:] > pv.price).any():
            continue
        cands.append(("SHORT", a, float(pv.price), b, float(l[b])))
    if not cands:
        return None, None
    direction, a, ap, b, bp = max(cands, key=lambda x: (x[3], abs(x[4] - x[2])))
    leg = abs(bp - ap)
    lvl, hw, inv = float(p["level"]), float(p["zone_half_width"]), float(p["invalid_retrace"])
    sgn = 1 if direction == "LONG" else -1  # LONG: retrace turun dari B
    price_at = lambda r: bp - sgn * r * leg
    fib_price = price_at(lvl)
    zone = tuple(sorted((price_at(lvl - hw), price_at(lvl + hw))))
    stop_ref = price_at(inv)
    price = float(c[-1])
    retrace_now = (bp - price) * sgn / leg
    invalid = ((c[b + 1:] - stop_ref) * sgn < 0).any() if b + 1 < n else False
    tl = int(p["touch_lookback"])
    last = slice(max(b + 1, n - tl), n)
    touched = bool(((l[last] <= zone[1]) if direction == "LONG" else (h[last] >= zone[0])).any()) if b + 1 < n else False
    last_dir_ok = (c[-1] > o[-1]) if direction == "LONG" else (c[-1] < o[-1])
    in_zone = zone[0] <= price <= zone[1]
    reaction = bool(touched and last_dir_ok and ((price >= zone[0]) if direction == "LONG" else (price <= zone[1])))
    if invalid:
        state = "invalidated"
    elif in_zone or (touched and reaction):
        state = "in_zone"
    else:
        state = classify_state(direction, price, zone, touched)
    setup = FibSetup(tf=tf, direction=direction, a_price=ap, a_time=seg.index[a], b_price=bp, b_time=seg.index[b],
                     level_price=float(fib_price), zone_lo=float(zone[0]), zone_hi=float(zone[1]),
                     stop_price=float(stop_ref), retrace_now=float(retrace_now), state=state)
    is_trigger = state == "in_zone" and reaction
    detail = (f"Impulse {'naik' if direction == 'LONG' else 'turun'} {fmt_price(ap)}→{fmt_price(bp)} "
              f"({leg / ap * 100:.1f}%), fib {lvl:g} = {fmt_price(fib_price)}, zona {fmt_price(zone[0])}–{fmt_price(zone[1])}, "
              f"retrace kini {retrace_now:.2f}. "
              + ("Ada REAKSI candle di zona -> trigger." if is_trigger else
                 "Belum ada reaksi di zona -> hanya level/konteks (tidak ada limit prediktif)."))
    f = Finding(kind="fib_075", role="trigger" if is_trigger else "context", direction=direction, tf=tf,
                strength=0.6 if is_trigger else 0.3, detail=detail, level=float(fib_price), entry=float(fib_price),
                entry_zone=(float(zone[0]), float(zone[1])), stop_ref=float(stop_ref), state=state,
                time=seg.index[-1], meta={"impulse_a": ap, "impulse_b": bp, "impulse_b_time": seg.index[b], "reaction": reaction,
                                          "level_name": f"Fib {lvl:g}", "level_kind": "fib", "level_tf": tf,
                                          "vol_spike": False})
    return setup, f
