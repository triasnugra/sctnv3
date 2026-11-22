"""OBV: divergensi & pola V sebagai konfirmasi volume.
Divergensi bullish (hipotesis): dua pivot low harga berurutan, low ke-2 lebih rendah, OBV pada low ke-2 lebih tinggi.
Pola V bullish: dalam v_window candle OBV turun lalu pulih >= v_recovery_frac dari ayunan,
dengan ayunan >= v_min_swing_vol_mult x rata-rata volume. Inverted-V = bearish."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..data.indicators import find_pivots, obv
from ..models import Finding
from ..utils import fmt_price, fmt_ts


def detect_obv(df: pd.DataFrame, p: dict, tf: str) -> List[Finding]:
    n = len(df)
    if n < 60:
        return []
    o = obv(df).to_numpy()
    out: List[Finding] = []
    lb = int(p["divergence_lookback"])
    seg = df.iloc[-lb:]
    off = n - len(seg)
    pn = int(p["pivot_n"])
    piv = find_pivots(seg, pn, pn)
    for kind, direction in (("L", "LONG"), ("H", "SHORT")):
        pts = [x for x in piv if x.kind == kind]
        if len(pts) < 2:
            continue
        a, b = pts[-2], pts[-1]
        gap = b.idx - a.idx
        if not (p["min_pivot_gap_bars"] <= gap <= p["max_pivot_gap_bars"]) or (len(seg) - 1 - b.idx) > p["recent_bars"]:
            continue
        md = p["min_price_diff_pct"] / 100
        if kind == "L":
            ok = b.price < a.price * (1 - md) and o[off + b.idx] > o[off + a.idx]
        else:
            ok = b.price > a.price * (1 + md) and o[off + b.idx] < o[off + a.idx]
        if ok:
            out.append(Finding("obv_divergence", "confirmation", direction, tf, 0.6,
                               f"Divergensi OBV {'bullish' if kind == 'L' else 'bearish'}: harga {'LL' if kind == 'L' else 'HH'} "
                               f"{fmt_price(a.price)}→{fmt_price(b.price)} sementara OBV {'lebih tinggi' if kind == 'L' else 'lebih rendah'} "
                               f"({fmt_ts(seg.index[b.idx])})", time=seg.index[b.idx]))
    # pola V
    w = int(p["v_window"])
    ow = o[-w:]
    avgv = float(df["volume"].tail(50).mean())
    if len(ow) == w and avgv > 0:
        for direction in ("LONG", "SHORT"):
            k = int(np.argmin(ow)) if direction == "LONG" else int(np.argmax(ow))
            pos = k / (w - 1)
            if not (p["v_min_pos"] <= pos <= p["v_max_pos"]):
                continue
            if direction == "LONG":
                move, recov = ow[0] - ow[k], ow[-1] - ow[k]
            else:
                move, recov = ow[k] - ow[0], ow[k] - ow[-1]
            if move >= p["v_min_swing_vol_mult"] * avgv and recov >= p["v_recovery_frac"] * move:
                out.append(Finding("obv_v", "confirmation", direction, tf, 0.5,
                                   f"Pola {'V' if direction == 'LONG' else 'V-terbalik'} OBV ({w} candle): ayunan "
                                   f"{move / avgv:.1f}x rata-rata volume, pulih {recov / move * 100:.0f}%"))
    return out
