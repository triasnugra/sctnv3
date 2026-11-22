"""Three-tap (tiga kali manipulasi) & Wolf Wave sederhana. Keduanya = KONFIRMASI, bukan trigger.

Three-tap (hipotesis): tiga pivot high berurutan naik (tiap langkah min_step..max_step %),
tap ke-3 gagal bertahan: close terakhir kembali < tap ke-2 -> SHORT (cermin untuk LONG).
Wolf (hipotesis): 5 pivot selang-seling; garis 1-3 dan 2-4 konvergen; titik 5 overshoot di luar garis 1-3
(stop-run) lalu harga bertahan di atasnya -> LONG (cermin untuk SHORT)."""
from __future__ import annotations

from typing import List

import pandas as pd

from ..data.indicators import find_pivots, merge_alternating
from ..models import Finding
from ..utils import fmt_price, fmt_ts


def detect_three_tap(df: pd.DataFrame, p: dict, tf: str) -> List[Finding]:
    seg = df.iloc[-int(p["pivot_lookback"]):]
    n = len(seg)
    if n < 30:
        return []
    c = seg["close"].to_numpy()
    pn = int(p["pivot_n"])
    piv = find_pivots(seg, pn, pn)
    out: List[Finding] = []
    lo_s, hi_s = p["min_step_pct"] / 100, p["max_step_pct"] / 100
    for kind in ("H", "L"):
        pts = [x for x in piv if x.kind == kind][-3:]
        if len(pts) < 3:
            continue
        p1, p2, p3 = pts
        s = 1 if kind == "H" else -1  # H: harga naik tiap tap
        st1, st2 = (p2.price - p1.price) * s / p1.price, (p3.price - p2.price) * s / p2.price
        if not (lo_s <= st1 <= hi_s and lo_s <= st2 <= hi_s):
            continue
        if p3.idx - p1.idx > p["max_span_bars"] or (n - 1 - p3.idx) > p["recent_bars"]:
            continue
        if (c[p3.idx + 1:] * s > p3.price * s).any():
            continue  # tap ke-3 sudah ditembus -> bukan failure
        confirmed = (c[-1] - p2.price) * s < 0  # close kembali melewati tap ke-2
        if not confirmed:
            continue
        direction = "SHORT" if kind == "H" else "LONG"
        out.append(Finding("three_tap", "confirmation", direction, tf, 0.6,
                           f"Three-tap {'atas' if kind == 'H' else 'bawah'} ({fmt_price(p1.price)} → {fmt_price(p2.price)} → "
                           f"{fmt_price(p3.price)}), tap-3 @ {fmt_ts(seg.index[p3.idx])} gagal; close kembali melewati tap-2",
                           level=p2.price, stop_ref=p3.price, time=seg.index[p3.idx]))
    return out


def _wolf_bull_geometry(xs, ps, ov_pct):
    """Geometri wolf bullish pada deret harga ps (5 titik: L H L H L). Return (ok, lower_line_x5)."""
    x1, x2, x3, x4, x5 = xs
    p1, p2, p3, p4, p5 = ps
    if not (p3 < p1 and p4 < p2 and p5 < p3):
        return False, 0.0
    s_l = (p3 - p1) / (x3 - x1)
    s_u = (p4 - p2) / (x4 - x2)
    lower = lambda x: p1 + s_l * (x - x1)
    upper = lambda x: p2 + s_u * (x - x2)
    gap2, gap5 = upper(x2) - lower(x2), upper(x5) - lower(x5)
    if not (0 < gap5 < gap2):
        return False, 0.0
    l5 = lower(x5)
    if (l5 - p5) < abs(l5) * ov_pct / 100:
        return False, 0.0
    return True, l5


def detect_wolf(df: pd.DataFrame, p: dict, tf: str) -> List[Finding]:
    seg = df.iloc[-int(p["pivot_lookback"]):]
    n = len(seg)
    if n < 40:
        return []
    c = seg["close"].to_numpy()
    pn = int(p["pivot_n"])
    piv = merge_alternating(find_pivots(seg, pn, pn))
    if len(piv) < 5:
        return []
    pts = piv[-5:]
    kinds = "".join(x.kind for x in pts)
    xs = [x.idx for x in pts]
    ps = [x.price for x in pts]
    if xs[4] - xs[0] > p["max_span_bars"] or (n - 1 - xs[4]) > p["recent_bars"]:
        return []
    if kinds == "LHLHL":
        ok, l5 = _wolf_bull_geometry(xs, ps, p["overshoot_min_pct"])
        if ok and c[-1] > ps[4] and not (seg["low"].to_numpy()[xs[4] + 1:] < ps[4]).any():
            return [Finding("wolf", "confirmation", "LONG", tf, 0.6,
                            f"Wolf wave bullish: titik-5 {fmt_price(ps[4])} overshoot garis 1-3 ({fmt_price(l5)}) lalu bertahan; "
                            f"target konservatif titik-4 {fmt_price(ps[3])}",
                            level=ps[3], stop_ref=ps[4], time=seg.index[xs[4]], meta={"target": ps[3]})]
    elif kinds == "HLHLH":
        neg = [-x for x in ps]
        ok, l5 = _wolf_bull_geometry(xs, neg, p["overshoot_min_pct"])
        if ok and c[-1] < ps[4] and not (seg["high"].to_numpy()[xs[4] + 1:] > ps[4]).any():
            return [Finding("wolf", "confirmation", "SHORT", tf, 0.6,
                            f"Wolf wave bearish: titik-5 {fmt_price(ps[4])} overshoot garis 1-3 ({fmt_price(-l5)}) lalu gagal bertahan; "
                            f"target konservatif titik-4 {fmt_price(ps[3])}",
                            level=ps[3], stop_ref=ps[4], time=seg.index[xs[4]], meta={"target": ps[3]})]
    return []
