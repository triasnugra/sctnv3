"""Bias arah dari Weekly + Daily (BUKAN trigger).
Aturan sederhana: di atas level kunci -> cari long; di bawah -> cari short. Skor transparan per komponen."""
from __future__ import annotations

import pandas as pd

from ..data.indicators import ema, find_pivots
from ..models import BiasResult
from ..utils import fmt_price


def compute_bias(df_w: pd.DataFrame, df_d: pd.DataFrame, price: float, p: dict) -> BiasResult:
    comps = []
    wts = p["weights"]

    def add(name, value, note):
        comps.append({"name": name, "value": value, "weight": wts[name], "note": note})

    w = df_w.iloc[-int(p["weekly_lookback"]):]
    w_eq = (w["high"].max() + w["low"].min()) / 2
    add("weekly_eq", 1 if price > w_eq else -1, f"harga {'>' if price > w_eq else '<'} Weekly EQ {fmt_price(w_eq)}")

    if len(df_w) >= 2:
        prev = df_w.iloc[-1]  # candle weekly terakhir yang sudah close
        wc = price
        v = 1 if wc > prev["high"] else (-1 if wc < prev["low"] else 0)
        add("weekly_prev_range", v, f"vs range weekly terakhir {fmt_price(prev['low'])}–{fmt_price(prev['high'])}")
    else:
        add("weekly_prev_range", 0, "data weekly kurang")

    d = df_d.iloc[-int(p["daily_lookback"]):]
    d_hi, d_lo = float(d["high"].max()), float(d["low"].min())
    d_eq = (d_hi + d_lo) / 2
    add("daily_eq", 1 if price > d_eq else -1, f"harga {'>' if price > d_eq else '<'} Daily EQ {fmt_price(d_eq)} (key level)")

    n = int(p["pivot_n"])
    piv = find_pivots(df_d.iloc[-90:], n, n)
    hs = [x.price for x in piv if x.kind == "H"]
    ls = [x.price for x in piv if x.kind == "L"]
    if len(hs) >= 2 and len(ls) >= 2:
        hh, hl = hs[-1] > hs[-2], ls[-1] > ls[-2]
        v = 1 if (hh and hl) else (-1 if (not hh and not hl) else 0)
        add("daily_structure", v, f"struktur daily: {'HH' if hh else 'LH'} + {'HL' if hl else 'LL'}")
    else:
        add("daily_structure", 0, "pivot daily kurang")

    e = float(ema(df_d["close"], int(p["daily_ema"])).iloc[-1])
    add("daily_ema", 1 if price > e else -1, f"harga {'>' if price > e else '<'} EMA{int(p['daily_ema'])} daily {fmt_price(e)}")

    total_w = sum(c["weight"] for c in comps)
    score = sum(c["value"] * c["weight"] for c in comps) / total_w
    thr = float(p["neutral_threshold"])
    direction = "LONG" if score >= thr else ("SHORT" if score <= -thr else "NEUTRAL")
    zone = "premium" if price > d_eq else "discount"
    summary = (f"Bias {direction} (skor {score:+.2f}); key level Daily EQ {fmt_price(d_eq)}, harga di zona {zone}. "
               + ("Di atas level kunci -> cari long." if direction == "LONG" else
                  "Di bawah level kunci -> cari short." if direction == "SHORT" else "Tidak ada bias jelas -> selektif."))
    return BiasResult(direction=direction, score=float(score), components=comps, key_level=float(d_eq),
                      key_level_name="Daily EQ", zone=zone, summary=summary)
