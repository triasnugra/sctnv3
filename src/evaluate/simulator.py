"""Simulasi hasil satu trade dari plan signal, candle 15m demi candle.
Aturan KONSERVATIF (hipotesis, hindari hasil terlalu bagus):
 - Stop dan target tersentuh di candle yang sama -> STOP dianggap duluan (setelah TP1: BE duluan).
 - Candle tempat limit pending terisi hanya dicek stop-nya (TP baru dicek candle berikutnya).
 - Pending dibatalkan bila TP1 tercapai sebelum terisi atau lewat pending_expiry_hours.
 - TP1 menutup tp1_close_pct; sisanya stop di BE+ lalu ke TP2. Timeout -> mark-to-market.
 - Biaya: fee_pct_roundtrip / stop% (dalam R) dikurangkan dari trade yang terisi."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

TF = pd.Timedelta(minutes=15)


def simulate_trade(s: Dict[str, Any], candles: pd.DataFrame, ev: Dict[str, Any]) -> Dict[str, Any]:
    long = s["direction"] == "LONG"
    sg = 1 if long else -1
    entry, stop, tp1, tp2, be = (float(s[k]) for k in ("entry", "stop", "tp1", "tp2", "be_stop"))
    dist = abs(entry - stop)
    f = float(s["tp1_close_pct"]) / 100
    r1, r2 = abs(tp1 - entry) / dist, abs(tp2 - entry) / dist
    be_r = (be - entry) * sg / dist
    cost = float(ev.get("fee_pct_roundtrip", 0.1)) / float(s["stop_pct"])
    asof = pd.Timestamp(s["asof"])
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    cs = candles[candles.index >= asof.ceil("15min")]
    hold = pd.Timedelta(hours=float(ev.get("max_hold_hours", 72)))
    expiry = asof + pd.Timedelta(hours=float(ev.get("pending_expiry_hours", 24)))

    res: Dict[str, Any] = {"status": "WAITING", "outcome": "", "r": None, "fill_time": None, "exit_time": None, "reason": ""}
    filled = s["status"] == "ENTRY ZONE"
    fill_time = asof if filled else None
    stage, last_close = 1, entry

    def closed(outcome, r, ts):
        res.update(status="CLOSED", outcome=outcome, r=round(r - cost, 4), exit_time=str(ts),
                   fill_time=str(fill_time))
        return res

    for ts, row in cs.iterrows():
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        close_ts = ts + TF
        if not filled:
            if ts >= expiry:
                res.update(status="CANCELLED", reason="pending kedaluwarsa")
                return res
            if (hi >= tp1) if long else (lo <= tp1):
                touched_first = (lo <= entry) if long else (hi >= entry)
                if not touched_first:
                    res.update(status="CANCELLED", reason="TP1 tercapai sebelum entry terisi")
                    return res
            if (lo <= entry) if long else (hi >= entry):
                filled, fill_time = True, ts
                if (lo <= stop) if long else (hi >= stop):
                    return closed("LOSS", -1.0, ts)
                last_close = cl
            continue
        if stage == 1:
            if (lo <= stop) if long else (hi >= stop):
                return closed("LOSS", -1.0, ts)
            if (hi >= tp1) if long else (lo <= tp1):
                stage = 2
                last_close = cl
                continue
        else:
            if (lo <= be) if long else (hi >= be):
                return closed("TP1_BE", f * r1 + (1 - f) * be_r, ts)
            if (hi >= tp2) if long else (lo <= tp2):
                return closed("TP2", f * r1 + (1 - f) * r2, ts)
        last_close = cl
        if close_ts - fill_time >= hold:
            mark = (cl - entry) * sg / dist
            r = mark if stage == 1 else f * r1 + (1 - f) * mark
            return closed("TIMEOUT", r, ts)
    if not filled:
        return res  # WAITING
    mark = (last_close - entry) * sg / dist
    r = mark if stage == 1 else f * r1 + (1 - f) * mark
    res.update(status="OPEN", r=round(r - cost, 4), fill_time=str(fill_time),
               reason="TP1 tercapai, sisa berjalan" if stage == 2 else "")
    return res
