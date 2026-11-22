"""Statistik R-multiple: expectancy, PF, SQN, CI95, max drawdown, kesiapan (UNVERIFIABLE sampai sampel cukup)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_stats(rs: List[float], times: Optional[List[pd.Timestamp]] = None, ev: Optional[dict] = None) -> Dict:
    ev = ev or {}
    n = len(rs)
    out: Dict = {"n": n, "status": "NO DATA"}
    if n == 0:
        return out
    a = np.array(rs, float)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    wins, losses = a[a > 0], a[a < 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else None
    sqn = float(math.sqrt(n) * mean / std) if std > 0 else None
    z = float(ev.get("ci_z", 1.96))
    ci = (mean - z * std / math.sqrt(n), mean + z * std / math.sqrt(n)) if n > 1 else (None, None)
    eq = np.cumsum(a)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    weeks = 0.0
    if times:
        ts = pd.to_datetime(pd.Series(times), utc=True)
        weeks = float((ts.max() - ts.min()).total_seconds() / 604800)
    ready = {
        "trades": n >= int(ev.get("min_trades", 150)),
        "weeks": weeks >= float(ev.get("min_weeks", 20)),
        "ci_lower_positive": ci[0] is not None and ci[0] > 0,
        "profit_factor": pf is not None and pf >= float(ev.get("min_profit_factor", 1.2)),
    }
    if not (ready["trades"] and ready["weeks"]):
        status = "UNVERIFIABLE"   # sampel belum cukup -> JANGAN simpulkan apa pun
    else:
        status = "PASS" if all(ready.values()) else "FAIL"
    out.update({
        "wins": int((a > 0).sum()), "losses": int((a < 0).sum()), "win_rate": float((a > 0).mean()),
        "avg_r": mean, "std_r": std, "total_r": float(a.sum()), "profit_factor": pf, "sqn": sqn,
        "ci_low": ci[0], "ci_high": ci[1], "max_drawdown_r": dd, "weeks": weeks,
        "best_r": float(a.max()), "worst_r": float(a.min()), "readiness": ready, "status": status,
        "targets": {"min_trades": int(ev.get("min_trades", 150)), "min_weeks": float(ev.get("min_weeks", 20)),
                    "min_profit_factor": float(ev.get("min_profit_factor", 1.2))},
    })
    return out


def group_stats(trades: List[dict], key: str, ev: Optional[dict] = None) -> Dict[str, Dict]:
    groups: Dict[str, List[float]] = {}
    for t in trades:
        groups.setdefault(str(t.get(key, "?")), []).append(float(t["r"]))
    res = {}
    for k, rs in groups.items():
        st = compute_stats(rs, None, ev)
        res[k] = {"n": st["n"], "win_rate": st.get("win_rate"), "avg_r": st.get("avg_r"),
                  "profit_factor": st.get("profit_factor"), "total_r": st.get("total_r")}
    return res
