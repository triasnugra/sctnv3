"""Monday Range: High/Low candle harian Senin (UTC) sebagai level acuan."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..models import Finding, Level, MondayRange
from ..utils import fmt_price


def compute_monday_range(df_d: pd.DataFrame, now: pd.Timestamp, p: dict) -> Optional[MondayRange]:
    if df_d is None or len(df_d) < 8:
        return None
    today = now.normalize()
    monday = today - pd.Timedelta(days=int(today.weekday()))
    options = [monday]
    if p.get("fallback_to_previous", True):
        options.append(monday - pd.Timedelta(days=7))
    for i, m in enumerate(options):
        if m in df_d.index:  # candle Senin berjalan belum close -> tidak ada di df
            r = df_d.loc[m]
            return MondayRange(high=float(r["high"]), low=float(r["low"]), date=m, is_previous=(i == 1))
    return None


def monday_levels(mr: Optional[MondayRange]) -> List[Level]:
    if mr is None:
        return []
    tag = " (minggu lalu)" if mr.is_previous else ""
    return [Level(f"Monday High{tag}", mr.high, "monday_high", "1d", ("up", "down")),
            Level(f"Monday Low{tag}", mr.low, "monday_low", "1d", ("up", "down"))]


def monday_finding(mr: Optional[MondayRange], price: float) -> Optional[Finding]:
    if mr is None:
        return None
    rng = f"{fmt_price(mr.low)} – {fmt_price(mr.high)}"
    if price > mr.high:
        return Finding("monday_level", "context", "LONG", "1d", 0.4, f"Harga di ATAS Monday High (Monday range {rng})")
    if price < mr.low:
        return Finding("monday_level", "context", "SHORT", "1d", 0.4, f"Harga di BAWAH Monday Low (Monday range {rng})")
    return Finding("monday_level", "context", "NEUTRAL", "1d", 0.0, f"Harga di DALAM Monday range ({rng})")
