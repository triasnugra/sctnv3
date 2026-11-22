"""Key Levels ala "Key Levels SpacemanBTC IDWM" (TradingView, open-source) — TAMPILAN SAJA.

Level yang dihitung (semua UTC, hanya dari candle yang sudah close atau candle pertama periode berjalan):
  day    : Open hari ini · High/Low/Mid hari sebelumnya
  week   : Open minggu ini (Senin 00:00) · High/Low/Mid minggu sebelumnya
  month  : Open bulan ini · High/Low/Mid bulan sebelumnya
  monday : High/Low/Mid candle harian Senin (dari MondayRange mesin; fallback Senin minggu lalu)
  year   : Open tahun ini

Ini replika konsep (level open/high/low/mid per periode), bukan salinan kode indikator aslinya.
Level TIDAK dipakai skor/veto/signal — hanya digambar di chart & ditampilkan di dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .indicators import tf_to_timedelta

GROUP_LABEL = {"day": "D", "week": "W", "month": "M", "monday": "Mon", "year": "Y"}
GROUP_COLOR = {"day": "#fb8c00", "week": "#1e88e5", "month": "#8e24aa", "monday": "#00acc1", "year": "#6d4c41"}


@dataclass
class KeyLevel:
    group: str       # day | week | month | monday | year
    kind: str        # open | high | low | mid
    price: float
    label: str       # label pendek untuk chart, mis. "PW High"
    name: str        # nama panjang untuk dashboard/laporan

    @property
    def color(self) -> str:
        return GROUP_COLOR.get(self.group, "#757575")


def _first_of(df: Optional[pd.DataFrame], tf: str, start: pd.Timestamp) -> Optional[float]:
    """Open candle pertama pada/setelah `start`, hanya bila itu memang candle pembuka periode."""
    if df is None or df.empty:
        return None
    pos = int(df.index.searchsorted(start))
    if pos >= len(df):
        return None
    if df.index[pos] - start > tf_to_timedelta(tf):  # data tidak mencapai awal periode
        return None
    return float(df["open"].iloc[pos])


def _hlm(rows: pd.DataFrame, min_rows: int) -> Optional[tuple]:
    if rows is None or len(rows) < min_rows:
        return None
    hi, lo = float(rows["high"].max()), float(rows["low"].min())
    return hi, lo, (hi + lo) / 2.0


def compute_key_levels(frames: Dict[str, pd.DataFrame], asof: pd.Timestamp, monday=None,
                       show: Optional[Dict[str, List[str]]] = None) -> List[KeyLevel]:
    """show: {group: [kind, ...]} — default: semua level. Level yang datanya tidak cukup dilewati."""
    out: List[KeyLevel] = []

    def want(group: str, kind: str) -> bool:
        return show is None or kind in (show.get(group) or [])

    def add(group: str, kind: str, price: Optional[float], label: str, name: str):
        if price is not None and want(group, kind):
            out.append(KeyLevel(group, kind, float(price), label, name))

    d1 = frames.get("1d")
    today = pd.Timestamp(asof).normalize()
    week0 = today - pd.Timedelta(days=int(today.weekday()))
    month0 = today.replace(day=1)
    year0 = today.replace(month=1, day=1)

    # ---- open periode berjalan (dari TF terkecil yang mencapai awal periode) ----
    def open_from(prefs, start):
        for tf in prefs:
            v = _first_of(frames.get(tf), tf, start)
            if v is not None:
                return v
        return None

    add("day", "open", open_from(["15m", "1h", "4h"], today), "D Open", "Open hari ini")
    add("week", "open", open_from(["1h", "4h", "15m", "1d"], week0), "W Open", "Open minggu ini")
    add("month", "open", open_from(["4h", "1d", "1h"], month0), "M Open", "Open bulan ini")
    add("year", "open", open_from(["1d"], year0), "Y Open", "Open tahun ini")

    # ---- high/low/mid periode sebelumnya (dari candle harian) ----
    if d1 is not None and len(d1):
        idx = d1.index
        spec = [
            ("day", today - pd.Timedelta(days=1), today, 1, "PD", "hari sebelumnya"),
            ("week", week0 - pd.Timedelta(days=7), week0, 5, "PW", "minggu sebelumnya"),
            ("month", (month0 - pd.Timedelta(days=1)).replace(day=1), month0, 20, "PM", "bulan sebelumnya"),
        ]
        for group, a, b, min_rows, pre, nm in spec:
            r = _hlm(d1[(idx >= a) & (idx < b)], min_rows)
            if r:
                hi, lo, mid = r
                add(group, "high", hi, f"{pre} High", f"High {nm}")
                add(group, "low", lo, f"{pre} Low", f"Low {nm}")
                add(group, "mid", mid, f"{pre} Mid", f"Mid {nm}")

    # ---- Monday range ----
    if monday is not None:
        tag = " (minggu lalu)" if getattr(monday, "is_previous", False) else ""
        add("monday", "high", monday.high, "Mon High", f"Monday High{tag}")
        add("monday", "low", monday.low, "Mon Low", f"Monday Low{tag}")
        add("monday", "mid", monday.mid, "Mon Mid", f"Monday Mid{tag}")
    return out
