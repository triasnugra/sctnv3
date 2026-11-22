from __future__ import annotations

from typing import Tuple

TRADABLE_STATES = ("in_zone", "waiting_retest")


def classify_state(direction: str, price: float, zone: Tuple[float, float], retested: bool) -> str:
    """Status entry terhadap zona:
    in_zone        : harga di dalam zona entry
    waiting_retest : harga di sisi 'menguntungkan', retest belum terjadi -> limit pending
    late           : retest sudah terjadi & harga sudah menjauh -> TIDAK dikejar
    deep_retest    : harga menembus zona ke sisi berlawanan (belum invalid) -> tunggu
    """
    lo, hi = zone
    if lo <= price <= hi:
        return "in_zone"
    adverse = price > hi if direction == "SHORT" else price < lo
    if adverse:
        return "deep_retest"
    return "late" if retested else "waiting_retest"
