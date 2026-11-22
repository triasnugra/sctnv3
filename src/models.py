"""Struktur data bersama antar modul."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Level:
    name: str
    price: float
    kind: str                      # range_high | range_low | monday_high | monday_low
    tf: str
    orientations: Tuple[str, ...] = ("up", "down")


@dataclass
class RangeBox:
    tf: str
    high: float
    low: float
    touches_high: int
    touches_low: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    edge_zone_frac: float = 0.2

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def height(self) -> float:
        return self.high - self.low

    @property
    def width_pct(self) -> float:
        return self.height / self.low * 100

    @property
    def demand_top(self) -> float:
        return self.low + self.edge_zone_frac * self.height

    @property
    def supply_bottom(self) -> float:
        return self.high - self.edge_zone_frac * self.height

    def position(self, price: float) -> str:
        if price > self.high:
            return "outside_up"
        if price < self.low:
            return "outside_down"
        if price <= self.demand_top:
            return "demand_edge"
        if price >= self.supply_bottom:
            return "supply_edge"
        return "mid"


@dataclass
class Finding:
    kind: str                      # deviation_reclaim | breakout_retest | fib_075 | fvg | obv_divergence | ...
    role: str                      # trigger | confirmation | context
    direction: str                 # LONG | SHORT | NEUTRAL
    tf: str
    strength: float = 0.5
    detail: str = ""
    level: Optional[float] = None
    entry: Optional[float] = None
    entry_zone: Optional[Tuple[float, float]] = None
    stop_ref: Optional[float] = None
    state: str = ""                # in_zone | waiting_retest | late | deep_retest | invalidated
    time: Optional[pd.Timestamp] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FVGZone:
    tf: str
    direction: str                 # LONG (bullish gap) | SHORT (bearish gap)
    low: float
    high: float
    time: pd.Timestamp
    touched_recent: bool = False


@dataclass
class FibSetup:
    tf: str
    direction: str                 # arah trade pullback: LONG (impulse naik) | SHORT
    a_price: float
    a_time: pd.Timestamp
    b_price: float
    b_time: pd.Timestamp
    level_price: float
    zone_lo: float
    zone_hi: float
    stop_price: float
    retrace_now: float
    state: str


@dataclass
class MondayRange:
    high: float
    low: float
    date: pd.Timestamp
    is_previous: bool = False

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class BiasResult:
    direction: str                 # LONG | SHORT | NEUTRAL
    score: float
    components: List[Dict[str, Any]]
    key_level: float
    key_level_name: str
    zone: str                      # premium | discount
    summary: str = ""


@dataclass
class SymbolAnalysis:
    symbol: str
    asof: pd.Timestamp
    price: float
    bias: BiasResult
    monday: Optional[MondayRange]
    ranges: Dict[str, RangeBox]
    levels: List[Level]
    findings: List[Finding]
    fvgs: List[FVGZone]
    fibs: List[FibSetup]
    atr: Dict[str, float]
    vol_ratio: Dict[str, float]
    frames: Dict[str, pd.DataFrame]
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoreItem:
    label: str
    points: float
    group: str                     # bias | trigger | state | confluence | confirm | penalty
    note: str = ""


@dataclass
class TradePlan:
    direction: str
    entry: float
    entry_zone: Optional[Tuple[float, float]]
    stop: float
    stop_pct: float
    tp1: float
    tp1_source: str
    tp2: float
    tp2_source: str
    rr1: float
    rr2: float
    blended_rr: float
    tp1_close_pct: float
    be_stop: float
    risk_amount: float
    notional: float
    margin: float
    leverage: int
    liq_price: float
    notes: List[str] = field(default_factory=list)


@dataclass
class Signal:
    symbol: str
    action: str                    # LONG | SHORT | NO TRADE
    status: str                    # ENTRY ZONE | PENDING LIMIT | -
    asof: pd.Timestamp
    price: float
    bias: str
    bias_score: float
    score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0
    primary: Optional[Finding] = None
    plan: Optional[TradePlan] = None
    score_items: List[ScoreItem] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    watch: List[str] = field(default_factory=list)
    error: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = ""
