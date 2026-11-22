"""Data sintetis untuk --demo / test. BUKAN data pasar. Hanya menguji pipeline."""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import pandas as pd

from .fetch_ohlcv import drop_unclosed

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
_RULE = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON"}
_BASE = {"BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0, "XRP": 0.6, "DOGE": 0.15}


def _resample(df15: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "15m":
        return df15.copy()
    kw = {"label": "left", "closed": "left"}
    out = df15.resample(_RULE[tf], **kw).agg(_AGG).dropna()
    return out


def generate_15m(symbol: str, now: pd.Timestamp, n_bars: int = 70000, seed: int = 7) -> pd.DataFrame:
    h = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + h % 10000)
    base = symbol.split("/")[0]
    p0 = _BASE.get(base, 1.0 + (h % 200))
    n_tail = 900
    n_head = n_bars - n_tail
    drift = 0.00002 * np.sin(np.arange(n_head) / 4000.0)
    head = p0 * np.exp(np.cumsum(rng.normal(drift, 0.0022, n_head)))
    center = head[-1]
    t = np.arange(n_tail)
    e = np.zeros(n_tail)
    for i in range(1, n_tail):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.0012)
    tail = center * (1 + 0.035 * np.sin(2 * np.pi * t / 300 + rng.uniform(0, 6))) * np.exp(e)
    lo = tail[:-30].min()
    inj = np.array([0.004, 0.0, -0.004, -0.007, -0.005, 0.002, 0.006, 0.008, 0.007, 0.006])
    tail[-10:] = lo * (1 + inj)
    close = np.concatenate([head, tail])
    openp = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.0008, n_bars)))
    lw = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.0008, n_bars)))
    vol = 100 * rng.lognormal(0, 0.5, n_bars) * (1 + 3 * (rng.random(n_bars) < 0.03))
    last_open = now.floor("15min") - pd.Timedelta(minutes=15)
    idx = pd.date_range(end=last_open, periods=n_bars, freq="15min", tz="UTC")
    return pd.DataFrame({"open": openp, "high": hi, "low": lw, "close": close, "volume": vol}, index=idx)


class DemoSource:
    """Mengikuti antarmuka OHLCVFetcher.fetch_symbol."""

    def fetch_symbol(self, symbol: str, candles: Dict[str, int], now: pd.Timestamp | None = None,
                     min_candles: Dict[str, int] | None = None) -> Dict[str, pd.DataFrame]:
        now = now or pd.Timestamp.now(tz="UTC")
        df15 = generate_15m(symbol, now)
        out = {}
        for tf, limit in candles.items():
            df = drop_unclosed(_resample(df15, tf), tf, now)
            out[tf] = df.iloc[-limit:]
        return out
