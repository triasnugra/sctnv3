"""Indikator dasar & primitif pattern (pivot, ATR, OBV, volume).
Detektor spesifik (deviation, fib, dll) ada di src/analysis/."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    unit, num = tf[-1], int(tf[:-1])
    if unit not in _TF_SECONDS:
        raise ValueError(f"Timeframe tidak didukung: {tf}")
    return pd.Timedelta(seconds=num * _TF_SECONDS[unit])


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def last_atr(df: pd.DataFrame, period: int = 14) -> float:
    v = atr(df, period).iloc[-1]
    if v is None or np.isnan(v):
        v = float((df["high"] - df["low"]).tail(period).mean())
    return float(v)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, recent: int = 3, base: int = 50) -> float:
    v = df["volume"]
    if len(v) < recent + 10:
        return float("nan")
    base = min(base, len(v) - recent)
    b = v.iloc[-(recent + base):-recent].mean()
    if b <= 0 or np.isnan(b):
        return float("nan")
    return float(v.iloc[-recent:].mean() / b)


def vol_sma_prior(v: np.ndarray, n: int = 20) -> np.ndarray:
    """SMA volume dari n candle SEBELUM tiap indeks (eksklusif)."""
    return pd.Series(v).rolling(n, min_periods=5).mean().shift(1).to_numpy()


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str  # 'H' | 'L'


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Pivot]:
    """Swing high/low terkonfirmasi (butuh `right` candle setelahnya).
    High: > semua `left` candle sebelumnya dan >= semua `right` candle sesudahnya."""
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(df)
    out: List[Pivot] = []
    for i in range(left, n - right):
        if h[i] > h[i - left:i].max() and h[i] >= h[i + 1:i + right + 1].max():
            out.append(Pivot(i, float(h[i]), "H"))
        if l[i] < l[i - left:i].min() and l[i] <= l[i + 1:i + right + 1].min():
            out.append(Pivot(i, float(l[i]), "L"))
    out.sort(key=lambda p: p.idx)
    return out


def merge_alternating(pivots: List[Pivot]) -> List[Pivot]:
    """Pastikan pivot selang-seling H/L; jika dua sejenis berurutan ambil yang lebih ekstrem."""
    out: List[Pivot] = []
    for p in sorted(pivots, key=lambda x: x.idx):
        if out and out[-1].kind == p.kind:
            if (p.kind == "H" and p.price > out[-1].price) or (p.kind == "L" and p.price < out[-1].price):
                out[-1] = p
        else:
            out.append(p)
    return out


# =====================================================================
# v2 — indikator untuk GRAFIK (tampilan saja; tidak dipakai skor/veto/signal)
# =====================================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder (RMA). Rata-rata kerugian 0 -> RSI 100."""
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~((al == 0.0) & ag.notna()), 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD klasik. Return (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend (ATR Wilder, hl2 +/- multiplier x ATR; default TradingView 10/3).
    Return (garis, arah): arah +1 = uptrend (garis di bawah harga), -1 = downtrend."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr(df, period).to_numpy(float)
    n = len(df)
    st = np.full(n, np.nan)
    dr = np.zeros(n, dtype=int)
    fu = fl = np.nan
    trend = 1
    started = False
    for i in range(n):
        if np.isnan(a[i]):
            continue
        hl2 = (h[i] + l[i]) / 2.0
        bu, bl = hl2 + multiplier * a[i], hl2 - multiplier * a[i]
        if not started:
            fu, fl = bu, bl
            trend = 1 if c[i] >= hl2 else -1
            started = True
        else:
            pc = c[i - 1]
            fu = bu if (bu < fu or pc > fu) else fu
            fl = bl if (bl > fl or pc < fl) else fl
            if trend == 1 and c[i] < fl:
                trend = -1
            elif trend == -1 and c[i] > fu:
                trend = 1
        st[i] = fl if trend == 1 else fu
        dr[i] = trend
    return pd.Series(st, index=df.index), pd.Series(dr, index=df.index)


def volume_profile(df: pd.DataFrame, bins: int = 48, value_area: float = 0.70):
    """Volume profile range-terlihat: volume tiap candle dibagi rata ke bin harga yang dilewati [low, high].
    Return dict(centers, volumes, edges, poc, vah, val) atau None bila data tidak cukup."""
    if df is None or len(df) < 5:
        return None
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for bl, bh, v in zip(df["low"].to_numpy(float), df["high"].to_numpy(float), df["volume"].to_numpy(float)):
        if v <= 0:
            continue
        if bh - bl < 1e-12:
            k = min(max(int(np.searchsorted(edges, bl, side="right")) - 1, 0), bins - 1)
            vol[k] += v
            continue
        ov = np.clip(np.minimum(edges[1:], bh) - np.maximum(edges[:-1], bl), 0.0, None)
        s = ov.sum()
        if s > 0:
            vol += v * ov / s
    total = vol.sum()
    if total <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    k = int(np.argmax(vol))
    lo_i = hi_i = k
    acc = vol[k]
    while acc < value_area * total and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return {"centers": centers, "volumes": vol, "edges": edges, "poc": float(centers[k]),
            "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i])}
