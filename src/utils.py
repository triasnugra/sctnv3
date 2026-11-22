"""Utilitas umum: format harga/waktu, serialisasi JSON, hash konfigurasi."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass

import numpy as np
import pandas as pd


def fmt_price(x) -> str:
    if x is None:
        return "-"
    try:
        if np.isnan(x):
            return "-"
    except TypeError:
        return str(x)
    ax = abs(float(x))
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}"
    return f"{x:.6f}"


def fmt_ts(ts) -> str:
    if ts is None:
        return "-"
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def to_jsonable(o):
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: to_jsonable(getattr(o, f.name)) for f in fields(o)}
    return str(o)


def config_hash(*dicts) -> str:
    """Hash singkat konfigurasi -> untuk forward test yang bisa direproduksi."""
    blob = json.dumps([to_jsonable(d) for d in dicts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def signal_id(sig) -> str:
    """ID stabil per kejadian setup (tidak berubah antar-run) -> dedup jurnal & forward test."""
    p = sig.primary
    key = p.meta.get("deviation_time") or p.meta.get("breakout_time") or p.meta.get("impulse_b_time") or p.time
    raw = f"{sig.symbol}|{sig.action}|{p.kind}|{p.tf}|{key}|{p.meta.get('level_kind', '')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]
