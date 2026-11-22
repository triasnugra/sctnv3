"""Manajemen risiko: filter konteks (event/volume), stop, target bertahap, sizing & leverage aman.
Prinsip: invalidasi sebelum target; lebih baik miss daripada stop lebar/likuidasi."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pandas as pd

from ..models import Finding, SymbolAnalysis, TradePlan
from ..utils import fmt_price


def context_vetoes(an: SymbolAnalysis, now: pd.Timestamp, rcfg: dict) -> List[str]:
    out: List[str] = []
    f = rcfg.get("filters", {}) or {}
    lv = f.get("low_volume", {}) or {}
    if lv.get("enabled"):
        r = an.vol_ratio.get(lv.get("timeframe", "1h"))
        if r is not None and not pd.isna(r) and r < lv.get("min_ratio", 0.5):
            out.append(f"Volume rendah: {r:.2f}x baseline (< {lv.get('min_ratio', 0.5)}x) — hindari entry")
    mc = f.get("monthly_close", {}) or {}
    if mc.get("enabled"):
        month_start = now.normalize().replace(day=1)
        next_start = month_start + pd.offsets.MonthBegin(1)
        if (next_start - now) <= pd.Timedelta(hours=mc.get("hours_before", 12)):
            out.append("Menjelang monthly close (UTC) — hindari entry baru")
        elif (now - month_start) <= pd.Timedelta(hours=mc.get("hours_after", 2)):
            out.append("Tepat setelah monthly close (UTC) — volatilitas tidak stabil, tunggu")
    for ev in f.get("event_blackouts") or []:
        try:
            s = pd.Timestamp(ev["start_utc"]).tz_convert("UTC") if pd.Timestamp(ev["start_utc"]).tzinfo else pd.Timestamp(ev["start_utc"], tz="UTC")
            e = pd.Timestamp(ev["end_utc"]).tz_convert("UTC") if pd.Timestamp(ev["end_utc"]).tzinfo else pd.Timestamp(ev["end_utc"], tz="UTC")
        except Exception:  # noqa: BLE001
            out.append(f"Event blackout '{ev.get('name', '?')}' format waktu tidak valid (diabaikan)")
            continue
        if s <= now <= e:
            out.append(f"Event blackout aktif: {ev.get('name', 'event')}")
    return out


def _dedupe(levels: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for p, lab in sorted(levels):
        if out and abs(p - out[-1][0]) / p < 0.001:
            continue
        out.append((p, lab))
    return out


def build_plan(an: SymbolAnalysis, direction: str, primary: Finding, atr_val: float,
               target_levels: List[Tuple[float, str]], rcfg: dict) -> Tuple[Optional[TradePlan], List[str], List[str]]:
    vetoes: List[str] = []
    warns: List[str] = []
    st, tg, lv, acc = rcfg["stop"], rcfg["targets"], rcfg["leverage"], rcfg["account"]
    long = direction == "LONG"
    sign = 1 if long else -1
    entry = float(an.price) if primary.state == "in_zone" else float(primary.entry)

    buf = max(entry * st["buffer_pct"] / 100, st["buffer_atr_mult"] * atr_val)
    stop = primary.stop_ref - buf if long else primary.stop_ref + buf
    dist = (entry - stop) * sign
    if dist <= 0:
        return None, ["Stop tidak valid: titik invalidasi berada di sisi salah dari entry (harga sudah melewati)"], warns
    stop_pct = dist / entry * 100
    if stop_pct < st["min_stop_pct_floor"]:
        dist = entry * st["min_stop_pct_floor"] / 100
        stop = entry - dist * sign
        stop_pct = st["min_stop_pct_floor"]
        warns.append(f"Stop struktural terlalu rapat (noise) -> dilebarkan ke {stop_pct:.2f}%")
    if stop_pct > st["max_stop_pct"]:
        return None, [f"Stop {stop_pct:.2f}% melebihi batas {st['max_stop_pct']}% — lebih baik miss daripada stop lebar"], warns
    fee_r = float(rcfg.get("evaluation", {}).get("fee_pct_roundtrip", 0.0)) / stop_pct
    if fee_r > st.get("max_fee_r", 1e9):
        return None, [f"Fee ≈ {fee_r:.2f}R (> {st['max_fee_r']}R) — stop {stop_pct:.2f}% terlalu rapat, biaya memakan edge"], warns
    lo, hi = st.get("preferred_stop_pct_range", [3.0, 6.0])
    if stop_pct < lo:
        warns.append(f"Stop {stop_pct:.2f}% lebih rapat dari kisaran 3–6% (stop struktural di titik invalidasi)")

    cands = _dedupe([(p, lab) for p, lab in target_levels if (p - entry) * sign > 0])
    cands.sort(key=lambda x: abs(x[0] - entry))
    r_of = lambda p: abs(p - entry) / dist

    tp1, src1 = None, ""
    for p, lab in cands:
        if r_of(p) >= tg["tp1_min_r"]:
            tp1, src1 = p, lab
            break
    if tp1 is None:
        tp1 = entry + sign * tg["tp1_fallback_r"] * dist
        src1 = f"fallback {tg['tp1_fallback_r']}R (tanpa level struktur)"
    rr1 = r_of(tp1)
    tp2, src2 = None, ""
    for p, lab in cands:
        if (p - tp1) * sign > 0 and r_of(p) >= max(tg["tp2_min_r"], rr1 + tg.get("tp2_min_gap_r", 0.5)):
            tp2, src2 = p, lab
            break
    if tp2 is None:
        r2 = max(tg["tp2_fallback_r"], rr1 + 0.5)
        tp2 = entry + sign * r2 * dist
        src2 = f"fallback {r2:.1f}R (moonbag)"
    rr2 = r_of(tp2)
    frac = tg["tp1_close_pct"] / 100
    blended = frac * rr1 + (1 - frac) * rr2
    if blended < tg["min_blended_rr"]:
        return None, [f"RR gabungan {blended:.2f} < minimum {tg['min_blended_rr']}"], warns

    sf = stop_pct / 100
    lev_liq = int(math.floor(1 / (sf * lv["liq_distance_vs_stop_mult"] + lv["maintenance_margin_rate"])))
    leverage = max(1, min(int(lv["max_leverage"]), lev_liq))
    equity = float(acc["equity_usdt"])
    risk_amt = equity * acc["risk_per_trade_pct"] / 100
    notional = risk_amt / sf
    margin = notional / leverage
    notes: List[str] = []
    if margin > equity:
        notional = equity * leverage
        margin = equity
        risk_amt = notional * sf
        notes.append("Ukuran posisi dibatasi oleh equity & leverage maksimum (risiko riil < target)")
    mmr = lv["maintenance_margin_rate"]
    liq = entry * (1 - 1 / leverage + mmr) if long else entry * (1 + 1 / leverage - mmr)
    be = entry + sign * entry * rcfg["targets"]["be_buffer_pct"] / 100
    notes.append("Estimasi likuidasi kasar (isolated); cek kalkulator exchange sebelum eksekusi")
    plan = TradePlan(direction=direction, entry=entry, entry_zone=primary.entry_zone, stop=float(stop),
                     stop_pct=float(stop_pct), tp1=float(tp1), tp1_source=src1, tp2=float(tp2), tp2_source=src2,
                     rr1=float(rr1), rr2=float(rr2), blended_rr=float(blended),
                     tp1_close_pct=float(tg["tp1_close_pct"]), be_stop=float(be), risk_amount=float(risk_amt),
                     notional=float(notional), margin=float(margin), leverage=leverage, liq_price=float(liq), notes=notes)
    return plan, vetoes, warns
