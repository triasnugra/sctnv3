"""Gabungkan semua temuan -> signal final dengan skor transparan.
Alur: (1) bias HTF  (2) trigger LTF  (3) konfluensi/konfirmasi  (4) veto & risk plan.
Setiap kemungkinan setup (arah x trigger) dinilai sendiri; yang valid & skor tertinggi dipilih.
Default: NO TRADE bila ada veto atau skor kurang."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from ..analysis.common import TRADABLE_STATES
from ..config_loader import Config
from ..models import BiasResult, Finding, ScoreItem, Signal, SymbolAnalysis, TradePlan
from ..utils import fmt_price, signal_id
from . import risk_manager

TRIG_KEYS = {"deviation_reclaim": "trigger_deviation_reclaim", "breakout_retest": "trigger_breakout_retest",
             "fib_075": "trigger_fib_075"}
EXTRA_KEYS = ["confluence_htf_range_edge", "confluence_monday_level", "confluence_fib_zone", "confirm_fvg",
              "confirm_obv_divergence", "confirm_obv_v", "confirm_three_tap", "confirm_wolf", "confirm_volume_spike"]


@dataclass
class Candidate:
    direction: str
    primary: Optional[Finding] = None
    items: List[ScoreItem] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    plan: Optional[TradePlan] = None

    @property
    def score(self) -> float:
        return sum(i.points for i in self.items)


def collect_target_levels(an: SymbolAnalysis) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for tf, rb in an.ranges.items():
        out += [(rb.mid, f"Range Mid ({tf})"), (rb.high, f"Range High ({tf})"), (rb.low, f"Range Low ({tf})")]
    if an.monday:
        out += [(an.monday.high, "Monday High"), (an.monday.low, "Monday Low")]
    for fs in an.fibs:
        out.append((fs.b_price, f"Ekstrem impulse fib ({fs.tf})"))
    return out


def _max_score(w: dict) -> float:
    return (w["bias_aligned"] + max(w[k] for k in TRIG_KEYS.values()) + w["entry_zone_reached"]
            + sum(w[k] for k in EXTRA_KEYS))


def _score_setup(an: SymbolAnalysis, D: str, t: Finding, cfg: Config, btc_bias: Optional[BiasResult],
                 ctx_vetoes: List[str]) -> Candidate:
    sc = cfg.params["scoring"]
    w = sc["weights"]
    c = Candidate(direction=D, primary=t)
    add = lambda label, pts, group, note="": c.items.append(ScoreItem(label, pts, group, note))
    # 1) bias HTF
    b = an.bias.direction
    if b == D:
        add("Bias HTF searah", w["bias_aligned"], "bias", an.bias.summary)
    elif b == "NEUTRAL":
        add("Bias HTF netral", w.get("penalty_neutral_bias", 0), "bias", "Tidak ada bias jelas — butuh konfirmasi lebih banyak")
    elif sc.get("allow_counter_bias", False):
        add("Bias HTF berlawanan (diizinkan config)", 0, "bias", an.bias.summary)
        c.warnings.append("Trade melawan bias HTF")
    else:
        c.vetoes.append(f"Bias HTF {b} berlawanan dengan {D} (di atas level kunci -> long, di bawah -> short)")
    # 2) trigger
    add(f"Trigger: {t.kind} ({t.tf})", w[TRIG_KEYS[t.kind]], "trigger", t.detail)
    if t.state == "in_zone":
        add("Harga di entry zone", w["entry_zone_reached"], "state", f"{fmt_price(t.entry_zone[0])}–{fmt_price(t.entry_zone[1])}")
    if t.meta.get("vol_spike"):
        add("Volume spike pada trigger", w["confirm_volume_spike"], "confirm")
    # 3) konfluensi di entry
    entry = float(t.entry)
    tol = sc["confluence_tol_pct"] / 100
    near = lambda p: abs(p - entry) <= entry * tol
    if t.meta.get("level_tf") != "4h":
        for rb in an.ranges.values():
            if rb.tf == "4h" and (near(rb.high) or near(rb.low)):
                add("Konfluensi tepi range 4H", w["confluence_htf_range_edge"], "confluence",
                    f"{fmt_price(rb.low)} / {fmt_price(rb.high)}")
                break
    if an.monday and not str(t.meta.get("level_kind", "")).startswith("monday"):
        for nm, p in (("Monday High", an.monday.high), ("Monday Low", an.monday.low)):
            if near(p):
                add("Konfluensi Monday level", w["confluence_monday_level"], "confluence", f"{nm} {fmt_price(p)}")
                break
    if t.kind != "fib_075":
        for fs in an.fibs:
            if fs.direction == D and fs.state != "invalidated" and fs.zone_lo * (1 - tol) <= entry <= fs.zone_hi * (1 + tol):
                add("Konfluensi fib 0.75", w["confluence_fib_zone"], "confluence",
                    f"zona {fmt_price(fs.zone_lo)}–{fmt_price(fs.zone_hi)} ({fs.tf})")
                break
    fvg_f = [f for f in an.findings if f.kind == "fvg" and f.direction == D]
    fvg_z = [z for z in an.fvgs if z.direction == D and z.low * (1 - tol) <= entry <= z.high * (1 + tol)]
    if fvg_f or fvg_z:
        add("FVG sejalan", w["confirm_fvg"], "confirm", (fvg_f[0].detail if fvg_f else
                                                         f"entry dalam FVG {fvg_z[0].tf} {fmt_price(fvg_z[0].low)}–{fmt_price(fvg_z[0].high)}"))
    for kind, key, lab in (("obv_divergence", "confirm_obv_divergence", "Divergensi OBV"),
                           ("obv_v", "confirm_obv_v", "Pola V OBV"),
                           ("three_tap", "confirm_three_tap", "Three-tap"),
                           ("wolf", "confirm_wolf", "Wolf wave")):
        fl = [f for f in an.findings if f.kind == kind and f.direction == D]
        if fl:
            add(lab, w[key], "confirm", f"[{fl[0].tf}] {fl[0].detail}")
    # penalti BTC
    bench = cfg.symbols_cfg.get("benchmark")
    if btc_bias is not None and an.symbol != bench and btc_bias.direction not in ("NEUTRAL", D):
        add("Konflik dengan bias BTC", w["penalty_btc_conflict"], "penalty", btc_bias.summary)
    # 4) veto
    if t.state == "waiting_retest":
        if not sc.get("allow_pending", True):
            c.vetoes.append("Entry pending (belum di zona) tidak diizinkan oleh config")
        else:
            atr_v = an.atr.get(t.tf, 0)
            if atr_v and abs(an.price - entry) / atr_v > sc["max_pending_distance_atr"]:
                c.vetoes.append(f"Level entry terlalu jauh dari harga (> {sc['max_pending_distance_atr']} ATR) — tidak dikejar/ditebak")
    if sc.get("block_mid_range", True) and t.kind != "breakout_retest":
        rb = an.ranges.get(cfg.tfs["primary_trigger"]) or an.ranges.get("4h")
        if rb and rb.position(entry) == "mid":
            c.vetoes.append(f"Entry {fmt_price(entry)} di TENGAH range {rb.tf} — trade di tepi, bukan di tengah")
    total = c.score
    if total < sc["min_score"]:
        c.vetoes.append(f"Skor {total:g} < minimum {sc['min_score']}")
    extras = sum(1 for i in c.items if i.group in ("confluence", "confirm") and i.points > 0)
    if extras < sc["min_extra_items"]:
        c.vetoes.append(f"Konfluensi/konfirmasi tambahan {extras} < minimum {sc['min_extra_items']}")
    c.vetoes += ctx_vetoes
    if not c.vetoes:
        plan, pv, pw = risk_manager.build_plan(an, D, t, an.atr.get(t.tf, 0.0), collect_target_levels(an), cfg.risk)
        c.warnings += pw
        c.vetoes += pv
        c.plan = plan
    return c


def _context(an: SymbolAnalysis) -> dict:
    return {
        "bias_components": an.bias.components, "bias_summary": an.bias.summary, "key_level": an.bias.key_level,
        "key_level_name": an.bias.key_level_name, "zone": an.bias.zone,
        "ranges": [{"tf": tf, "low": rb.low, "high": rb.high, "mid": rb.mid, "width_pct": rb.width_pct,
                    "position": rb.position(an.price), "touches_low": rb.touches_low, "touches_high": rb.touches_high}
                   for tf, rb in an.ranges.items()],
        "monday": ({"high": an.monday.high, "low": an.monday.low, "is_previous": an.monday.is_previous}
                   if an.monday else None),
        "atr": an.atr, "vol_ratio": an.vol_ratio,
    }


def _watch_levels(an: SymbolAnalysis, cfg: Config) -> List[str]:
    out: List[str] = []
    ptf = cfg.tfs["primary_trigger"]
    rb = an.ranges.get(ptf) or an.ranges.get("4h")
    if rb:
        out.append(f"LONG bila: harga menembus Range Low {rb.tf} ({fmt_price(rb.low)}) sebentar lalu close kembali di atasnya "
                   f"(deviation+reclaim) -> entry di retest, stop di bawah wick deviasi.")
        out.append(f"SHORT bila: harga menembus Range High {rb.tf} ({fmt_price(rb.high)}) sebentar lalu close kembali di bawahnya "
                   f"-> entry di retest, stop di atas wick deviasi.")
        out.append(f"Tengah range ({fmt_price(rb.demand_top)}–{fmt_price(rb.supply_bottom)}) = zona tunggu, bukan zona entry.")
    if an.monday:
        out.append(f"Level acuan Monday: High {fmt_price(an.monday.high)} / Low {fmt_price(an.monday.low)}"
                   f"{' (minggu lalu)' if an.monday.is_previous else ''}.")
    out.append(f"Bias HTF: {an.bias.direction} — key level {an.bias.key_level_name} {fmt_price(an.bias.key_level)}.")
    return out


def evaluate_symbol(an: SymbolAnalysis, cfg: Config, btc_bias: Optional[BiasResult], now: pd.Timestamp) -> Signal:
    sc = cfg.params["scoring"]
    w = sc["weights"]
    prio = sc["trigger_priority"]
    ctx = risk_manager.context_vetoes(an, now, cfg.risk)
    cands: List[Candidate] = []
    skipped: List[str] = []
    for D in ("LONG", "SHORT"):
        trigs = [f for f in an.findings if f.role == "trigger" and f.direction == D]
        for t in trigs:
            if t.state in TRADABLE_STATES:
                cands.append(_score_setup(an, D, t, cfg, btc_bias, ctx))
            else:
                skipped.append(f"{t.kind} {D} [{t.tf}] status '{t.state}' — "
                               f"{'entry sudah lewat, tidak dikejar' if t.state == 'late' else 'tidak dapat dieksekusi'}")
    valid = [c for c in cands if not c.vetoes]
    sig = Signal(symbol=an.symbol, action="NO TRADE", status="-", asof=an.asof, price=an.price,
                 bias=an.bias.direction, bias_score=an.bias.score, max_score=_max_score(w), min_score=sc["min_score"])
    sig.watch = _watch_levels(an, cfg)
    sig.context = _context(an)
    sig.warnings = list(an.notes)
    if valid:
        best = max(valid, key=lambda c: (c.score, -prio.index(c.primary.kind) if c.primary.kind in prio else -99))
        sig.action = best.direction
        sig.status = "ENTRY ZONE" if best.primary.state == "in_zone" else "PENDING LIMIT"
        sig.score, sig.primary, sig.plan = best.score, best.primary, best.plan
        sig.signal_id = signal_id(sig)
        sig.score_items = best.items
        sig.warnings += best.warnings
        sig.reasons = [f"{i.label}: {i.points:+g}" + (f" — {i.note}" if i.note and i.group in ('confluence', 'confirm', 'penalty') else "")
                       for i in best.items]
        sig.reasons.append(f"Total skor {best.score:g} ≥ minimum {sc['min_score']} dan semua veto lolos.")
        return sig
    # NO TRADE -> jelaskan
    sig.reasons.append("Diam juga posisi: tidak ada setup yang lolos semua aturan.")
    if ctx:
        sig.vetoes += ctx
    if cands:
        nb = max(cands, key=lambda c: c.score)
        sig.score, sig.primary, sig.score_items = nb.score, nb.primary, nb.items
        own = [v for v in nb.vetoes if v not in ctx]
        sig.reasons.append(f"Kandidat terdekat: {nb.direction} via {nb.primary.kind} [{nb.primary.tf}], skor {nb.score:g}/{sig.max_score:g}.")
        sig.vetoes += own
        sig.warnings += nb.warnings
    else:
        sig.reasons.append("Tidak ada trigger valid (deviation+reclaim, breakout+retest, atau fib 0.75 dengan reaksi).")
    sig.vetoes += [f"Dilewati: {s}" for s in skipped]
    return sig
