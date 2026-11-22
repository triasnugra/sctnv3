"""Laporan Markdown + ringkasan signal."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..models import Signal, SymbolAnalysis
from ..utils import fmt_price, fmt_ts, safe_name

DISCLAIMER = ("> ⚠️ **Disclaimer**: laporan ini keluaran alat analisis berbasis aturan yang dioperasionalkan dari "
              "hipotesis. Tidak ada klaim profitabilitas, bukan nasihat finansial. Validasi lewat paper-trading / "
              "forward-test sebelum memakai modal nyata.")


def _row(s: Signal) -> str:
    if s.error:
        return f"| {s.symbol} | - | - | ⚠️ DATA ERROR | - | - | - | - | - | - |"
    p = s.plan
    icon = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}.get(s.action, "⚪ NO TRADE")
    if p:
        return (f"| {s.symbol} | {fmt_price(s.price)} | {s.bias} | {icon} | {s.status} | {s.score:g}/{s.max_score:g} | "
                f"{fmt_price(p.entry)} | {fmt_price(p.stop)} ({p.stop_pct:.2f}%) | {fmt_price(p.tp1)} | {p.blended_rr:.2f} |")
    return f"| {s.symbol} | {fmt_price(s.price)} | {s.bias} | {icon} | - | {s.score:g}/{s.max_score:g} | - | - | - | - |"


def _num(x, d=2) -> str:
    return "-" if x is None else f"{x:,.{d}f}"


def _ind_lines(ind: dict) -> List[str]:
    """Pembacaan indikator (v2). TAMPILAN/INFORMASI SAJA — tidak masuk skor."""
    tfs = ind.get("timeframes") or {}
    if not tfs:
        return []
    L = ["", "**Indikator (informasi saja — tidak masuk skor)**", "",
         "| TF | RSI | MACD | EMA | Supertrend | OBV | Volume | UCTS≈ (proksi) |", "|---|---|---|---|---|---|---|---|"]
    for tf, d in tfs.items():
        r, m, e, st = d.get("rsi", {}), d.get("macd", {}), d.get("ema", {}), d.get("supertrend")
        ob, vo, uc = d.get("obv"), d.get("volume"), d.get("ucts_proxy")
        stt = "-" if not st else (f"{'▲ naik' if st['dir'] == 1 else '▼ turun'} ({_num(st.get('dist_pct'))}% dari garis)")
        L.append(f"| {tf} | {_num(r.get('value'), 1)} ({r.get('zone') or '-'}) | {m.get('state', '-')}, {m.get('zero', '')} | "
                 f"{(e.get('stack') or '-').split(' (')[0]}, {e.get('vs_200', '-')} | {stt} | "
                 f"{'-' if not ob else ob['trend']} | {'-' if not vo else vo['state']} | "
                 f"{'-' if not uc else uc['side'] + ' ' + str(uc['bars_ago']) + ' bar lalu'} |")
    mtf = ind.get("mtf_supertrend") or {}
    if mtf:
        L += ["", "Supertrend multi-TF: " + " · ".join(f"{k} {'▲' if v == 1 else '▼'}" for k, v in mtf.items())]
    ptf = ind.get("primary_tf")
    vtf = ptf if ptf in tfs and tfs[ptf].get("volume_profile") else next((t for t, d in tfs.items() if d.get("volume_profile")), None)
    vp = tfs[vtf]["volume_profile"] if vtf else None
    if vp:
        L.append(f"- Volume Profile ({vtf}): POC {fmt_price(vp['poc'])} · VAH {fmt_price(vp['vah'])} · VAL {fmt_price(vp['val'])} "
                 f"(harga {vp['position']})")
    kl = ind.get("key_levels") or []
    if kl:
        near = sorted(kl, key=lambda k: abs(k["dist_pct"]))[:6]
        L.append("- Key Levels (Spaceman) terdekat: " + " · ".join(
            f"{k['label']} {fmt_price(k['price'])} ({k['dist_pct']:+.2f}%)" for k in near))
    L.append("- ℹ️ UCTS≈ adalah proksi terbuka (Supertrend flip + momentum MACD), bukan sinyal UCTS asli (skrip invite-only).")
    return L


def _detail(s: Signal, an: SymbolAnalysis | None, chart_rel: str | None) -> List[str]:
    L: List[str] = [f"### {s.symbol} — {s.action}" + (f" ({s.status})" if s.action != "NO TRADE" else ""), ""]
    if s.error:
        return L + [f"Data/analisis gagal: `{s.error}`", ""]
    if chart_rel:
        L += [f"![{s.symbol}]({chart_rel})", ""]
    L.append(f"- Harga: **{fmt_price(s.price)}** · Bias HTF: **{s.bias}** ({s.bias_score:+.2f})")
    if an:
        L.append(f"- {an.bias.summary}")
        for tf, rb in an.ranges.items():
            L.append(f"- Range {tf}: {fmt_price(rb.low)} – {fmt_price(rb.high)} · posisi harga: `{rb.position(an.price)}`")
        if an.monday:
            L.append(f"- Monday range{' (minggu lalu)' if an.monday.is_previous else ''}: {fmt_price(an.monday.low)} – {fmt_price(an.monday.high)}")
    L += _ind_lines((s.context or {}).get("indicators") or {})
    L.append("")
    L.append("**Alasan**")
    L += [f"- {r}" for r in s.reasons]
    if s.vetoes:
        L += ["", "**Veto / penghambat**"] + [f"- {v}" for v in s.vetoes]
    if s.score_items:
        L += ["", "**Rincian skor**", "", "| Item | Poin | Grup |", "|---|---|---|"]
        L += [f"| {i.label} | {i.points:+g} | {i.group} |" for i in s.score_items]
        L.append(f"| **Total** | **{s.score:g}** (min {s.min_score:g}) | |")
    if s.plan:
        p = s.plan
        L += ["", "**Rencana trade (scalp)**", "",
              "| Parameter | Nilai |", "|---|---|",
              f"| Arah / status | {p.direction} / {s.status} |",
              f"| Entry | {fmt_price(p.entry)}" + (f" (zona {fmt_price(p.entry_zone[0])}–{fmt_price(p.entry_zone[1])})" if p.entry_zone else "") + " |",
              f"| Stop (invalidasi) | {fmt_price(p.stop)} ({p.stop_pct:.2f}%) |",
              f"| TP1 ({p.tp1_close_pct:.0f}% posisi) | {fmt_price(p.tp1)} · {p.rr1:.2f}R · {p.tp1_source} |",
              f"| Moonbag TP2 | {fmt_price(p.tp2)} · {p.rr2:.2f}R · {p.tp2_source} |",
              f"| Setelah TP1 | stop → BE+ ({fmt_price(p.be_stop)}) |",
              f"| RR gabungan | {p.blended_rr:.2f} |",
              f"| Ukuran (risiko {p.risk_amount:.2f} USDT) | notional {p.notional:,.2f} · margin {p.margin:,.2f} · lev maks aman {p.leverage}x |",
              f"| Estimasi likuidasi | {fmt_price(p.liq_price)} |"]
        L += [""] + [f"- ℹ️ {n}" for n in p.notes]
    if s.warnings:
        L += [""] + [f"- ⚠️ {w}" for w in s.warnings]
    if s.watch:
        L += ["", "**Watchlist (jika–maka)**"] + [f"- {w}" for w in s.watch]
    L.append("")
    return L


def build_report(signals: List[Signal], analyses: Dict[str, SymbolAnalysis], charts: Dict[str, str],
                 now: pd.Timestamp, cfg_hash: str, source_note: str, errors: Dict[str, str]) -> str:
    n_act = sum(1 for s in signals if s.action in ("LONG", "SHORT"))
    L = [f"# CryptoEliz Scalp Engine v2 — Laporan {fmt_ts(now)} UTC", "", DISCLAIMER, "",
         f"- Sumber data: {source_note}",
         f"- Config hash: `{cfg_hash}` (catat untuk forward test)",
         f"- Signal aktif: **{n_act}** dari {len(signals)} simbol. *Default = NO TRADE; diam juga posisi.*", "",
         "## Ringkasan", "",
         "| Simbol | Harga | Bias HTF | Signal | Status | Skor | Entry | Stop | TP1 | RR |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L += [_row(s) for s in signals]
    L += ["", "## Detail per simbol", ""]
    for s in signals:
        L += _detail(s, analyses.get(s.symbol), charts.get(s.symbol))
    if errors:
        L += ["## Error data", ""] + [f"- **{k}**: {v}" for k, v in errors.items()] + [""]
    L += ["---", "Aturan: trade di tepi range · trigger dulu, bukan prediksi · invalidasi sebelum target · "
          "bias sederhana dari HTF · patience is a position · pisahkan investasi (spot) dan trading."]
    return "\n".join(L) + "\n"
