"""Chart candlestick (matplotlib, headless) — sctnv2.

Panel harga : candle + range box + Monday/Key Levels (Spaceman) + Fib + FVG + entry/stop/target
              + EMA 12/26/50/100/200 + Supertrend + Volume Profile (POC/VAH/VAL) + UCTS≈ (proksi)
Panel bawah : Volume (+SMA) · OBV · RSI · MACD
Indikator = TAMPILAN SAJA (tidak mempengaruhi skor/signal). Semua bisa dimatikan di config/chart_indicators.yaml.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..config_loader import Config  # noqa: E402
from ..data.indicators import volume_profile  # noqa: E402
from ..data.key_levels import compute_key_levels  # noqa: E402
from ..models import Signal, SymbolAnalysis  # noqa: E402
from ..utils import fmt_price  # noqa: E402
from .indicator_suite import Suite, build_suites  # noqa: E402

log = logging.getLogger("engine.chart")
PAD = 30
EMA_STYLE = {12: ("#ff9800", 0.9), 26: ("#e91e63", 0.9), 50: ("#2196f3", 1.1), 100: ("#9c27b0", 1.2), 200: ("#263238", 1.6)}
UP, DN = "#26a69a", "#ef5350"


def chart_timeframe(an: SymbolAnalysis, sig: Signal, cfg: Config) -> str:
    ccfg = cfg.params.get("chart", {})
    tf = ccfg.get("timeframe", "1h")
    if ccfg.get("follow_trigger_tf", True) and sig.primary is not None and sig.primary.tf in an.frames:
        tf = sig.primary.tf
    if tf not in an.frames:
        tf = cfg.tfs["primary_trigger"]
    return tf


def _spread(ys: List[float], gap: float, lo: float, hi: float) -> List[float]:
    """Renggangkan label margin kanan agar tidak bertumpuk (maju lalu mundur supaya tetap di dalam area)."""
    out = list(ys)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    if out and out[-1] > hi - gap / 2:
        out[-1] = hi - gap / 2
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    return out


def _fmt_vol(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:.0f}"


def generate_chart(an: SymbolAnalysis, sig: Signal, cfg: Config, out_path: Path, tf: Optional[str] = None,
                   suites: Optional[Dict[str, Suite]] = None) -> bool:
    ccfg = cfg.params.get("chart", {})
    ci = cfg.chart_ind
    use_ind = bool(ci.get("enabled", True))
    tf = tf or chart_timeframe(an, sig, cfg)
    if tf not in an.frames:
        return False
    full = an.frames[tf]
    n = min(len(full), int(ccfg.get("candles", 160)))
    df = full.iloc[-n:]
    if n < 20:
        return False
    if use_ind and suites is None:
        suites = build_suites(an, cfg)
    suite: Optional[Suite] = (suites or {}).get(tf) if use_ind else None
    if use_ind and suite is None:
        use_ind = False

    pcfg = ci.get("panels", {})
    panels = [k for k in ("volume", "obv", "rsi", "macd") if use_ind and pcfg.get(k, True)]
    ratios = [6.0] + [{"volume": 1.3, "obv": 1.4, "rsi": 1.5, "macd": 1.7}[k] for k in panels]
    fig, axes = plt.subplots(len(ratios), 1, sharex=True, figsize=(14, 7.2 + 1.55 * len(panels)),
                             dpi=int(ccfg.get("dpi", 110)), gridspec_kw={"height_ratios": ratios, "hspace": 0.06})
    axes = np.atleast_1d(axes)
    ax = axes[0]

    x = np.arange(n)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    up = c >= o
    col = np.where(up, UP, DN)
    ax.vlines(x, l, h, colors=col, linewidth=0.8, zorder=3)
    ax.bar(x, np.maximum(np.abs(c - o), c.mean() * 1e-5), bottom=np.minimum(o, c), width=0.6, color=col, zorder=4)

    ymin, ymax = float(l.min()), float(h.max())
    plan = sig.plan
    if plan:
        ymin, ymax = min(ymin, plan.stop), max(ymax, plan.tp1)
        ymin, ymax = min(ymin, plan.tp1), max(ymax, plan.stop)
    span = ymax - ymin
    lo_v, hi_v = ymin - 0.15 * span, ymax + 0.15 * span

    def to_x(ts):
        return int(min(max(df.index.searchsorted(ts), 0), n - 1))

    labels: List[list] = []

    def hline(price, label, color, ls="--", lw=1.0):
        if not (lo_v <= price <= hi_v):
            return
        ax.hlines(price, 0, n + 1, colors=color, linestyles=ls, linewidth=lw, zorder=2)
        labels.append([price, f"{label} {fmt_price(price)}", color])

    # ---------------- Volume Profile (di belakang candle, anchor kiri) ----------------
    vcfg = ci.get("volume_profile", {})
    if use_ind and vcfg.get("enabled", True):
        vp = volume_profile(df, int(vcfg.get("bins", 48)), float(vcfg.get("value_area", 0.70)))
        if vp:
            cen, vol, edges = vp["centers"], vp["volumes"], vp["edges"]
            w = vol / vol.max() * float(vcfg.get("width_frac", 0.14)) * n
            in_va = (cen >= vp["val"]) & (cen <= vp["vah"])
            ax.barh(cen, w, height=(edges[1] - edges[0]) * 0.92, left=0, color=np.where(in_va, "#78909c", "#b0bec5"),
                    alpha=0.40, zorder=1, linewidth=0)
            hline(vp["poc"], "POC", "#f9a825", "-", 1.1)
            hline(vp["vah"], "VAH", "#78909c", ":", 0.9)
            hline(vp["val"], "VAL", "#78909c", ":", 0.9)

    # ---------------- range box & level lama ----------------
    for rtf, rb in an.ranges.items():
        if rtf == tf:
            x0 = to_x(rb.start_time)
            ax.add_patch(Rectangle((x0, rb.low), n - x0, rb.height, color="#9e9e9e", alpha=0.10, zorder=1))
            ax.add_patch(Rectangle((x0, rb.low), n - x0, rb.demand_top - rb.low, color=UP, alpha=0.12, zorder=1))
            ax.add_patch(Rectangle((x0, rb.supply_bottom), n - x0, rb.high - rb.supply_bottom, color=DN, alpha=0.12, zorder=1))
            hline(rb.high, f"Range H {rtf}", "#616161", "-", 1.2)
            hline(rb.low, f"Range L {rtf}", "#616161", "-", 1.2)
            hline(rb.mid, "Mid", "#9e9e9e", ":", 0.8)
        else:
            hline(rb.high, f"Range H {rtf}", "#7e57c2", ":", 1.0)
            hline(rb.low, f"Range L {rtf}", "#7e57c2", ":", 1.0)

    # ---------------- Spaceman Key Levels (Monday sudah termasuk di sini) ----------------
    klcfg = ci.get("key_levels", {})
    kl_on = use_ind and klcfg.get("enabled", True)
    if kl_on:
        ls_map = {"open": "-.", "high": "--", "low": "--", "mid": ":"}
        for k in compute_key_levels(an.frames, an.asof, an.monday, klcfg.get("show")):
            hline(k.price, k.label, k.color, ls_map.get(k.kind, "--"), 1.0 if k.kind != "mid" else 0.8)
    elif an.monday:
        hline(an.monday.high, "Monday H", "#1e88e5", "--")
        hline(an.monday.low, "Monday L", "#1e88e5", "--")

    for fs in an.fibs:
        if fs.tf == tf and fs.state != "invalidated":
            ax.add_patch(Rectangle((to_x(fs.b_time), fs.zone_lo), n - to_x(fs.b_time), fs.zone_hi - fs.zone_lo,
                                   color="#ffb300", alpha=0.18, zorder=1))
            ax.plot([to_x(fs.a_time), to_x(fs.b_time)], [fs.a_price, fs.b_price], color="#ffb300", lw=1.2, ls="-.", zorder=2)
            hline(fs.level_price, "Fib 0.75", "#ff8f00", "--")
    near_fvgs = sorted([z for z in an.fvgs if z.tf == tf], key=lambda z: abs((z.low + z.high) / 2 - an.price))[:4]
    for z in near_fvgs:
        x0 = to_x(z.time)
        ax.add_patch(Rectangle((x0, z.low), n - x0, z.high - z.low, color=UP if z.direction == "LONG" else DN,
                               alpha=0.20, zorder=1))

    # ---------------- overlay indikator: EMA, Supertrend, UCTS≈ ----------------
    handles: List = []
    if use_ind:
        for p, ser in sorted(suite.ema.items()):
            if len(full) < p:  # belum 'hangat' -> jangan menyesatkan
                continue
            colr, lw = EMA_STYLE.get(p, ("#607d8b", 1.0))
            ax.plot(x, ser.iloc[-n:].to_numpy(), color=colr, lw=lw, zorder=5)
            handles.append(Line2D([0], [0], color=colr, lw=max(lw, 1.2), label=f"EMA {p}"))
        if suite.st is not None:
            st, d = suite.st.iloc[-n:], suite.st_dir.iloc[-n:]
            ax.plot(x, st.where(d == 1).to_numpy(), color="#2e7d32", lw=1.6, zorder=5)
            ax.plot(x, st.where(d == -1).to_numpy(), color="#c62828", lw=1.6, zorder=5)
            sc = ci.get("supertrend", {})
            handles.append(Line2D([0], [0], color="#2e7d32", lw=1.8,
                                  label=f"Supertrend {sc.get('period', 10)}/{sc.get('multiplier', 3):g}"))
        if suite.ucts_buy is not None:
            off = 0.025 * (hi_v - lo_v)
            xb = np.flatnonzero(suite.ucts_buy.iloc[-n:].to_numpy())
            xs = np.flatnonzero(suite.ucts_sell.iloc[-n:].to_numpy())
            if len(xb):
                ax.scatter(xb, l[xb] - off, marker="^", s=60, color="#00c853", edgecolor="black", linewidth=0.5, zorder=7)
            if len(xs):
                ax.scatter(xs, h[xs] + off, marker="v", s=60, color="#ff1744", edgecolor="black", linewidth=0.5, zorder=7)
            handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#00c853", markeredgecolor="black",
                                  markersize=8, label="UCTS≈ Buy"))
            handles.append(Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff1744", markeredgecolor="black",
                                  markersize=8, label="UCTS≈ Sell"))
        if vcfg.get("enabled", True):
            handles.append(Line2D([0], [0], color="#f9a825", lw=1.5, label="POC (VP)"))

    # ---------------- trigger & rencana trade ----------------
    if sig.primary is not None and sig.primary.time is not None and sig.primary.tf == tf:
        xi = to_x(sig.primary.time)
        ax.annotate(sig.primary.kind.replace("_", " "), xy=(xi, c[xi]),
                    xytext=(xi - 12, c[xi] + (0.06 if sig.primary.direction == "SHORT" else -0.06) * span),
                    arrowprops=dict(arrowstyle="->", color="black"), fontsize=8, zorder=8)
    if plan:
        if plan.entry_zone:
            ax.axhspan(plan.entry_zone[0], plan.entry_zone[1], color="#1976d2", alpha=0.18, zorder=1)
        hline(plan.entry, "ENTRY", "#1976d2", "-", 1.6)
        hline(plan.stop, f"STOP ({plan.stop_pct:.2f}%)", "#d32f2f", "-", 1.6)
        hline(plan.tp1, f"TP1 ({plan.rr1:.1f}R)", "#2e7d32", "-", 1.4)
        hline(plan.tp2, f"TP2 ({plan.rr2:.1f}R)", "#66bb6a", "--", 1.2)

    # label margin kanan (anti-tumpang-tindih); harga terakhir ikut diatur sebagai kotak gelap
    labels.append([float(c[-1]), fmt_price(c[-1]), "#37474f", True])
    labels.sort(key=lambda t: t[0])
    gap = (hi_v - lo_v) * 0.027
    ys = _spread([t[0] for t in labels], gap, lo_v, hi_v)
    for lab, y in zip(labels, ys):
        price, text, color = lab[0], lab[1], lab[2]
        if len(lab) > 3:
            ax.text(n + 1.5, y, text, fontsize=7.5, va="center", color="white", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))
        else:
            ax.text(n + 1.5, y, text, fontsize=7.2, va="center", color=color)
        if abs(y - price) > 1e-12:
            ax.plot([n + 1, n + 1.4], [price, y], color=color, lw=0.6)
    ax.set_xlim(-1, n + PAD)
    ax.set_ylim(lo_v, hi_v)
    ax.grid(alpha=0.2)
    if handles:
        ax.legend(handles=handles, loc="upper left", ncol=5, fontsize=7, framealpha=0.8, borderpad=0.4, columnspacing=1.0)
    title = f"{an.symbol} {tf} — {sig.action}" + (f" ({sig.status})" if sig.action != "NO TRADE" else "")
    ax.set_title(f"{title} | skor {sig.score:g}/{sig.max_score:g} (min {sig.min_score:g}) | bias HTF {sig.bias}", fontsize=11)

    # ---------------- panel bawah ----------------
    def tail(series):
        return series.iloc[-n:].to_numpy()

    def side_label(a, y, text, color):
        a.text(n + 1.5, y, text, fontsize=7.5, va="center", color=color)

    def tag(a, text):
        a.text(0.004, 0.94, text, transform=a.transAxes, fontsize=8, fontweight="bold", va="top", color="#455a64")

    for key, a in zip(panels, axes[1:]):
        a.grid(alpha=0.2)
        if key == "volume":
            v = df["volume"].to_numpy()
            a.bar(x, v, width=0.7, color=col, alpha=0.75, zorder=2)
            vs = tail(suite.vol_sma)
            a.plot(x, vs, color="#ff6f00", lw=1.1, zorder=3)
            tag(a, f"Volume (SMA {ci.get('volume', {}).get('sma', 20)})")
            side_label(a, np.nanmax(v) * 0.5, _fmt_vol(v[-1]), "#455a64")
            a.set_ylim(0, np.nanmax(v) * 1.15)
        elif key == "obv":
            ov = tail(suite.obv)
            a.plot(x, ov, color="#3f51b5", lw=1.2, zorder=3)
            tag(a, "OBV")
            side_label(a, ov[-1], _fmt_vol(ov[-1]), "#3f51b5")
        elif key == "rsi":
            rv = tail(suite.rsi)
            ob, os_ = ci["rsi"]["overbought"], ci["rsi"]["oversold"]
            a.plot(x, rv, color="#7e57c2", lw=1.2, zorder=3)
            a.axhline(ob, color=DN, ls="--", lw=0.8)
            a.axhline(os_, color=UP, ls="--", lw=0.8)
            a.axhline(50, color="#9e9e9e", ls=":", lw=0.7)
            a.fill_between(x, ob, 100, color=DN, alpha=0.06)
            a.fill_between(x, 0, os_, color=UP, alpha=0.06)
            a.set_ylim(0, 100)
            a.set_yticks([os_, 50, ob])
            tag(a, f"RSI {ci['rsi']['period']}")
            if np.isfinite(rv[-1]):
                side_label(a, rv[-1], f"{rv[-1]:.1f}", "#7e57c2")
        elif key == "macd":
            mc, sg, hs = tail(suite.macd), tail(suite.macd_sig), tail(suite.macd_hist)
            prev = np.concatenate([[hs[0]], hs[:-1]])
            hc = np.where(hs >= 0, np.where(hs >= prev, "#26a69a", "#b2dfdb"), np.where(hs <= prev, "#ef5350", "#ffcdd2"))
            a.bar(x, hs, width=0.7, color=hc, zorder=2)
            a.plot(x, mc, color="#1565c0", lw=1.1, zorder=3)
            a.plot(x, sg, color="#ff6f00", lw=1.1, zorder=3)
            a.axhline(0, color="#9e9e9e", lw=0.7)
            m = ci["macd"]
            tag(a, f"MACD {m['fast']}/{m['slow']}/{m['signal']}")
            if np.isfinite(mc[-1]):
                side_label(a, mc[-1], f"{mc[-1]:.4g}", "#1565c0")

    bottom = axes[-1]
    step = max(1, n // 10)
    bottom.set_xticks(x[::step])
    bottom.set_xticklabels([df.index[i].strftime("%m-%d %H:%M") for i in x[::step]], rotation=0, fontsize=8)
    for a in axes:
        a.set_xlim(-1, n + PAD)
    foot = (f"asof {an.asof.strftime('%Y-%m-%d %H:%M')} UTC — alat analisis, bukan nasihat finansial; parameter = hipotesis. "
            "Indikator = tampilan saja (tidak masuk skor). UCTS≈ = proksi terbuka, bukan UCTS asli.")
    fig.text(0.01, 0.004, foot, fontsize=6.8, color="#757575")
    if use_ind and suites:
        parts = []
        for mtf in ci.get("mtf_timeframes", []):
            s = suites.get(mtf)
            if s is not None and s.st_dir is not None and int(s.st_dir.iloc[-1]) != 0:
                parts.append(f"{mtf} {'▲' if int(s.st_dir.iloc[-1]) == 1 else '▼'}")
        if parts:
            fig.text(0.99, 0.004, "Supertrend MTF: " + "  ".join(parts), fontsize=7.5, color="#37474f", ha="right",
                     fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.075 if len(panels) else 0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True
