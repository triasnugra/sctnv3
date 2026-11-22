"""Backtest walk-forward (replay as-of): pada tiap titik waktu, mesin HANYA melihat candle yang sudah close
pada saat itu (tanpa lookahead), menghasilkan signal yang sama seperti live, lalu hasilnya disimulasikan
dengan candle 15m sesudahnya (simulator yang sama dengan forward test).
Jalankan: python -m src.backtest.run --days 45 [--demo] [--symbols BTC/USDT,ETH/USDT]
CATATAN: backtest ≠ bukti edge. Parameter adalah hipotesis; hati-hati overfitting & data snooping."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..analysis.pipeline import analyze_symbol
from ..config_loader import load_config
from ..data.fetch_ohlcv import OHLCVFetcher
from ..data.indicators import tf_to_timedelta
from ..evaluate.simulator import simulate_trade
from ..evaluate.stats import compute_stats, group_stats
from ..logging_setup import setup_logging
from ..signal.signal_engine import evaluate_symbol
from ..utils import config_hash, to_jsonable, utcnow

log = logging.getLogger("engine.backtest")


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--step-hours", type=float, default=0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    out = Path(a.output_dir) / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(os.getenv("LOG_LEVEL", "INFO"), str(Path(a.output_dir) / "engine.log"))
    if a.symbols:
        os.environ["ENGINE_SYMBOLS"] = a.symbols
    cfg = load_config(a.config_dir)
    bt = cfg.params.get("backtest", {})
    days = a.days or int(bt.get("days", 45))
    step = a.step_hours or float(bt.get("step_hours", 1))
    ev = cfg.risk.get("evaluation", {})
    now = utcnow().floor("15min")
    symbols = cfg.symbols
    bench = cfg.symbols_cfg.get("benchmark")
    if bench in symbols:
        symbols = [bench] + [s for s in symbols if s != bench]

    if a.demo:
        from ..data.synthetic import DemoSource
        source = DemoSource()
    else:
        d = cfg.symbols_cfg.get("data", {})
        source = OHLCVFetcher(d.get("providers", ["binance_rest"]), int(d.get("retries", 3)),
                              float(d.get("backoff_seconds", 2.0)), int(d.get("timeout_seconds", 20)))
    limits = {tf: int(cfg.symbols_cfg["candles"][tf]) for tf in cfg.all_timeframes}
    need = {tf: limits[tf] + int(math.ceil(days * 86400 / tf_to_timedelta(tf).total_seconds())) + 10 for tf in limits}

    full: Dict[str, Dict[str, pd.DataFrame]] = {}
    close_idx: Dict[str, Dict[str, pd.DatetimeIndex]] = {}
    for sym in symbols:
        try:
            full[sym] = source.fetch_symbol(sym, need, now=now)
            close_idx[sym] = {tf: df.index + tf_to_timedelta(tf) for tf, df in full[sym].items()}
            log.info("%s: data backtest siap", sym)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: data gagal, dilewati — %s", sym, exc)
    symbols = [s for s in symbols if s in full]
    if not symbols:
        log.error("Tidak ada data. Backtest dibatalkan.")
        return 1

    start = (now - pd.Timedelta(days=days)).ceil("h")
    asofs = list(pd.date_range(start, now - pd.Timedelta(hours=1), freq=pd.Timedelta(hours=step), tz="UTC"))
    log.info("Replay %d titik waktu x %d simbol (%s → %s)", len(asofs), len(symbols), asofs[0], asofs[-1])
    seen, trades = set(), []
    for k, asof in enumerate(asofs):
        slices: Dict[str, Dict[str, pd.DataFrame]] = {}
        for sym in symbols:
            sl = {}
            for tf, df in full[sym].items():
                n = close_idx[sym][tf].searchsorted(asof, side="right")
                sl[tf] = df.iloc[max(0, n - limits[tf]):n]
            slices[sym] = sl
        btc_bias = None
        for sym in symbols:
            try:
                an = analyze_symbol(sym, slices[sym], cfg, asof)
                if sym == bench:
                    btc_bias = an.bias
                sig = evaluate_symbol(an, cfg, btc_bias, asof)
            except Exception as exc:  # noqa: BLE001
                log.debug("skip %s @ %s: %s", sym, asof, exc)
                continue
            if sig.action == "NO TRADE" or not sig.plan or sig.signal_id in seen:
                continue
            seen.add(sig.signal_id)
            p = sig.plan
            s = {"direction": sig.action, "status": sig.status, "asof": asof, "entry": p.entry, "stop": p.stop,
                 "tp1": p.tp1, "tp2": p.tp2, "tp1_close_pct": p.tp1_close_pct, "be_stop": p.be_stop, "stop_pct": p.stop_pct}
            res = simulate_trade(s, full[sym]["15m"], ev)
            trades.append({"signal_id": sig.signal_id, "asof": str(asof), "symbol": sym, "direction": sig.action,
                           "trigger": sig.primary.kind, "tf": sig.primary.tf, "status_at_signal": sig.status,
                           "score": sig.score, "entry": p.entry, "stop": p.stop, "tp1": p.tp1, "tp2": p.tp2, **res})
        if k % 100 == 0:
            log.info("progress %d/%d — trade terkumpul: %d", k, len(asofs), len(trades))

    closed = [t for t in trades if t["status"] == "CLOSED"]
    stats = compute_stats([t["r"] for t in closed], [pd.Timestamp(t["asof"]) for t in closed], ev)
    eq, c = [], 0.0
    for t in sorted(closed, key=lambda t: t["exit_time"]):
        c += t["r"]
        eq.append({"t": t["exit_time"], "r": round(c, 3)})
    summary = {"generated": now, "demo": bool(a.demo), "days": days, "step_hours": step, "symbols": symbols,
               "period": [str(asofs[0]), str(asofs[-1])], "config_hash": config_hash(cfg.symbols_cfg, cfg.params, cfg.risk),
               "stats": stats, "by_trigger": group_stats(closed, "trigger", ev), "by_symbol": group_stats(closed, "symbol", ev),
               "by_direction": group_stats(closed, "direction", ev),
               "counts": {k: sum(1 for t in trades if t["status"] == k) for k in ("CLOSED", "OPEN", "WAITING", "CANCELLED")},
               "equity": eq,
               "warning": "Backtest bukan bukti edge. Sampel kecil, sinyal berkorelasi, parameter = hipotesis. "
                          "Validasi dengan forward test."}
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=1, ensure_ascii=False), encoding="utf-8")
    if trades:
        keys = list(trades[0].keys())
        with open(out / "trades.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(trades)
    log.info("Backtest selesai: %d signal, %d closed, status %s", len(trades), len(closed), stats["status"])
    if stats.get("n"):
        log.info("Win rate %.1f%% | avg R %.3f | PF %s | SQN %s", stats["win_rate"] * 100, stats["avg_r"],
                 stats["profit_factor"], stats["sqn"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
