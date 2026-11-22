"""Entry point: python -m src.main [--demo] [--symbols BTC/USDT,ETH/USDT] [--no-charts]"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .analysis.pipeline import analyze_symbol
from .config_loader import ConfigError, load_config
from .data.fetch_ohlcv import OHLCVFetcher
from .data.synthetic import DemoSource
from .logging_setup import setup_logging
from .models import Signal, SymbolAnalysis
from .notify import send_alerts
from .report.chart_generator import chart_timeframe, generate_chart
from .report.generate_report import build_report
from .report.indicator_suite import build_suites, summarize_symbol
from .signal.signal_engine import evaluate_symbol
from .utils import config_hash, fmt_ts, safe_name, to_jsonable, utcnow

log = logging.getLogger("engine.main")

LOG_HEADER = ["signal_id", "asof_utc", "symbol", "action", "status", "score", "price", "entry", "stop", "stop_pct",
              "tp1", "tp2", "rr1", "rr2", "blended_rr", "tp1_close_pct", "be_stop", "trigger", "trigger_tf",
              "bias", "config_hash", "demo"]


def ensure_log_schema(log_path: Path, header=None) -> bool:
    """Pastikan jurnal memakai skema kolom terbaru. Jurnal lama (skema berbeda, mis. tanpa signal_id) dipindah ke
    signal_log.legacy.csv agar baris baru tidak tertulis di bawah header yang salah. True bila migrasi terjadi."""
    header = header or LOG_HEADER
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False
    with open(log_path, newline="", encoding="utf-8") as fh:
        first = next(csv.reader(fh), [])
    if first == header:
        return False
    legacy = log_path.with_name("signal_log.legacy.csv")
    n = 1
    while legacy.exists():
        legacy = log_path.with_name(f"signal_log.legacy{n}.csv")
        n += 1
    log_path.replace(legacy)
    log.warning("Skema jurnal lama terdeteksi -> dipindah ke %s; jurnal baru dimulai", legacy.name)
    return True


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="CryptoEliz Scalp Analysis Engine v2 (sctnv2)")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--symbols", default="", help="override simbol, pisah koma (contoh: BTC/USDT,ETH/USDT)")
    ap.add_argument("--demo", action="store_true", help="pakai data SINTETIS (tes pipeline, bukan pasar nyata)")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return ap.parse_args(argv)


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def run(args) -> int:
    _load_dotenv()
    out_dir = Path(args.output_dir)
    setup_logging(args.log_level, str(out_dir / "engine.log"))
    try:
        if args.symbols:
            os.environ["ENGINE_SYMBOLS"] = args.symbols
        cfg = load_config(args.config_dir)
    except ConfigError as exc:
        log.error("Konfigurasi bermasalah: %s", exc)
        return 2

    now = utcnow().floor("s")
    chash = config_hash(cfg.symbols_cfg, cfg.params, cfg.risk)
    dcfg = cfg.symbols_cfg.get("data", {})
    if args.demo:
        source = DemoSource()
        source_note = "**DEMO — data sintetis (bukan pasar nyata)**"
    else:
        source = OHLCVFetcher(dcfg.get("providers", ["binance_rest"]), int(dcfg.get("retries", 3)),
                              float(dcfg.get("backoff_seconds", 2.0)), int(dcfg.get("timeout_seconds", 20)),
                              bool(cfg.params.get("general", {}).get("drop_unclosed_candle", True)))
        source_note = "exchange publik via " + ", ".join(dcfg.get("providers", []))
    candles = {tf: int(cfg.symbols_cfg["candles"][tf]) for tf in cfg.all_timeframes}
    min_c = cfg.params.get("general", {}).get("min_candles", {})

    symbols = cfg.symbols
    bench = cfg.symbols_cfg.get("benchmark")
    if bench in symbols:
        symbols = [bench] + [s for s in symbols if s != bench]
    log.info("Mulai analisis %d simbol @ %s UTC (config %s)", len(symbols), fmt_ts(now), chash)

    analyses: Dict[str, SymbolAnalysis] = {}
    errors: Dict[str, str] = {}
    for sym in symbols:
        try:
            data = source.fetch_symbol(sym, candles, now=now, min_candles=min_c)
            analyses[sym] = analyze_symbol(sym, data, cfg, now)
            log.info("%s: analisis selesai (harga %.6g, bias %s)", sym, analyses[sym].price, analyses[sym].bias.direction)
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)[:400]
            log.error("%s: GAGAL — %s", sym, exc)
            log.debug(traceback.format_exc())

    btc_bias = analyses[bench].bias if bench in analyses else None
    signals: List[Signal] = []
    for sym in symbols:
        if sym in analyses:
            try:
                sig = evaluate_symbol(analyses[sym], cfg, btc_bias, now)
                log.info("%s: %s (skor %g/%g) %s", sym, sig.action, sig.score, sig.max_score, sig.status)
            except Exception as exc:  # noqa: BLE001
                log.exception("%s: evaluasi signal gagal", sym)
                errors[sym] = f"evaluasi gagal: {exc}"
                sig = Signal(sym, "NO TRADE", "-", now, analyses[sym].price, analyses[sym].bias.direction,
                             analyses[sym].bias.score, error=str(exc))
        else:
            sig = Signal(sym, "NO TRADE", "-", now, float("nan"), "-", 0.0, error=errors.get(sym, "unknown"))
        signals.append(sig)

    # ---- output ----
    latest = out_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    (latest / "charts").mkdir(parents=True, exist_ok=True)
    charts: Dict[str, str] = {}
    ci = cfg.chart_ind
    # --- v2: indikator pada grafik (tampilan & informasi saja; TIDAK masuk skor/signal/config_hash) ---
    suites_by_sym: Dict[str, dict] = {}
    if ci.get("enabled", True):
        for sig in signals:
            an = analyses.get(sig.symbol)
            if an is None or sig.error:
                continue
            try:
                suites_by_sym[sig.symbol] = build_suites(an, cfg)
                sig.context["indicators"] = summarize_symbol(an, cfg, suites_by_sym[sig.symbol])
            except Exception as exc:  # noqa: BLE001
                log.exception("%s: ringkasan indikator gagal: %s", sig.symbol, exc)
    if not args.no_charts:
        only_act = bool(cfg.params.get("chart", {}).get("only_actionable", False))
        for sig in signals:
            if sig.error or sig.symbol not in analyses or (only_act and sig.action == "NO TRADE"):
                continue
            an = analyses[sig.symbol]
            main_tf = chart_timeframe(an, sig, cfg)
            tf_list = [main_tf] + ([t for t in ci.get("extra_timeframes", []) if t != main_tf and t in an.frames]
                                   if ci.get("enabled", True) else [])
            sig.context["charts"] = {}
            for tf in tf_list:
                try:
                    rel = f"charts/{safe_name(sig.symbol)}.png" if tf == main_tf else f"charts/{safe_name(sig.symbol)}_{tf}.png"
                    if generate_chart(an, sig, cfg, latest / rel, tf=tf, suites=suites_by_sym.get(sig.symbol)):
                        sig.context["charts"][tf] = rel
                        if tf == main_tf:
                            charts[sig.symbol] = rel
                            sig.context["chart"] = rel
                            sig.context["chart_tf"] = tf
                except Exception as exc:  # noqa: BLE001
                    log.exception("%s %s: chart gagal: %s", sig.symbol, tf, exc)
    md = build_report(signals, analyses, charts, now, chash, source_note, errors)
    (latest / "report.md").write_text(md, encoding="utf-8")
    (latest / "signals.json").write_text(json.dumps(to_jsonable({
        "asof": now, "config_hash": chash, "demo": bool(args.demo), "source_note": source_note,
        "benchmark": bench, "min_score": cfg.params["scoring"]["min_score"], "signals": signals, "errors": errors}),
        indent=2, ensure_ascii=False), encoding="utf-8")

    new_sigs: List[Signal] = []
    if not args.demo:  # data sintetis TIDAK boleh mencemari jurnal forward test
        log_path = out_dir / "signal_log.csv"
        header = LOG_HEADER
        ensure_log_schema(log_path, header)
        known = set()
        if log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as fh:
                known = {r.get("signal_id") for r in csv.DictReader(fh)}
        fresh = [x for x in signals if x.action != "NO TRADE" and x.plan and x.signal_id and x.signal_id not in known]
        if fresh or not log_path.exists():
            is_new_file = not log_path.exists()
            with open(log_path, "a", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                if is_new_file:
                    wr.writerow(header)
                for x in fresh:
                    p = x.plan
                    wr.writerow([x.signal_id, now.isoformat(), x.symbol, x.action, x.status, x.score, x.price, p.entry,
                                 p.stop, round(p.stop_pct, 4), p.tp1, p.tp2, round(p.rr1, 4), round(p.rr2, 4),
                                 round(p.blended_rr, 4), p.tp1_close_pct, p.be_stop, x.primary.kind, x.primary.tf,
                                 x.bias, chash, 0])
        new_sigs = fresh
        log.info("Jurnal: %d signal baru dicatat", len(new_sigs))
        if cfg.params.get("notify", {}).get("enabled", True):
            send_alerts(new_sigs)
    if not args.no_archive:
        arch = out_dir / "archive" / now.strftime("%Y%m%d_%H%M")
        shutil.copytree(latest, arch, dirs_exist_ok=True)
    log.info("Selesai. Laporan: %s", latest / "report.md")

    if analyses:
        return 0
    log.error("Semua simbol gagal diambil/dianalisis.")
    return 1


def main():
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
