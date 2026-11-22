"""Evaluator forward test: baca output/signal_log.csv, simulasikan hasil tiap signal dengan candle 15m aktual,
simpan output/forward_test.json. Trade yang sudah CLOSED di-cache (tidak dihitung ulang).
Jalankan: python -m src.evaluate.forward_test"""
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

from ..config_loader import load_config
from ..data.fetch_ohlcv import OHLCVFetcher
from ..logging_setup import setup_logging
from ..utils import to_jsonable, utcnow
from .simulator import simulate_trade
from .stats import compute_stats, group_stats

log = logging.getLogger("engine.forward")
NUM = ("price", "entry", "stop", "tp1", "tp2", "rr1", "rr2", "blended_rr", "tp1_close_pct", "be_stop", "stop_pct", "score")


def read_log(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("demo", "0")) == "1":
                continue
            for k in NUM:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = None
            rows.append(r)
    return rows


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--source", default="live", choices=["live", "demo"])
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    out = Path(a.output_dir)
    setup_logging(os.getenv("LOG_LEVEL", "INFO"), str(out / "engine.log"))
    cfg = load_config(a.config_dir)
    ev = cfg.risk.get("evaluation", {})
    now = utcnow().floor("s")
    rows = read_log(out / "signal_log.csv")
    ft_path = out / "forward_test.json"
    cache: Dict[str, dict] = {}
    if ft_path.exists():
        try:
            for t in json.loads(ft_path.read_text(encoding="utf-8")).get("trades", []):
                if t.get("status") in ("CLOSED", "CANCELLED"):
                    cache[t["signal_id"]] = t
        except Exception as exc:  # noqa: BLE001
            log.warning("forward_test.json lama tidak terbaca: %s", exc)

    pending = [r for r in rows if r["signal_id"] not in cache]
    by_sym: Dict[str, List[dict]] = {}
    for r in pending:
        by_sym.setdefault(r["symbol"], []).append(r)

    if a.source == "demo":
        from ..data.synthetic import DemoSource
        source = DemoSource()
    else:
        d = cfg.symbols_cfg.get("data", {})
        source = OHLCVFetcher(d.get("providers", ["binance_rest"]), int(d.get("retries", 3)),
                              float(d.get("backoff_seconds", 2.0)), int(d.get("timeout_seconds", 20)))
    results: Dict[str, dict] = dict(cache)
    for sym, rs in by_sym.items():
        t0 = min(pd.Timestamp(r["asof_utc"]) for r in rs)
        n = int(math.ceil((now - t0).total_seconds() / 900)) + 8
        try:
            df = source.fetch_symbol(sym, {"15m": min(n, 4000)}, now=now)["15m"]
        except Exception as exc:  # noqa: BLE001
            log.error("%s: gagal ambil candle evaluasi: %s", sym, exc)
            continue
        for r in rs:
            s = dict(r, direction=r["action"], asof=r["asof_utc"])
            try:
                res = simulate_trade(s, df, ev)
            except Exception as exc:  # noqa: BLE001
                log.exception("simulasi gagal %s: %s", r["signal_id"], exc)
                continue
            results[r["signal_id"]] = {
                "signal_id": r["signal_id"], "asof": r["asof_utc"], "symbol": sym, "direction": r["action"],
                "trigger": r["trigger"], "tf": r["trigger_tf"], "entry": r["entry"], "stop": r["stop"],
                "tp1": r["tp1"], "tp2": r["tp2"], "score": r["score"], **res}
    trades = sorted(results.values(), key=lambda t: t["asof"])
    closed = [t for t in trades if t["status"] == "CLOSED"]
    stats = compute_stats([t["r"] for t in closed], [pd.Timestamp(t["asof"]) for t in closed], ev)
    doc = {"updated": now, "stats": stats, "by_trigger": group_stats(closed, "trigger", ev),
           "by_symbol": group_stats(closed, "symbol", ev),
           "counts": {k: sum(1 for t in trades if t["status"] == k) for k in ("CLOSED", "OPEN", "WAITING", "CANCELLED")},
           "assumptions": "Simulasi konservatif (stop dulu bila ambigu), fee/slippage kasar, entry pending kedaluwarsa. Bukan hasil eksekusi nyata.",
           "trades": trades}
    out.mkdir(parents=True, exist_ok=True)
    ft_path.write_text(json.dumps(to_jsonable(doc), indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("Forward test: %d trade (%d closed) — status %s", len(trades), len(closed), stats["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
