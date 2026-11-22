"""Smoke & unit test (unittest). Jalankan: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.data.indicators import find_pivots, obv
from src.data.synthetic import DemoSource
from src.analysis.pipeline import analyze_symbol
from src.analysis.liquidity_deviation import detect_deviations
from src.analysis.fvg import detect_fvgs
from src.analysis.range_detector import detect_range
from src.models import Level, RangeBox
from src.signal.signal_engine import evaluate_symbol
from src.signal import risk_manager


def mk(closes, vol=100.0, spread=0.001, start="2026-01-01"):
    c = np.array(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + spread)
    l = np.minimum(o, c) * (1 - spread)
    idx = pd.date_range(start, periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": np.full(len(c), vol)}, index=idx)


class TestDetectors(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("config")

    def test_pivots(self):
        df = mk([1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1])
        kinds = [p.kind for p in find_pivots(df, 2, 2)]
        self.assertIn("H", kinds)

    def test_deviation_long(self):
        base = [100.0] * 60
        df = mk(base)
        # wick di bawah 100 lalu close kembali di atas
        df.iloc[-3, df.columns.get_loc("low")] = 99.3
        df.iloc[-3, df.columns.get_loc("close")] = 100.1
        lv = Level("Range Low (1h)", 100.0, "range_low", "1h", ("down",))
        f = detect_deviations(df, [lv], self.cfg.tf_params("deviation", "1h"), "1h")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].direction, "LONG")
        self.assertLess(f[0].stop_ref, 100.0)

    def test_no_deviation_when_closes_outside(self):
        df = mk([100.0] * 60)
        for i in (-4, -3, -2, -1):
            df.iloc[i, df.columns.get_loc("close")] = 98.5
            df.iloc[i, df.columns.get_loc("low")] = 98.4
        lv = Level("Range Low", 100.0, "range_low", "1h", ("down",))
        self.assertEqual(detect_deviations(df, [lv], self.cfg.tf_params("deviation", "1h"), "1h"), [])

    def test_fvg_bullish(self):
        df = mk([100] * 40)
        df.iloc[-3] = [100, 100.5, 99.5, 100.2, 100]
        df.iloc[-2] = [100.2, 103, 100.1, 102.9, 300]
        df.iloc[-1] = [102.9, 104, 102, 103.5, 100]
        zones, _ = detect_fvgs(df, self.cfg.tf_params("fvg", "1h"), 0.5, "1h")
        self.assertTrue(any(z.direction == "LONG" for z in zones))

    def test_range_on_oscillation(self):
        t = np.arange(240)
        c = 100 + 3 * np.sin(2 * np.pi * t / 40)
        df = mk(c, spread=0.0005)
        rb = detect_range(df, self.cfg.tf_params("range", "1h"), "1h")
        self.assertIsNotNone(rb)
        self.assertAlmostEqual(rb.mid, 100, delta=1.0)

    def test_range_position(self):
        rb = RangeBox("1h", 110, 100, 2, 2, pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"))
        self.assertEqual(rb.position(101), "demand_edge")
        self.assertEqual(rb.position(105), "mid")
        self.assertEqual(rb.position(109.5), "supply_edge")
        self.assertEqual(rb.position(111), "outside_up")

    def test_obv_sign(self):
        df = mk([1, 2, 3, 2, 2])
        self.assertEqual(obv(df).iloc[-1], 100.0)  # +100 +100 -100 +0


class TestRisk(unittest.TestCase):
    def test_wide_stop_vetoed(self):
        cfg = load_config("config")
        now = pd.Timestamp.now(tz="UTC")
        data = DemoSource().fetch_symbol("BTC/USDT", {tf: 500 for tf in cfg.all_timeframes}, now=now)
        an = analyze_symbol("BTC/USDT", data, cfg, now)
        from src.models import Finding
        f = Finding("deviation_reclaim", "trigger", "LONG", "1h", 0.5, entry=100.0, entry_zone=(99.8, 100.2),
                    stop_ref=80.0, state="waiting_retest")
        plan, vetoes, _ = risk_manager.build_plan(an, "LONG", f, 1.0, [(110.0, "x")], cfg.risk)
        self.assertIsNone(plan)
        self.assertTrue(any("batas" in v for v in vetoes))

    def test_plan_ok(self):
        cfg = load_config("config")
        now = pd.Timestamp.now(tz="UTC")
        data = DemoSource().fetch_symbol("BTC/USDT", {tf: 500 for tf in cfg.all_timeframes}, now=now)
        an = analyze_symbol("BTC/USDT", data, cfg, now)
        from src.models import Finding
        f = Finding("deviation_reclaim", "trigger", "SHORT", "1h", 0.5, entry=100.0, entry_zone=(99.8, 100.2),
                    stop_ref=102.0, state="waiting_retest")
        plan, vetoes, _ = risk_manager.build_plan(an, "SHORT", f, 0.5, [(95.0, "mid"), (90.0, "low")], cfg.risk)
        self.assertIsNotNone(plan, vetoes)
        self.assertGreater(plan.stop, 102.0)
        self.assertLess(plan.tp1, plan.entry)
        self.assertLess(plan.tp2, plan.tp1)


class TestEndToEnd(unittest.TestCase):
    def test_demo_all_symbols(self):
        cfg = load_config("config")
        now = pd.Timestamp.now(tz="UTC")
        src = DemoSource()
        candles = {tf: int(cfg.symbols_cfg["candles"][tf]) for tf in cfg.all_timeframes}
        for sym in cfg.symbols[:3]:
            an = analyze_symbol(sym, src.fetch_symbol(sym, candles, now=now), cfg, now)
            sig = evaluate_symbol(an, cfg, None, now)
            self.assertIn(sig.action, ("LONG", "SHORT", "NO TRADE"))
            if sig.action != "NO TRADE":
                self.assertIsNotNone(sig.plan)
                self.assertGreaterEqual(sig.score, cfg.params["scoring"]["min_score"])
                self.assertLessEqual(sig.plan.stop_pct, cfg.risk["stop"]["max_stop_pct"])
            else:
                self.assertTrue(sig.reasons)


if __name__ == "__main__":
    unittest.main()


class TestEvaluate(unittest.TestCase):
    EV = {"fee_pct_roundtrip": 0.0, "pending_expiry_hours": 24, "max_hold_hours": 72}

    @staticmethod
    def candles(rows, start="2026-01-01 00:00"):
        idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
        return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)

    def sig(self, **kw):
        s = {"direction": "LONG", "status": "ENTRY ZONE", "asof": "2026-01-01 00:00", "entry": 100.0, "stop": 99.0,
             "tp1": 101.5, "tp2": 103.0, "tp1_close_pct": 60, "be_stop": 100.1, "stop_pct": 1.0}
        s.update(kw)
        return s

    def test_loss(self):
        from src.evaluate.simulator import simulate_trade
        r = simulate_trade(self.sig(), self.candles([[100, 100.5, 98.9, 99.0]]), self.EV)
        self.assertEqual((r["status"], r["outcome"], r["r"]), ("CLOSED", "LOSS", -1.0))

    def test_tp2_path(self):
        from src.evaluate.simulator import simulate_trade
        c = self.candles([[100, 101.6, 99.8, 101.5], [101.5, 103.2, 101.0, 103.0]])
        r = simulate_trade(self.sig(), c, self.EV)
        self.assertEqual(r["outcome"], "TP2")
        self.assertAlmostEqual(r["r"], 0.6 * 1.5 + 0.4 * 3.0, places=3)

    def test_same_candle_stop_first(self):
        from src.evaluate.simulator import simulate_trade
        r = simulate_trade(self.sig(), self.candles([[100, 102, 98.5, 101]]), self.EV)
        self.assertEqual(r["outcome"], "LOSS")  # ambigu -> stop dulu (konservatif)

    def test_tp1_then_be(self):
        from src.evaluate.simulator import simulate_trade
        c = self.candles([[100, 101.6, 99.8, 101.4], [101.4, 101.5, 100.0, 100.2]])
        r = simulate_trade(self.sig(), c, self.EV)
        self.assertEqual(r["outcome"], "TP1_BE")
        self.assertGreater(r["r"], 0.8)

    def test_pending_cancelled_when_tp1_first(self):
        from src.evaluate.simulator import simulate_trade
        s = self.sig(status="PENDING LIMIT", entry=99.5, stop=98.5, tp1=101.0, tp2=102.0, be_stop=99.6)
        r = simulate_trade(s, self.candles([[100.2, 101.2, 100.0, 101.1]]), self.EV)
        self.assertEqual(r["status"], "CANCELLED")

    def test_pending_waits(self):
        from src.evaluate.simulator import simulate_trade
        s = self.sig(status="PENDING LIMIT", entry=99.5, stop=98.5, tp1=101.0, tp2=102.0, be_stop=99.6)
        r = simulate_trade(s, self.candles([[100.2, 100.4, 100.0, 100.1]]), self.EV)
        self.assertEqual(r["status"], "WAITING")

    def test_fee_reduces_r(self):
        from src.evaluate.simulator import simulate_trade
        ev = dict(self.EV, fee_pct_roundtrip=0.1)
        r = simulate_trade(self.sig(), self.candles([[100, 100.5, 98.9, 99.0]]), ev)
        self.assertAlmostEqual(r["r"], -1.1, places=3)

    def test_stats_unverifiable(self):
        from src.evaluate.stats import compute_stats
        st = compute_stats([1.0, -1.0, 2.0], [pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"),
                                              pd.Timestamp("2026-01-03", tz="UTC")], {"min_trades": 150, "min_weeks": 20})
        self.assertEqual(st["status"], "UNVERIFIABLE")
        self.assertAlmostEqual(st["avg_r"], 2 / 3, places=4)

    def test_signal_id_stable(self):
        from src.utils import signal_id
        from src.models import Finding, Signal
        t = pd.Timestamp("2026-01-01", tz="UTC")
        f = Finding("deviation_reclaim", "trigger", "LONG", "1h", meta={"deviation_time": t, "level_kind": "range_low"}, time=t)
        s1 = Signal("BTC/USDT", "LONG", "ENTRY ZONE", t, 1.0, "LONG", 0.5, primary=f)
        f2 = Finding("deviation_reclaim", "trigger", "LONG", "1h", entry=123.0, meta={"deviation_time": t, "level_kind": "range_low"}, time=t)
        s2 = Signal("BTC/USDT", "LONG", "PENDING LIMIT", t, 2.0, "LONG", 0.5, primary=f2)
        self.assertEqual(signal_id(s1), signal_id(s2))


class TestSite(unittest.TestCase):
    def test_build_site(self):
        import tempfile, pathlib
        from src.site_builder import build
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "output"
            (out / "latest").mkdir(parents=True)
            (out / "latest" / "signals.json").write_text("{}")
            (out / "signal_log.csv").write_text("a,b\n")
            site = build(str(out), str(pathlib.Path(d) / "site"), "index.html")
            self.assertTrue((site / "index.html").exists())
            self.assertTrue((site / "data" / "latest" / "signals.json").exists())
            self.assertTrue((site / "data" / "signal_log.csv").exists())
