"""Tes indikator chart v2 (sctnv2). Jalankan: python -m unittest discover -s tests -v"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.pipeline import analyze_symbol
from src.config_loader import load_config
from src.data.indicators import macd, rsi, supertrend, volume_profile
from src.data.key_levels import compute_key_levels
from src.data.synthetic import DemoSource
from src.models import MondayRange
from src.report.chart_generator import chart_timeframe, generate_chart
from src.report.indicator_suite import build_suites, compute_suite, summarize_symbol
from src.signal.signal_engine import evaluate_symbol
from src.utils import config_hash, to_jsonable


def frame(closes, vol=100.0, spread=0.002, start="2026-01-01", freq="1h"):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range(start, periods=len(c), freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + spread), "low": np.minimum(o, c) * (1 - spread),
                         "close": c, "volume": np.full(len(c), vol)}, index=idx)


class TestPrimitives(unittest.TestCase):
    def test_rsi_extremes(self):
        up = pd.Series(np.linspace(100, 200, 100))
        self.assertEqual(rsi(up).iloc[-1], 100.0)
        self.assertAlmostEqual(rsi(up[::-1].reset_index(drop=True)).iloc[-1], 0.0, places=6)

    def test_rsi_bounded(self):
        rng = np.random.default_rng(1)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
        r = rsi(s).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_macd_identity(self):
        rng = np.random.default_rng(2)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
        line, sig, hist = macd(s, 12, 26, 9)
        np.testing.assert_allclose(hist.to_numpy(), (line - sig).to_numpy())

    def test_supertrend_direction(self):
        up = frame(np.linspace(100, 200, 200))
        dn = frame(np.linspace(200, 100, 200))
        self.assertEqual(int(supertrend(up)[1].iloc[-1]), 1)
        self.assertEqual(int(supertrend(dn)[1].iloc[-1]), -1)
        st, d = supertrend(up)
        self.assertLess(st.iloc[-1], up["close"].iloc[-1])  # uptrend: garis di bawah harga
        st, d = supertrend(dn)
        self.assertGreater(st.iloc[-1], dn["close"].iloc[-1])  # downtrend: garis di atas harga

    def test_volume_profile_conserves_and_finds_poc(self):
        df = frame(np.full(60, 100.0), vol=10.0, spread=0.0005)
        df.iloc[20:30, df.columns.get_loc("volume")] = 500.0  # volume besar di sekitar 100
        df.iloc[0, df.columns.get_loc("high")] = 110.0       # bentangkan range ke atas
        vp = volume_profile(df, bins=40, value_area=0.7)
        self.assertAlmostEqual(vp["volumes"].sum(), df["volume"].sum(), places=6)
        self.assertAlmostEqual(vp["poc"], 100.0, delta=0.6)
        self.assertLessEqual(vp["val"], vp["poc"])
        self.assertGreaterEqual(vp["vah"], vp["poc"])

    def test_volume_profile_too_short(self):
        self.assertIsNone(volume_profile(frame([1, 2, 3])))


class TestKeyLevels(unittest.TestCase):
    def setUp(self):
        # 120 hari harian + 1h/4h/15m yang mencakup periode berjalan; asof = Rabu 2026-04-15 10:05 UTC
        self.asof = pd.Timestamp("2026-04-15 10:05", tz="UTC")
        rng = np.random.default_rng(3)
        idx_d = pd.date_range("2025-12-01", "2026-04-14", freq="1D", tz="UTC")
        c = 100 + np.cumsum(rng.normal(0, 1, len(idx_d)))
        self.d1 = pd.DataFrame({"open": c, "high": c + 2, "low": c - 2, "close": c, "volume": 1.0}, index=idx_d)

        def mk(freq, start):
            idx = pd.date_range(start, "2026-04-15 09:00", freq=freq, tz="UTC")
            v = np.linspace(50, 60, len(idx))
            return pd.DataFrame({"open": v, "high": v + 1, "low": v - 1, "close": v, "volume": 1.0}, index=idx)
        self.frames = {"1d": self.d1, "4h": mk("4h", "2026-01-01"), "1h": mk("1h", "2026-03-10"), "15m": mk("15min", "2026-04-10")}

    def test_previous_periods(self):
        kl = {(k.group, k.kind): k.price for k in compute_key_levels(self.frames, self.asof)}
        d1 = self.d1
        yday = d1.loc["2026-04-14"]
        self.assertAlmostEqual(kl[("day", "high")], float(yday["high"]))
        self.assertAlmostEqual(kl[("day", "low")], float(yday["low"]))
        pw = d1.loc["2026-04-06":"2026-04-12"]  # Senin..Minggu sebelum minggu ini (Sen 04-13)
        self.assertAlmostEqual(kl[("week", "high")], float(pw["high"].max()))
        self.assertAlmostEqual(kl[("week", "low")], float(pw["low"].min()))
        self.assertAlmostEqual(kl[("week", "mid")], (float(pw["high"].max()) + float(pw["low"].min())) / 2)
        pm = d1.loc["2026-03-01":"2026-03-31"]
        self.assertAlmostEqual(kl[("month", "high")], float(pm["high"].max()))
        self.assertAlmostEqual(kl[("month", "low")], float(pm["low"].min()))

    def test_current_opens(self):
        kl = {(k.group, k.kind): k.price for k in compute_key_levels(self.frames, self.asof)}
        f = self.frames
        self.assertAlmostEqual(kl[("day", "open")], float(f["15m"].loc["2026-04-15 00:00", "open"]))
        self.assertAlmostEqual(kl[("week", "open")], float(f["1h"].loc["2026-04-13 00:00", "open"]))
        self.assertAlmostEqual(kl[("month", "open")], float(f["4h"].loc["2026-04-01 00:00", "open"]))

    def test_year_open(self):
        kl = {(k.group, k.kind): k.price for k in compute_key_levels(self.frames, self.asof)}
        self.assertAlmostEqual(kl[("year", "open")], float(self.d1.loc["2026-01-01", "open"]))
        late = {"1d": self.d1.loc["2026-02-01":]}  # data tidak mencapai 1 Januari -> level dilewati, bukan ditebak
        self.assertNotIn(("year", "open"), {(k.group, k.kind) for k in compute_key_levels(late, self.asof)})

    def test_monday_and_show_filter(self):
        mr = MondayRange(high=120.0, low=110.0, date=pd.Timestamp("2026-04-13", tz="UTC"))
        only = compute_key_levels(self.frames, self.asof, mr, {"monday": ["high", "mid"]})
        self.assertEqual({(k.group, k.kind) for k in only}, {("monday", "high"), ("monday", "mid")})
        self.assertAlmostEqual([k for k in only if k.kind == "mid"][0].price, 115.0)

    def test_partial_period_not_reported(self):
        short = {"1d": self.d1.loc["2026-04-10":]}  # minggu lalu cuma 3 hari, bulan lalu kosong
        kl = {(k.group, k.kind) for k in compute_key_levels(short, self.asof)}
        self.assertNotIn(("week", "high"), kl)
        self.assertNotIn(("month", "high"), kl)
        self.assertIn(("day", "high"), kl)  # hari sebelumnya tetap ada


class TestSuiteAndChart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("config")
        cls.now = pd.Timestamp.now(tz="UTC").floor("s")
        data = DemoSource().fetch_symbol("SOL/USDT", {tf: int(cls.cfg.symbols_cfg["candles"][tf]) for tf in cls.cfg.all_timeframes},
                                         now=cls.now)
        cls.an = analyze_symbol("SOL/USDT", data, cls.cfg, cls.now)
        cls.sig = evaluate_symbol(cls.an, cls.cfg, cls.an.bias, cls.now)

    def test_ucts_proxy_buy_after_reversal(self):
        c = np.concatenate([np.linspace(200, 100, 80), np.linspace(100, 220, 80)])
        s = compute_suite(frame(c), self.cfg.chart_ind, "1h")
        self.assertTrue(s.ucts_buy.any())
        first_buy = int(np.flatnonzero(s.ucts_buy.to_numpy())[0])
        self.assertGreater(first_buy, 80)  # buy muncul SETELAH pembalikan, bukan saat tren turun
        self.assertFalse(s.ucts_sell.iloc[100:].any())

    def test_suite_lengths(self):
        s = compute_suite(self.an.frames["1h"], self.cfg.chart_ind, "1h")
        n = len(self.an.frames["1h"])
        for ser in (s.rsi, s.macd, s.macd_hist, s.obv, s.st, s.st_dir, *s.ema.values()):
            self.assertEqual(len(ser), n)
        self.assertEqual(sorted(s.ema), [12, 26, 50, 100, 200])

    def test_summary_is_json_serialisable(self):
        suites = build_suites(self.an, self.cfg)
        summ = summarize_symbol(self.an, self.cfg, suites)
        blob = json.dumps(to_jsonable(summ))
        back = json.loads(blob)
        self.assertEqual(set(back["timeframes"]), {"15m", "1h", "4h"})
        self.assertIn("rsi", back["timeframes"]["1h"])
        self.assertIn("note", back)
        self.assertTrue(back["key_levels"])

    def test_chart_all_timeframes(self):
        suites = build_suites(self.an, self.cfg)
        with tempfile.TemporaryDirectory() as d:
            for tf in ("15m", "1h", "4h"):
                out = Path(d) / f"c_{tf}.png"
                self.assertTrue(generate_chart(self.an, self.sig, self.cfg, out, tf=tf, suites=suites))
                self.assertGreater(out.stat().st_size, 20_000)

    def test_chart_with_indicators_disabled_and_partial_panels(self):
        cfg = copy.deepcopy(self.cfg)
        with tempfile.TemporaryDirectory() as d:
            cfg.chart_ind["enabled"] = False
            out = Path(d) / "plain.png"
            self.assertTrue(generate_chart(self.an, self.sig, cfg, out))
            cfg2 = copy.deepcopy(self.cfg)
            cfg2.chart_ind["panels"] = {"volume": False, "obv": False, "rsi": True, "macd": False}
            cfg2.chart_ind["ema"]["periods"] = [20, 50]
            cfg2.chart_ind["key_levels"]["enabled"] = False
            out2 = Path(d) / "partial.png"
            self.assertTrue(generate_chart(self.an, self.sig, cfg2, out2))
            self.assertTrue(out.exists() and out2.exists())

    def test_chart_timeframe_resolution(self):
        self.assertIn(chart_timeframe(self.an, self.sig, self.cfg), self.an.frames)


class TestDisplayOnlyInvariant(unittest.TestCase):
    """Indikator chart tidak boleh mempengaruhi skor/signal/forward test."""

    def test_signal_code_never_imports_indicator_layer(self):
        root = Path("src")
        banned = ("indicator_suite", "chart_ind", "key_levels", "chart_generator")
        for sub in ("analysis", "signal", "evaluate", "backtest"):
            for f in (root / sub).rglob("*.py"):
                txt = f.read_text(encoding="utf-8")
                for b in banned:
                    self.assertNotIn(b, txt, f"{f} menyentuh lapisan indikator ({b}) — harus tampilan saja")

    def test_config_hash_ignores_chart_indicators(self):
        base = load_config("config")
        h0 = config_hash(base.symbols_cfg, base.params, base.risk)
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree("config", Path(d) / "config")
            p = Path(d) / "config" / "chart_indicators.yaml"
            p.write_text(p.read_text(encoding="utf-8").replace("periods: [12, 26, 50, 100, 200]", "periods: [9, 21]")
                         .replace("multiplier: 3.0", "multiplier: 2.0"), encoding="utf-8")
            alt = load_config(Path(d) / "config")
        self.assertEqual(alt.chart_ind["ema"]["periods"], [9, 21])
        self.assertEqual(h0, config_hash(alt.symbols_cfg, alt.params, alt.risk))

    def test_signal_identical_with_indicators_toggled(self):
        cfg_a = load_config("config")
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b.chart_ind["enabled"] = False
        now = pd.Timestamp.now(tz="UTC").floor("s")
        data = DemoSource().fetch_symbol("ETH/USDT", {tf: int(cfg_a.symbols_cfg["candles"][tf]) for tf in cfg_a.all_timeframes}, now=now)
        sa = evaluate_symbol(analyze_symbol("ETH/USDT", data, cfg_a, now), cfg_a, None, now)
        sb = evaluate_symbol(analyze_symbol("ETH/USDT", data, cfg_b, now), cfg_b, None, now)
        self.assertEqual((sa.action, sa.score, sa.status, sa.signal_id), (sb.action, sb.score, sb.status, sb.signal_id))


if __name__ == "__main__":
    unittest.main()
