"""Tes scanner universe (tanpa jaringan: HTTP dipalsukan)."""
import copy
import csv
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.data.universe import (UniverseError, fetch_cmc_ranks, hash_basis, resolve_universe, rules_hash,
                               scanner_active, scanner_cfg)
from src.utils import config_hash

NOW = pd.Timestamp("2026-11-23 00:05", tz="UTC")


class R:
    def __init__(self, data, code=200):
        self._d, self.status_code = data, code

    def json(self):
        return self._d


def make_get(routes):
    """routes: {substring_url: payload | Exception | callable}"""
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        for key, resp in routes.items():
            if key in url:
                resp = resp() if callable(resp) else resp
                if isinstance(resp, Exception):
                    raise resp
                return resp if isinstance(resp, R) else R(resp)
        return R({}, 404)

    get.calls = calls
    return get


BINANCE = [
    {"symbol": "BTCUSDT", "quoteVolume": "900000000"}, {"symbol": "ETHUSDT", "quoteVolume": "500000000"},
    {"symbol": "USDCUSDT", "quoteVolume": "800000000"},   # stablecoin -> keluar
    {"symbol": "BTCUPUSDT", "quoteVolume": "300000000"},  # leveraged (root BTC terdaftar) -> keluar
    {"symbol": "SOLUSDT", "quoteVolume": "300000000"}, {"symbol": "JUPUSDT", "quoteVolume": "60000000"},  # JUP = koin asli
    {"symbol": "XRPUSDT", "quoteVolume": "200000000"}, {"symbol": "TINYUSDT", "quoteVolume": "1000000"},  # < min volume
    {"symbol": "BTCEUR", "quoteVolume": "999999999"},     # bukan pasangan USDT
    {"symbol": "PAXGUSDT", "quoteVolume": "400000000"},   # token emas -> keluar
]
BITGET = {"code": "00000", "data": [
    {"symbol": "BTCUSDT", "usdtVolume": "700000000", "quoteVolume": "1"}, {"symbol": "DOGEUSDT", "usdtVolume": "250000000"},
    {"symbol": "ETHUSDT", "usdtVolume": "400000000"}, {"symbol": "USDCUSDT", "usdtVolume": "900000000"}]}
CMC = {"status": {"error_code": 0}, "data": [
    {"symbol": "BTC", "cmc_rank": 1}, {"symbol": "ETH", "cmc_rank": 2}, {"symbol": "USDT", "cmc_rank": 3},
    {"symbol": "XRP", "cmc_rank": 4}, {"symbol": "BNB", "cmc_rank": 5}, {"symbol": "NOEX", "cmc_rank": 6},
    {"symbol": "SOL", "cmc_rank": 7}]}


def cfg_with(**scanner):
    cfg = copy.deepcopy(load_config("config"))
    cfg.symbols_cfg["scanner"] = {"enabled": True, "top_n": 3, "min_quote_volume_usdt": 5_000_000, **scanner}
    cfg.symbols_overridden = False
    return cfg


class TestSelection(unittest.TestCase):
    def test_binance_volume_filters_and_orders(self):
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["binance_volume"], top_n=4), NOW, Path(d),
                                 http_get=make_get({"binance": BINANCE}), env={})
        self.assertEqual(u.symbols, ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
        self.assertEqual((u.mode, u.source, u.status), ("scanner", "binance_volume", "fresh"))
        for bad in ("USDC/USDT", "BTCUP/USDT", "PAXG/USDT", "TINY/USDT", "BTCEUR/USDT"):
            self.assertNotIn(bad, u.symbols)

    def test_jup_is_not_mistaken_for_leveraged(self):
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["binance_volume"], top_n=10), NOW, Path(d),
                                 http_get=make_get({"binance": BINANCE}), env={})
        self.assertIn("JUP/USDT", u.symbols)

    def test_bitget_volume(self):
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["bitget_volume"], top_n=3), NOW, Path(d),
                                 http_get=make_get({"bitget": BITGET}), env={})
        self.assertEqual(u.symbols, ["BTC/USDT", "ETH/USDT", "DOGE/USDT"])
        self.assertEqual(u.volumes["BTC/USDT"], 700000000.0)

    def test_cmc_rank_requires_exchange_availability_and_key(self):
        routes = {"coinmarketcap": CMC, "binance": [r for r in BINANCE if r["symbol"] != "BNBUSDT"]}
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["coinmarketcap_rank"], top_n=4), NOW, Path(d),
                                 http_get=make_get(routes), env={"CMC_API_KEY": "k"})
        # USDT (stable), BNB & NOEX (tidak ada pasangan di bursa) dilewati
        self.assertEqual(u.symbols, ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"])
        self.assertEqual(u.ranks["BTC/USDT"], 1)
        with self.assertRaises(UniverseError):
            fetch_cmc_ranks(make_get({}), "", 5, 10)

    def test_cmc_error_status_raises(self):
        bad = {"status": {"error_code": 1001, "error_message": "API key invalid"}, "data": None}
        with self.assertRaises(UniverseError):
            fetch_cmc_ranks(make_get({"coinmarketcap": bad}), "k", 5, 10)

    def test_benchmark_first_and_always_include(self):
        data = [dict(r, quoteVolume="1000") if r["symbol"] == "BTCUSDT" else r for r in BINANCE]  # BTC terdaftar tapi tak lolos peringkat
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["binance_volume"], top_n=2, always_include=["TRUMP/USDT"]), NOW, Path(d),
                                 http_get=make_get({"binance": data}), env={})
        self.assertEqual(u.symbols[0], "BTC/USDT")  # benchmark selalu ada & di depan
        self.assertIn("TRUMP/USDT", u.symbols)
        self.assertEqual(u.symbols, ["BTC/USDT", "ETH/USDT", "SOL/USDT", "TRUMP/USDT"])


class TestCacheAndFallback(unittest.TestCase):
    def test_cache_reuse_refresh_and_history(self):
        data = {"rows": list(BINANCE)}
        get = make_get({"binance": lambda: data["rows"]})
        cfg = cfg_with(sources=["binance_volume"], top_n=3, refresh_hours=24)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            u1 = resolve_universe(cfg, NOW, d, http_get=get, env={})
            n1 = len(get.calls)
            u2 = resolve_universe(cfg, NOW + timedelta(hours=3), d, http_get=get, env={})
            self.assertEqual((u1.status, u2.status), ("fresh", "cached"))
            self.assertEqual(len(get.calls), n1, "cache valid -> tidak boleh ada HTTP")
            self.assertEqual(u1.symbols, u2.symbols)
            data["rows"] = [r for r in BINANCE if r["symbol"] != "ETHUSDT"]  # ETH keluar, XRP masuk
            u3 = resolve_universe(cfg, NOW + timedelta(hours=25), d, http_get=get, env={})
            self.assertEqual(u3.status, "fresh")
            self.assertNotIn("ETH/USDT", u3.symbols)
            with (d / "universe_history.csv").open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertIn("ETH/USDT", rows[1]["removed"])
            self.assertIn("XRP/USDT", rows[1]["added"])

    def test_rules_change_invalidates_cache(self):
        get = make_get({"binance": BINANCE})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            resolve_universe(cfg_with(sources=["binance_volume"], top_n=3), NOW, d, http_get=get, env={})
            u = resolve_universe(cfg_with(sources=["binance_volume"], top_n=4), NOW + timedelta(hours=1), d, http_get=get, env={})
        self.assertEqual(u.status, "fresh")
        self.assertEqual(len([s for s in u.symbols]), 4)

    def test_source_fallback_order(self):
        get = make_get({"binance": RuntimeError("451 geo-block"), "bitget": BITGET})
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg_with(sources=["binance_volume", "bitget_volume"], top_n=2), NOW, Path(d), http_get=get, env={})
        self.assertEqual((u.source, u.status), ("bitget_volume", "fresh"))
        self.assertTrue(any("binance_volume" in w for w in u.warnings))

    def test_all_fail_uses_stale_cache_then_static(self):
        cfg = cfg_with(sources=["binance_volume"], top_n=3)
        dead = make_get({"binance": RuntimeError("down")})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            u0 = resolve_universe(cfg, NOW, d, http_get=dead, env={})  # belum ada cache
            self.assertEqual(u0.status, "fallback_static")
            self.assertEqual(u0.source, "static_fallback")
            self.assertEqual(set(u0.symbols), set(cfg.symbols) | {cfg.symbols_cfg["benchmark"]})
            resolve_universe(cfg, NOW, d, http_get=make_get({"binance": BINANCE}), env={})  # buat cache
            u1 = resolve_universe(cfg, NOW + timedelta(hours=30), d, http_get=dead, env={})
            self.assertEqual(u1.status, "stale_cache")
            self.assertEqual(u1.symbols, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
            self.assertTrue(u1.warnings)

    def test_stale_cache_from_different_rules_is_not_used(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            resolve_universe(cfg_with(sources=["binance_volume"], top_n=3), NOW, d, http_get=make_get({"binance": BINANCE}), env={})
            u = resolve_universe(cfg_with(sources=["binance_volume"], top_n=5), NOW + timedelta(hours=30), d,
                                 http_get=make_get({"binance": RuntimeError("down")}), env={})
        self.assertEqual(u.status, "fallback_static")

    def test_override_and_demo_bypass_scanner(self):
        get = make_get({"binance": BINANCE})
        cfg = cfg_with(sources=["binance_volume"])
        cfg.symbols_overridden = True
        cfg.symbols_cfg["symbols"] = ["ETH/USDT"]
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg, NOW, Path(d), http_get=get, env={})
            self.assertEqual((u.mode, u.source, u.symbols), ("static", "manual_override", ["ETH/USDT"]))
            cfg.symbols_overridden = False
            u2 = resolve_universe(cfg, NOW, Path(d), http_get=get, env={}, force_static=True)
            self.assertEqual((u2.mode, u2.source), ("static", "demo"))
        self.assertEqual(get.calls, [])

    def test_disabled_scanner_is_static(self):
        cfg = cfg_with()
        cfg.symbols_cfg["scanner"]["enabled"] = False
        with tempfile.TemporaryDirectory() as d:
            u = resolve_universe(cfg, NOW, Path(d), http_get=make_get({}), env={})
        self.assertEqual((u.mode, u.symbols), ("static", list(cfg.symbols)))


class TestConfigHash(unittest.TestCase):
    def test_disabled_scanner_hash_identical_to_legacy(self):
        cfg = load_config("config")
        legacy = {k: v for k, v in cfg.symbols_cfg.items() if k != "scanner"}
        cfg.symbols_cfg["scanner"]["enabled"] = False
        self.assertEqual(config_hash(hash_basis(cfg.symbols_cfg, False), cfg.params, cfg.risk),
                         config_hash(legacy, cfg.params, cfg.risk))

    def test_active_hash_tracks_rules_not_symbols(self):
        cfg = cfg_with(sources=["binance_volume"], top_n=30)
        h = lambda c: config_hash(hash_basis(c.symbols_cfg, True), c.params, c.risk)  # noqa: E731
        h0 = h(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2.symbols_cfg["symbols"] = ["BTC/USDT"]           # daftar statis (fallback) tidak mengubah hash
        self.assertEqual(h0, h(cfg2))
        cfg3 = copy.deepcopy(cfg)
        cfg3.symbols_cfg["scanner"]["top_n"] = 20            # aturan berubah -> hash berubah
        self.assertNotEqual(h0, h(cfg3))
        self.assertNotEqual(rules_hash(cfg.symbols_cfg), rules_hash(cfg3.symbols_cfg))

    def test_scanner_active_respects_override(self):
        cfg = cfg_with()
        self.assertTrue(scanner_active(cfg))
        cfg.symbols_overridden = True
        self.assertFalse(scanner_active(cfg))

    def test_defaults_merge(self):
        s = scanner_cfg({"scanner": {"top_n": 10}})
        self.assertEqual(s["top_n"], 10)
        self.assertIn("USDC", s["exclude"])
        self.assertEqual(s["refresh_hours"], 24)


if __name__ == "__main__":
    unittest.main()
