import csv
import tempfile
import unittest
from pathlib import Path

from src.main import LOG_HEADER, ensure_log_schema


class TestLogSchema(unittest.TestCase):
    def test_legacy_log_is_moved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "signal_log.csv"
            p.write_text("asof_utc,symbol,action\n2026-01-01,BTC/USDT,LONG\n", encoding="utf-8")
            self.assertTrue(ensure_log_schema(p))
            self.assertFalse(p.exists())
            self.assertTrue((Path(d) / "signal_log.legacy.csv").exists())

    def test_current_log_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "signal_log.csv"
            with open(p, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(LOG_HEADER)
            self.assertFalse(ensure_log_schema(p))
            self.assertTrue(p.exists())

    def test_missing_log(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(ensure_log_schema(Path(d) / "signal_log.csv"))


if __name__ == "__main__":
    unittest.main()
