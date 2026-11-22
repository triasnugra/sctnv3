"""Bangun folder site/ (untuk GitHub Pages): index.html + data/. Jalankan: python -m src.site_builder"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build(output_dir="output", site_dir="site", index="index.html") -> Path:
    out, site = Path(output_dir), Path(site_dir)
    if site.exists():
        shutil.rmtree(site)
    (site / "data").mkdir(parents=True)
    src_index = Path(index)
    if not src_index.exists():
        raise FileNotFoundError("index.html tidak ditemukan di root repo")
    shutil.copy(src_index, site / "index.html")
    (site / ".nojekyll").write_text("")
    if (out / "latest").exists():
        shutil.copytree(out / "latest", site / "data" / "latest")
    for name in ("signal_log.csv", "forward_test.json"):
        if (out / name).exists():
            shutil.copy(out / name, site / "data" / name)
    bt = out / "backtest"
    if bt.exists():
        (site / "data" / "backtest").mkdir()
        for name in ("summary.json", "trades.csv"):
            if (bt / name).exists():
                shutil.copy(bt / name, site / "data" / "backtest" / name)
    return site


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--site-dir", default="site")
    a = ap.parse_args()
    print("Site dibangun di:", build(a.output_dir, a.site_dir))
