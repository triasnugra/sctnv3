"""Loader konfigurasi YAML + validasi dasar."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


class ConfigError(Exception):
    pass


# Default indikator chart (v2). Kunci yang tidak ada di config/chart_indicators.yaml memakai nilai ini.
CHART_IND_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "extra_timeframes": ["15m", "1h", "4h"],
    "summary_timeframes": ["15m", "1h", "4h"],
    "mtf_timeframes": ["15m", "1h", "4h", "1d"],
    "panels": {"volume": True, "obv": True, "rsi": True, "macd": True},
    "ema": {"enabled": True, "periods": [12, 26, 50, 100, 200]},
    "supertrend": {"enabled": True, "period": 10, "multiplier": 3.0},
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "volume": {"sma": 20},
    "volume_profile": {"enabled": True, "bins": 48, "value_area": 0.70, "width_frac": 0.14},
    "key_levels": {"enabled": True, "show": {"day": ["open", "high", "low"], "week": ["open", "high", "low", "mid"],
                                             "month": ["open", "high", "low", "mid"], "monday": ["high", "low", "mid"],
                                             "year": ["open"]}},
    "ucts_proxy": {"enabled": True, "use_ema_filter": False, "ema_filter_period": 50},
}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Config:
    symbols_cfg: Dict[str, Any]
    params: Dict[str, Any]
    risk: Dict[str, Any]
    config_dir: Path = field(default_factory=lambda: Path("config"))
    # Indikator chart: TAMPILAN SAJA, sengaja tidak masuk config_hash (lihat utils.config_hash di main.py)
    chart_ind: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(CHART_IND_DEFAULTS))
    # True bila simbol di-override manual (ENGINE_SYMBOLS / --symbols / input workflow) -> scanner dimatikan
    symbols_overridden: bool = False

    @property
    def symbols(self) -> List[str]:
        return list(self.symbols_cfg["symbols"])

    @property
    def tfs(self) -> Dict[str, Any]:
        return self.symbols_cfg["timeframes"]

    @property
    def all_timeframes(self) -> List[str]:
        t = self.tfs
        out: List[str] = []
        for key in ("bias", "structure", "trigger"):
            for tf in t.get(key, []):
                if tf not in out:
                    out.append(tf)
        return out

    def tf_params(self, section: str, tf: str) -> Dict[str, Any]:
        """Gabungkan <section>.default dengan <section>.by_tf.<tf>."""
        sec = self.params.get(section, {})
        out = copy.deepcopy(sec.get("default", {}))
        out.update(copy.deepcopy(sec.get("by_tf", {}).get(tf, {})))
        return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"File konfigurasi tidak ditemukan: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML tidak valid di {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Isi {path} harus berupa mapping (key: value).")
    return data


def load_config(config_dir: str | Path = "config") -> Config:
    d = Path(config_dir)
    symbols_cfg = _load_yaml(d / "symbols.yaml")
    params = _load_yaml(d / "parameters.yaml")
    risk = _load_yaml(d / "risk.yaml")

    # override dari environment
    env_syms = os.getenv("ENGINE_SYMBOLS", "").strip()
    symbols_overridden = bool(env_syms)
    if env_syms:
        symbols_cfg["symbols"] = [s.strip() for s in env_syms.split(",") if s.strip()]
    env_prov = os.getenv("ENGINE_PROVIDERS", "").strip()
    if env_prov:
        symbols_cfg.setdefault("data", {})["providers"] = [s.strip() for s in env_prov.split(",") if s.strip()]

    for key in ("symbols", "timeframes", "candles"):
        if key not in symbols_cfg:
            raise ConfigError(f"symbols.yaml: key '{key}' wajib ada.")
    if not symbols_cfg["symbols"]:
        raise ConfigError("symbols.yaml: daftar symbols kosong.")
    for key in ("bias", "structure", "trigger", "primary_trigger"):
        if key not in symbols_cfg["timeframes"]:
            raise ConfigError(f"symbols.yaml: timeframes.{key} wajib ada.")
    for tf in set(sum([symbols_cfg["timeframes"][k] for k in ("bias", "structure", "trigger")], [])):
        if tf not in symbols_cfg["candles"]:
            raise ConfigError(f"symbols.yaml: candles.{tf} belum didefinisikan.")
    for key in ("bias", "range", "deviation", "breakout_retest", "fibonacci", "scoring"):
        if key not in params:
            raise ConfigError(f"parameters.yaml: section '{key}' wajib ada.")
    for key in ("account", "stop", "targets", "leverage"):
        if key not in risk:
            raise ConfigError(f"risk.yaml: section '{key}' wajib ada.")
    chart_ind = copy.deepcopy(CHART_IND_DEFAULTS)
    ci_path = d / "chart_indicators.yaml"
    if ci_path.exists():  # opsional: bila tidak ada, pakai default bawaan
        chart_ind = _deep_merge(chart_ind, _load_yaml(ci_path))
    return Config(symbols_cfg=symbols_cfg, params=params, risk=risk, config_dir=d, chart_ind=chart_ind,
                  symbols_overridden=symbols_overridden)
