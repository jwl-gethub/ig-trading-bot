"""Config loading + validation.

HARD_LIMITS are code-level ceilings. Even if config.yaml is edited, the bot
refuses to start with riskier values. CODEX: never raise these.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

HARD_LIMITS = {
    "risk_per_trade_pct": 2.0,
    "max_trade_risk_pct": 2.0,      # any single trade, % of the WHOLE account (sleeves included)
    "max_total_leverage": 10.0,     # total exposure across all open trades, x account equity
    "max_total_open_risk_pct": 6.0,
    "max_open_positions": 5,
    "max_margin_utilisation_pct": 70.0,
    "daily_loss_limit_pct": 5.0,
    "weekly_loss_limit_pct": 10.0,
    "max_drawdown_halt_pct": 25.0,
}

REQUIRED_MARKET_KEYS = {
    "name", "epic", "group", "modules", "allow_short", "point_value", "price_scale",
    "min_size", "size_step", "margin_pct", "max_spread_points", "backtest_spread_points",
    "funding_long_pct", "funding_short_pct",
}


class ConfigError(ValueError):
    pass


def validate(cfg: dict) -> dict:
    for section in ("markets", "strategy", "risk", "safety", "backtest", "promotion"):
        if section not in cfg:
            raise ConfigError(f"missing section: {section}")

    risk = cfg["risk"]
    for key, ceiling in HARD_LIMITS.items():
        if key not in risk:
            raise ConfigError(f"risk.{key} missing")
        if float(risk[key]) <= 0:
            raise ConfigError(f"risk.{key} must be > 0")
        if float(risk[key]) > ceiling:
            raise ConfigError(f"risk.{key}={risk[key]} exceeds hard limit {ceiling}")
    if risk["risk_per_trade_pct"] > risk["max_total_open_risk_pct"]:
        raise ConfigError("risk_per_trade_pct cannot exceed max_total_open_risk_pct")
    if risk["max_trade_risk_pct"] > risk["max_total_open_risk_pct"]:
        raise ConfigError("max_trade_risk_pct cannot exceed max_total_open_risk_pct")

    names = set()
    for m in cfg["markets"]:
        missing = REQUIRED_MARKET_KEYS - set(m)
        if missing:
            raise ConfigError(f"market {m.get('name')} missing keys: {sorted(missing)}")
        if m["name"] in names:
            raise ConfigError(f"duplicate market {m['name']}")
        names.add(m["name"])
        bad = set(m["modules"]) - {"trend", "meanrev", "sweep"}
        if bad:
            raise ConfigError(f"market {m['name']} unknown modules {bad}")
        if m["min_size"] <= 0 or m["size_step"] <= 0 or m["point_value"] <= 0:
            raise ConfigError(f"market {m['name']} sizes/point_value must be > 0")
    from .sleeves import validate_sleeves
    validate_sleeves(cfg)
    return cfg


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return validate(cfg)


def market_map(cfg: dict) -> dict[str, dict]:
    return {m["name"]: m for m in cfg["markets"]}


def with_overrides(cfg: dict, **risk_overrides) -> dict:
    """Copy of cfg with risk overrides, re-validated (used by tests/tools)."""
    new = copy.deepcopy(cfg)
    new["risk"].update(risk_overrides)
    return validate(new)
