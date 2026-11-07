import pandas as pd
import pytest

from bot.config import ConfigError, validate, with_overrides
from bot.risk import RiskManager
from bot.strategy import PositionState

TS = pd.Timestamp("2026-03-03 12:00", tz="UTC")  # a Tuesday


def rm(cfg):
    r = RiskManager(cfg["risk"], cfg["safety"])
    r.on_bar(TS, 10_000)
    return r


def pos(market="EURUSD", group="fx_usd", risk_pts=50, size=1.0):
    return PositionState(market=market, direction=1, module="trend", entry_price=1000, stop_level=1000 - risk_pts,
                         size=size, point_value=1.0, initial_risk_points=risk_pts, entry_ts=str(TS),
                         group=group, margin=100)


def test_sizing_math(cfg):
    r = rm(cfg)
    res = r.size(10_000, stop_points=50, point_value=1.0, min_size=0.5, size_step=0.01)
    assert res.ok and res.size == 2.0 and res.risk_amount == pytest.approx(100)


def test_sizing_rounds_down_never_up(cfg):
    r = rm(cfg)
    res = r.size(10_000, stop_points=33, point_value=1.0, min_size=0.5, size_step=0.01)
    assert res.ok and res.size == 3.03 and res.risk_amount <= 100


def test_unsizeable_when_min_stake_too_big(cfg):
    r = rm(cfg)
    res = r.size(1_000, stop_points=60, point_value=1.0, min_size=0.5, size_step=0.01)
    assert not res.ok and res.reason.startswith("UNSIZEABLE")


def test_no_stop_no_size(cfg):
    r = rm(cfg)
    assert not r.size(10_000, 0, 1.0, 0.5, 0.01).ok


def test_group_and_market_limits(cfg):
    r = rm(cfg)
    mkt = {"name": "GBPUSD", "group": "fx_usd"}
    ok, why = r.check_entry(TS, mkt, [pos()], 10_000, 100, 100)
    assert not ok and why.startswith("GROUP_LIMIT")
    ok, why = r.check_entry(TS, {"name": "EURUSD", "group": "x"}, [pos()], 10_000, 100, 100)
    assert not ok and why == "ALREADY_IN_MARKET"


def test_total_open_risk_cap(cfg):
    r = rm(cfg)
    existing = [pos("A", "g1", 100), pos("B", "g2", 100)]          # £200 open risk
    ok, why = r.check_entry(TS, {"name": "C", "group": "g3"}, existing, 10_000, 150, 10)
    assert not ok and why == "TOTAL_OPEN_RISK_CAP"                  # 350 > 3% of 10k


def test_max_positions(cfg):
    r = rm(cfg)
    existing = [pos(f"M{i}", f"g{i}", 1, 0.5) for i in range(3)]
    ok, why = r.check_entry(TS, {"name": "Z", "group": "gz"}, existing, 10_000, 10, 10)
    assert not ok and why == "MAX_OPEN_POSITIONS"


def test_daily_loss_limit_blocks_entries(cfg):
    r = rm(cfg)
    r.on_bar(TS + pd.Timedelta(hours=4), 9_650)
    ok, why = r.check_entry(TS + pd.Timedelta(hours=4), {"name": "X", "group": "g"}, [], 9_650, 10, 10)
    assert not ok and why == "DAILY_LOSS_LIMIT"
    nxt = TS + pd.Timedelta(days=1)
    r.on_bar(nxt, 9_650)
    assert r.check_entry(nxt, {"name": "X", "group": "g"}, [], 9_650, 10, 10)[0]


def test_drawdown_halt_is_sticky(cfg):
    r = rm(cfg)
    r.on_bar(TS + pd.Timedelta(days=7), 8_400)
    assert r.state.halted
    r.on_bar(TS + pd.Timedelta(days=8), 12_000)       # recovery does NOT un-halt
    assert r.state.halted
    assert not r.check_entry(TS + pd.Timedelta(days=8), {"name": "X", "group": "g"}, [], 12_000, 1, 1)[0]
    r.reset_halt()
    assert not r.state.halted


def test_loss_streak_halves_risk_then_pauses(cfg):
    r = rm(cfg)
    for _ in range(4):
        r.on_trade_closed(-10, TS)
    assert r.current_risk_pct() == cfg["risk"]["risk_per_trade_pct"] / 2
    for _ in range(2):
        r.on_trade_closed(-10, TS)
    ok, why = r.check_entry(TS + pd.Timedelta(hours=4), {"name": "X", "group": "g"}, [], 10_000, 1, 1)
    assert not ok and why.startswith("PAUSED")
    r.on_trade_closed(+50, TS)
    assert r.current_risk_pct() == cfg["risk"]["risk_per_trade_pct"]


def test_friday_cutoff_and_weekend(cfg):
    r = rm(cfg)
    fri = pd.Timestamp("2026-03-06 17:00", tz="UTC")
    assert r.check_entry(fri, {"name": "X", "group": "g"}, [], 10_000, 1, 1)[1] == "FRIDAY_CUTOFF"
    sat = pd.Timestamp("2026-03-07 10:00", tz="UTC")
    assert r.check_entry(sat, {"name": "X", "group": "g"}, [], 10_000, 1, 1)[1] == "WEEKEND"


def test_hard_limits_reject_risky_config(cfg):
    with pytest.raises(ConfigError):
        with_overrides(cfg, risk_per_trade_pct=5.0)
    with pytest.raises(ConfigError):
        with_overrides(cfg, max_drawdown_halt_pct=40)
    bad = dict(cfg)
    bad["markets"] = [dict(cfg["markets"][0], modules=["martingale"])]
    with pytest.raises(ConfigError):
        validate(bad)
