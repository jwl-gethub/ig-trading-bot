import copy

import pandas as pd
import pytest

from bot.config import ConfigError, validate
from bot.risk import RiskManager
from bot.sleeves import Sleeves
from bot.strategy import PositionState

TS = pd.Timestamp("2026-03-03 12:00", tz="UTC")


def pos(sleeve, market="EURUSD", group="fx_usd", notional=0.0, entry=1000.0, size=1.0):
    return PositionState(market=market, direction=1, module="x", entry_price=entry, stop_level=entry - 50,
                         size=size, point_value=1.0, initial_risk_points=50, entry_ts=str(TS),
                         group=group, sleeve=sleeve, notional=notional)


def test_config_sleeves_are_10_45_45(cfg):
    s = Sleeves(cfg)
    s.ensure_bases(10_000)
    assert {n: st.base for n, st in s.states.items()} == {"leveraged": 1000, "trend": 4500, "meanrev": 4500}
    assert s.by_module == {"sweep": "leveraged", "trend": "trend", "meanrev": "meanrev"}


@pytest.mark.parametrize("mutate", [
    lambda c: c["sleeves"]["trend"].update(weight=0.9),                 # weights > 1
    lambda c: c["sleeves"]["trend"]["modules"].append("sweep"),         # module in two sleeves
    lambda c: c["sleeves"]["leveraged"].update(risk_per_trade_pct=25),  # above sleeve hard limit
    lambda c: c["sleeves"]["trend"].update(risk_per_trade_pct=5),       # 0.45 x 5% = 2.25% of account > 2%
    lambda c: c["sleeves"]["leveraged"].update(max_leverage=50),        # above IG max
])
def test_sleeve_validation_rejects_risky_configs(cfg, mutate):
    bad = copy.deepcopy(cfg)
    mutate(bad)
    with pytest.raises(ConfigError):
        validate(bad)


def test_leveraged_sleeve_sizes_from_its_own_equity_and_caps_exposure(cfg):
    s = Sleeves(cfg)
    s.ensure_bases(10_000)
    rm = RiskManager(cfg["risk"], cfg["safety"])
    eq = s.equity("leveraged", [], {})
    assert eq == 1_000
    # EURUSD-like: stop 55pts, price 11,141. 10% of £1,000 = £100 -> 1.81/pt, notional ~£20.2k > 20x cap
    room = s.max_size_by_leverage("leveraged", eq, [], 11_141, 1.0)
    sz = rm.size(eq, 55.0, 1.0, 0.04, 0.01, risk_pct=s.risk_pct("leveraged"), max_size=room)
    assert sz.ok and sz.size * 11_141 <= 20 * 1_000 + 1e-6 and sz.risk_amount <= 100
    # cap already used up -> leverage rejection, never a bigger trade
    full = [pos("leveraged", "GBPUSD", notional=20_000)]
    room = s.max_size_by_leverage("leveraged", eq, full, 11_141, 1.0)
    assert not rm.size(eq, 55.0, 1.0, 0.04, 0.01, risk_pct=10, max_size=room).ok


def test_sleeve_drawdown_halts_only_that_sleeve(cfg):
    s = Sleeves(cfg)
    s.ensure_bases(10_000)
    s.update("leveraged", 1_000, TS)
    assert s.update("leveraged", 490, TS) is not None          # -51% > 50% limit
    assert not s.check("leveraged", TS, "EURUSD", "fx_usd", [])[0]
    assert s.check("trend", TS, "EURUSD", "fx_usd", [])[0]     # other sleeves unaffected
    s.reset("leveraged")
    assert s.check("leveraged", TS, "EURUSD", "fx_usd", [])[0]


def test_two_sleeves_can_hold_the_same_market(cfg):
    s = Sleeves(cfg)
    rm = RiskManager(cfg["risk"], cfg["safety"])
    rm.on_bar(TS, 10_000)
    held = [pos("trend", "EURUSD", notional=5_000)]
    assert s.check("leveraged", TS, "EURUSD", "fx_usd", held)[0]
    assert rm.check_entry(TS, {"name": "EURUSD", "group": "fx_usd"}, held, 10_000, 50, 10, 1_000,
                          sleeve="leveraged")[0]
    assert s.check("trend", TS, "EURUSD", "fx_usd", held)[1] == "ALREADY_IN_MARKET"


def test_account_caps_still_bind_across_sleeves(cfg):
    rm = RiskManager(cfg["risk"], cfg["safety"])
    rm.on_bar(TS, 10_000)
    held = [pos("trend", "EURUSD", notional=30_000), pos("meanrev", "UK100", "equity_index", notional=15_000)]
    ok, why = rm.check_entry(TS, {"name": "GOLD", "group": "metals"}, held, 10_000, 50, 10, 10_000,
                             sleeve="leveraged")
    assert not ok and why == "TOTAL_LEVERAGE_CAP"              # 55k > 5 x 10k
    ok, why = rm.check_entry(TS, {"name": "GOLD", "group": "metals"}, [], 10_000, 250, 10, 0, sleeve="x")
    assert not ok and why == "TRADE_RISK_CEILING"              # £250 > 2% of account


def test_sleeve_loss_streak_is_per_sleeve(cfg):
    s = Sleeves(cfg)
    for _ in range(4):
        s.on_trade_closed("leveraged", -100, TS)
    assert s.risk_pct("leveraged") == 5.0 and s.risk_pct("trend") == 1.0
