import copy
import json

import numpy as np
import pandas as pd

from bot.config import validate
from bot.live import LiveRunner
from tests.conftest import FakeBroker

EPIC = "CS.D.EURUSD.TODAY.IP"
T0 = pd.Timestamp("2025-01-08 10:00", tz="UTC")           # a Wednesday, inside the 07-17 session


class MinuteBroker(FakeBroker):
    """FakeBroker fed with 1-minute bars; IG's 15-minute history is the same data resampled."""
    def bars(self, epic, resolution, n):
        m = self._visible(epic)
        b = m.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        return b.tail(n)


def _minutes(after):
    """260 rising 15-minute bars (+1 each) up to T0, then `after`: per-bar close changes."""
    start = T0 - 260 * pd.Timedelta(minutes=15)
    closes = list(5000 + np.arange(1, 261))
    for chg in after:
        closes.append(closes[-1] + chg)
    idx = pd.date_range(start, periods=len(closes) * 15, freq="1min", tz="UTC")
    prev = np.r_[5000.0, closes[:-1]]
    px = np.concatenate([np.linspace(p, c, 16)[1:] for p, c in zip(prev, closes)])
    wig = np.tile([0.0, 2.0, -2.0, 0.0, 0.0], len(px) // 5)           # gives each bar a +-2 range
    px = px + wig
    return pd.DataFrame({"open": px, "high": px + 0.1, "low": px - 0.1, "close": px}, index=idx)


def _setup(cfg, tmp_path, after, dry_run=False):
    c = copy.deepcopy(cfg)
    c["markets"] = [m for m in c["markets"] if m["name"] == "EURUSD"]
    c["markets"][0]["modules"] = ["scalp"]
    c["scalp"]["enabled"] = True
    c["safety"]["kill_file"], c["safety"]["pause_file"] = str(tmp_path / "KILL"), str(tmp_path / "PAUSE")
    c = validate(c)
    broker = MinuteBroker({EPIC: _minutes(after)}, equity=10_000, spread=0.2, min_size=0.04)
    broker.now = T0 - pd.Timedelta(minutes=1)
    r = LiveRunner(c, broker, tmp_path / "state", tmp_path / "logs", dry_run=dry_run)
    r.sleeves.ensure_bases(10_000)
    r.scalp.startup(T0)
    return r, broker


def _run(r, broker, start, minutes):
    for k in range(minutes):
        now = start + pd.Timedelta(minutes=k, seconds=5)
        broker.now = now
        r._info_cache = {}
        r.scalp.tick(now)


def _events(tmp_path, kind):
    return [e for e in map(json.loads, (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines())
            if e["event"] == kind]


def test_seeds_history_enters_on_pullback_with_stop_and_target_then_time_stop(cfg, tmp_path):
    flat = [0.3] * 12
    r, broker = _setup(cfg, tmp_path, [-5, -5] + flat)
    ready = _events(tmp_path, "SCALP_READY")[0]
    assert ready["warm"] and ready["bars"] >= 259
    _run(r, broker, T0, 29)                                  # two falling bars still forming
    assert broker.orders == []
    _run(r, broker, T0 + pd.Timedelta(minutes=29), 3)        # 10:30: second bar closed -> RSI(2) < 10 in an uptrend
    assert len(broker.orders) == 1
    o = broker.orders[0]
    assert o["direction"] == "BUY" and o["stop_distance"] > 0
    assert abs(o["limit_distance"] / o["stop_distance"] - 1.5) < 0.05
    pos = r.positions["scalp:EURUSD"]
    assert pos.module == "scalp" and pos.sleeve == "scalp"
    assert pos.size * o["stop_distance"] <= 0.01 * 3000 + 0.5            # risk <= 1% of the £3,000 pot
    _run(r, broker, T0 + pd.Timedelta(minutes=32), 125)      # drifts sideways: neither stop nor target
    assert "scalp:EURUSD" not in r.positions and len(broker.closes) == 1
    j = _events(tmp_path, "TRADE_JOURNAL")[-1]
    assert j["module"] == "scalp" and j["exit_reason"] == "scalp_time"


def test_take_profit_on_ig_side_is_journalled_as_target(cfg, tmp_path):
    r, broker = _setup(cfg, tmp_path, [-5, -5, 6, 6, 6])
    _run(r, broker, T0, 80)
    assert len(broker.orders) == 1 and "scalp:EURUSD" not in r.positions
    assert _events(tmp_path, "TRADE_JOURNAL")[-1]["exit_reason"] == "target_server_side"


def test_no_trade_outside_session_or_in_dry_run(cfg, tmp_path):
    r, broker = _setup(cfg, tmp_path, [-5, -5, 0.3], dry_run=True)
    _run(r, broker, T0, 35)
    assert broker.orders == [] and _events(tmp_path, "DRY_RUN_ORDER")
    r2, broker2 = _setup(cfg, tmp_path / "b", [-5, -5, 0.3])
    r2.scalp.markets["EURUSD"]["scalp"]["hours_utc"] = [12, 17]          # 10:30 is outside
    _run(r2, broker2, T0, 35)
    assert broker2.orders == []


def test_low_allowance_means_no_seed_and_no_trading_until_warm(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    FakeBroker.allowance_remaining, keep = 500, FakeBroker.allowance_remaining
    try:
        r, broker = _setup(c, tmp_path, [-5, -5, 0.3])
        assert not _events(tmp_path, "SCALP_READY")[0]["warm"]
        _run(r, broker, T0, 35)
        assert broker.orders == []
    finally:
        FakeBroker.allowance_remaining = keep
