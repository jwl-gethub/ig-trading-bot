import copy
import json

import numpy as np
import pandas as pd
import pytest

from bot.config import ConfigError, validate
from bot.live import LiveRunner
from bot.sleeves import Sleeves
from bot.strategy import PositionState

TUE = "2025-01-14"          # winter: the US cash close (16:00 New York) is 21:00 UTC


def dip_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["markets"] = [m for m in c["markets"] if m["name"] in ("EURUSD", "UK100")]
    nas = copy.deepcopy(next(m for m in c["markets"] if m["name"] == "UK100"))
    nas.update(name="NAS", epic="IX.D.NASDAQ.DAILY.IP", modules=["dip"],
               dip={"source": "USATECHIDXUSD", "price_range": [100, 100000], "schedule": "us_index",
                    "long_only": True})
    c["markets"].append(nas)
    c["sleeves"]["trend"]["weight"] = 0.10
    c["sleeves"]["meanrev"]["weight"] = 0.10
    c["sleeves"]["dip"] = {"modules": ["dip"], "weight": 0.40, "risk_per_trade_pct": 1.0, "max_leverage": 3,
                           "max_open_positions": 3, "max_positions_per_group": 2, "max_drawdown_pct": 20}
    c["dip"] = {"enabled": True}
    return validate(c)


def _hourly_from_daily(closes: pd.Series) -> pd.DataFrame:
    """Hourly bars whose close at 21:00 UTC each weekday equals the daily close."""
    pts = closes.copy()
    pts.index = pts.index.tz_localize("UTC") + pd.Timedelta(hours=21)
    hours = pd.date_range(pts.index[0] - pd.Timedelta(hours=21), pts.index[-1] + pd.Timedelta(hours=2),
                          freq="1h", tz="UTC")
    c = pts.reindex(hours.union(pts.index)).interpolate("time").bfill().ffill().reindex(hours)
    # the bar that ENDS at 21:00 starts at 20:00: shift so close(20:00 bar) = value at 21:00
    c = c.shift(-1).ffill()
    h = pd.DataFrame({"close": c.to_numpy()}, index=hours)
    h["open"] = h.close.shift().bfill()
    h["high"] = h[["open", "close"]].max(axis=1) * 1.0002
    h["low"] = h[["open", "close"]].min(axis=1) * 0.9998
    h["volume"] = 1.0
    return h[h.index.dayofweek < 5]


def _closes(rebound=False):
    days = pd.bdate_range("2023-10-02", "2025-01-15" if rebound else TUE)
    c = pd.Series(10_000 * np.exp(0.0007 * np.arange(len(days))), index=days)
    i = days.get_loc(pd.Timestamp(TUE))
    c.iloc[i - 2:i + 1] = c.iloc[i - 3] * np.array([0.985, 0.97, 0.955])     # three sharp down days
    if rebound:
        c.iloc[i + 1] = c.iloc[i - 3] * 1.0                                     # back above the 5-day average
    return c


class FakeSource:
    def __init__(self, hourly):
        self.h = hourly

    def hourly(self, inst, start, today, price_range):
        return self.h[self.h.index < pd.Timestamp(today, tz="UTC")], []


def _runner(cfg, make_broker, tmp_path, hourly, dry_run=False):
    cfg = copy.deepcopy(cfg)
    cfg["safety"]["kill_file"] = str(tmp_path / "KILL")
    cfg["safety"]["pause_file"] = str(tmp_path / "PAUSE")
    broker = make_broker(cfg, n_bars=600, equity=10_000, min_size=0.04)     # IG's real index minimum
    r = LiveRunner(cfg, broker, tmp_path / "state", tmp_path / "logs", dry_run=dry_run)
    r.sleeves.ensure_bases(10_000)
    r.dip.source = FakeSource(hourly)
    return r, broker


def _events(tmp_path, kind):
    lines = (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == kind]


def test_dip_enters_after_the_close_and_exits_on_the_rebound(cfg, make_broker, tmp_path):
    c = dip_cfg(cfg)
    r, broker = _runner(c, make_broker, tmp_path, _hourly_from_daily(_closes()))
    early = pd.Timestamp("2025-01-15 14:59", tz="UTC")              # 09:59 New York: not yet
    r.dip.tick(early)
    assert broker.orders == []
    now = pd.Timestamp("2025-01-15 15:00:30", tz="UTC")              # 10:00 New York, the morning after
    broker.now = now
    r.dip.tick(now)
    d = _events(tmp_path, "DIP_DECISION")[-1]
    assert d["action"] == "long" and d["rsi2"] < 5 and d["close"] > d["sma200"]
    assert len(broker.orders) == 1 and broker.orders[0]["epic"] == "IX.D.NASDAQ.DAILY.IP"
    pos = r.positions["dip:NAS"]
    assert pos.module == "dip" and pos.sleeve == "dip" and pos.direction == 1
    mid = r.dip_prices["NAS"]
    assert broker.orders[0]["stop_distance"] == pytest.approx(2.5 * d["atr_pct"] / 100 * mid, rel=0.02)
    r.dip.tick(now + pd.Timedelta(minutes=5))                        # same day: nothing more
    assert len(broker.orders) == 1
    # Wednesday closes back above the 5-day average -> exit Thursday morning
    r.dip.source = FakeSource(_hourly_from_daily(_closes(rebound=True)))
    thu = pd.Timestamp("2025-01-16 15:00:30", tz="UTC")
    broker.now = thu
    r.dip.tick(thu)
    assert _events(tmp_path, "DIP_DECISION")[-1]["action"] == "exit"
    assert len(broker.closes) == 1 and "dip:NAS" not in r.positions
    assert _events(tmp_path, "TRADE_JOURNAL")[-1]["module"] == "dip"


def test_waits_for_data_then_skips_after_the_grace_window(cfg, make_broker, tmp_path):
    c = dip_cfg(cfg)
    stale = _hourly_from_daily(_closes())
    stale = stale[stale.index < pd.Timestamp(TUE, tz="UTC")]           # Tuesday's close not published
    r, broker = _runner(c, make_broker, tmp_path, stale)
    now = pd.Timestamp("2025-01-15 15:00:30", tz="UTC")
    r.dip.tick(now)
    assert _events(tmp_path, "DIP_DATA_NOT_READY") and broker.orders == []
    r.dip.tick(now + pd.Timedelta(minutes=5))                         # retry is 15 min away
    assert len(_events(tmp_path, "DIP_DATA_NOT_READY")) == 1
    r.dip.tick(now + pd.Timedelta(hours=4))                           # past the 3h grace window
    assert _events(tmp_path, "DIP_MISSED") and broker.orders == []


def test_dry_run_and_pause_do_not_trade(cfg, make_broker, tmp_path):
    c = dip_cfg(cfg)
    r, broker = _runner(c, make_broker, tmp_path, _hourly_from_daily(_closes()), dry_run=True)
    now = pd.Timestamp("2025-01-15 15:00:30", tz="UTC")
    r.dip.tick(now)
    assert broker.orders == [] and _events(tmp_path, "DRY_RUN_ORDER")
    r2, broker2 = _runner(c, make_broker, tmp_path / "p", _hourly_from_daily(_closes()))
    (tmp_path / "p" / "PAUSE").parent.mkdir(parents=True, exist_ok=True)
    r2.cfg["safety"]["pause_file"] = str(tmp_path / "p" / "PAUSE")
    (tmp_path / "p" / "PAUSE").write_text("x")
    r2.dip.tick(now)
    assert broker2.orders == []


def test_four_hour_management_never_touches_dip_positions(cfg, make_broker, tmp_path):
    c = dip_cfg(cfg)
    r, broker = _runner(c, make_broker, tmp_path, _hourly_from_daily(_closes()))
    first = next(iter(broker._bars.values()))
    now = first.index[450] + pd.Timedelta(hours=4, minutes=1)
    broker.now = now
    r.startup(now)
    r.positions["dip:UK100"] = PositionState(market="UK100", direction=1, module="dip", entry_price=100.0,
                                             stop_level=1.0, size=0.1, point_value=1.0, initial_risk_points=99.0,
                                             entry_ts=str(now), extreme=100.0, deal_id="X1", sleeve="dip")
    seen = []
    r._manage = lambda key, row, now: seen.append(key)
    later = now + pd.Timedelta(hours=4)
    broker.now = later
    r.tick(later)
    assert "dip:UK100" not in seen


def test_sleeves_are_recut_from_the_same_pot():
    cfg = {"risk": {"loss_streak_halve_risk": 4}, "sleeves": {
        "leveraged": {"modules": ["sweep"], "weight": 0.1}, "trend": {"modules": ["trend"], "weight": 0.3},
        "meanrev": {"modules": ["meanrev"], "weight": 0.2}, "dip": {"modules": ["dip"], "weight": 0.4}}}
    s = Sleeves(cfg, {"leveraged": {"base": 1000, "peak": 1000}, "trend": {"base": 4500, "peak": 4600},
                      "meanrev": {"base": 4500, "peak": 4500}})
    changes = s.ensure_bases(account_equity=10_200)
    bases = {n: st.base for n, st in s.states.items()}
    assert bases == {"leveraged": 1000, "trend": 3000, "meanrev": 2000, "dip": 4000}
    assert s.states["trend"].peak == 3100                              # peak keeps its gain over base
    assert {c["sleeve"] for c in changes} == {"trend", "meanrev"}
    assert s.ensure_bases(10_500) == []                                # stable once re-cut


def test_config_rejects_bad_dip_markets(cfg):
    c = dip_cfg(cfg)
    bad = copy.deepcopy(c)
    del next(m for m in bad["markets"] if m["name"] == "NAS")["dip"]["source"]
    with pytest.raises(ConfigError):
        validate(bad)
    bad = copy.deepcopy(c)
    next(m for m in bad["markets"] if m["name"] == "NAS")["dip"]["schedule"] = "moon"
    with pytest.raises(ConfigError):
        validate(bad)


def test_prefetch_warms_every_market_and_logs(cfg, make_broker, tmp_path):
    c = dip_cfg(cfg)
    r, _ = _runner(c, make_broker, tmp_path, _hourly_from_daily(_closes()))
    r.dip.prefetch(pd.Timestamp("2025-01-15 06:00", tz="UTC"))
    ev = _events(tmp_path, "DIP_PREFETCH")
    assert [e["market"] for e in ev] == ["NAS"] and ev[0]["hourly_bars"] > 1000
