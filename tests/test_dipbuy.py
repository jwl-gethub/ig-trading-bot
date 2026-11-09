import numpy as np
import pandas as pd

from bot.dipbuy import DipRules, DipSchedule, daily_bars, decide, fill_time, indicators

FX_DAY = DipSchedule("America/New_York", 17, "Europe/London", 8)
NY_INDEX = DipSchedule("America/New_York", 16, "America/New_York", 10)


def _hourly(start, end, f=lambda i: 100.0 + i):
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    c = np.array([f(i) for i in range(len(idx))], dtype=float)
    return pd.DataFrame({"open": c - 0.5, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}, index=idx)


def test_fx_daily_close_is_5pm_new_york_and_weekends_drop():
    h = _hourly("2025-01-05 22:00", "2025-01-11 00:00")              # Sun evening .. Fri night (winter)
    d = daily_bars(h, FX_DAY)
    assert [str(t) for t in d.index[:2]] == ["2025-01-06 22:00:00+00:00", "2025-01-07 22:00:00+00:00"]
    bar = h.loc["2025-01-06 21:00"]                                   # the bar ending 17:00 New York
    assert d.close.iloc[0] == bar.close
    assert (d.index.dayofweek < 5).all()


def test_us_index_close_moves_with_daylight_saving():
    winter = daily_bars(_hourly("2025-01-06", "2025-01-08"), NY_INDEX)
    summer = daily_bars(_hourly("2025-07-07", "2025-07-09"), NY_INDEX)
    assert winter.index[0].hour == 21 and summer.index[0].hour == 20  # 16:00 New York in UTC


def test_fill_time_is_next_weekday_morning():
    fri_close = pd.Timestamp("2025-01-10 22:00", tz="UTC")
    assert fill_time(fri_close, FX_DAY) == pd.Timestamp("2025-01-13 08:00", tz="UTC")       # Monday London 8am
    mon_close = pd.Timestamp("2025-07-07 20:00", tz="UTC")
    assert fill_time(mon_close, NY_INDEX) == pd.Timestamp("2025-07-08 14:00", tz="UTC")     # Tue 10:00 New York


def _row(close, trend, exit_, rsi, atr=1.0):
    return pd.Series({"close": close, "sma_trend": trend, "sma_exit": exit_, "rsi": rsi, "atr": atr})


def test_decide_rules():
    r = DipRules()
    assert decide(_row(110, 100, 112, 3), 0, r) == "long"
    assert decide(_row(110, 100, 112, 8), 0, r) is None              # not oversold enough
    assert decide(_row(90, 100, 88, 97), 0, r) is None               # long-only: no shorts
    assert decide(_row(90, 100, 88, 97), 0, DipRules(long_only=False)) == "short"
    assert decide(_row(113, 100, 112, 60), 1, r) == "exit"
    assert decide(_row(111, 100, 112, 60), 1, r) is None
    assert decide(_row(87, 100, 88, 30), -1, r) == "exit"
    assert decide(_row(110, np.nan, 112, 3), 0, r) is None           # not enough history


def test_indicators_do_not_look_ahead():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    d = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c},
                     index=pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC"))
    a = indicators(d, DipRules())
    d2 = d.copy()
    d2.iloc[300:] *= 1.3
    b = indicators(d2, DipRules())
    cols = ["sma_trend", "sma_exit", "rsi", "atr"]
    assert np.allclose(a[cols].iloc[:300].to_numpy(), b[cols].iloc[:300].to_numpy(), equal_nan=True)


def test_screen_backtest_has_no_edge_on_random_walk():
    from research.dip_markets import backtest
    rng = np.random.default_rng(4)
    rs = []
    for k in range(4):
        n = 24 * 5 * 52 * 4
        idx = pd.bdate_range("2019-01-01", periods=n // 24, tz="UTC")
        hours = pd.DatetimeIndex([t + pd.Timedelta(hours=h) for t in idx for h in range(24)])
        c = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.0015, len(hours))))
        h = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.0005, "low": c * 0.9995, "close": c,
                          "volume": 1.0}, index=hours)
        trades, _ = backtest(h, FX_DAY, DipRules(long_only=False), 0.0, 0.0, 0.0)
        rs += list(trades.ret)
    rs = np.array(rs)
    t = rs.mean() / (rs.std() / np.sqrt(len(rs)))
    assert len(rs) > 40 and abs(t) < 3, (len(rs), t)
