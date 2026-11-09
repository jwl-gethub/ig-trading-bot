import numpy as np
import pandas as pd

from research.dip import DipParams, backtest_dip, signals, daily_bars

M = 390


def _sm_from_daily(closes, intraday_vol=0.0, seed=0):
    """Session matrix whose minutes drift linearly from the previous close to today's close."""
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, float)
    prev = np.r_[closes[0], closes[:-1]]
    path = prev[:, None] + (closes - prev)[:, None] * np.linspace(0, 1, M)[None, :]
    path = path * np.exp(rng.normal(0, intraday_vol, path.shape)) if intraday_vol else path
    path[:, -1] = closes
    o = np.c_[prev, path[:, :-1]]
    return {"open": o, "high": np.maximum(o, path), "low": np.minimum(o, path), "close": path,
            "volume": np.ones_like(path), "days": pd.bdate_range("2020-01-01", periods=len(closes))}


def _uptrend_with_dip(n=260, dip_at=240):
    c = 3000 * np.exp(np.linspace(0, 0.25, n))
    c[dip_at:dip_at + 3] *= np.array([0.985, 0.97, 0.955])        # 3 sharp down closes
    c[dip_at + 3:] *= 0.96                                         # then recovers with the trend
    c[dip_at + 4:] *= 1.03
    return c


def test_buys_the_dip_in_uptrend_and_exits_on_rebound():
    sm = _sm_from_daily(_uptrend_with_dip())
    trades, daily = backtest_dip(sm, DipParams(cost_side_points=0.0, funding_annual=0.0, stop_atr=None))
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t.reason == "exit" and t.ret > 0 and str(t.day.date()) == "2020-12-09"
    assert (daily.lev > 0).mean() < 0.2                             # mostly out of the market


def test_near_close_signal_ignores_the_final_ten_minutes():
    c = _uptrend_with_dip()
    sm = _sm_from_daily(c)
    p = DipParams(fill="near_close")
    a = signals(daily_bars(sm, p.pre_close_minute), p)
    sm2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sm.items()}
    sm2["close"][242, p.pre_close_minute + 1:] *= 1.05             # change after 15:50 only
    b = signals(daily_bars(sm2, p.pre_close_minute), p)
    assert np.isclose(a.rsi.iloc[242], b.rsi.iloc[242]) and np.isclose(a.sma_exit.iloc[242], b.sma_exit.iloc[242])


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    c = _uptrend_with_dip()
    c[244] = 3520.0                                                 # the day after the 12-04 entry
    sm = _sm_from_daily(c)
    sm["open"][244, 0] = 3500.0                                     # gaps below the ~3610 stop
    sm["low"][244] = np.minimum(sm["low"][244], 3500.0)
    trades, _ = backtest_dip(sm, DipParams(cost_side_points=0.0, funding_annual=0.0, stop_atr=6))
    t = trades[trades.reason == "stop"].iloc[0]
    assert np.isclose(t.exit, 3500.0)


def test_funding_costs_money_while_holding():
    sm = _sm_from_daily(_uptrend_with_dip())
    free = backtest_dip(sm, DipParams(cost_side_points=0.0, funding_annual=0.0))[1].ret.sum()
    paid = backtest_dip(sm, DipParams(cost_side_points=0.0, funding_annual=0.10))[1].ret.sum()
    assert paid < free


def test_no_edge_on_random_walk():
    rng = np.random.default_rng(11)
    rs = []
    for seed in range(6):
        c = 3000 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 1500)))
        tr, _ = backtest_dip(_sm_from_daily(c), DipParams(cost_side_points=0.0, funding_annual=0.0, stop_atr=None))
        rs += list(tr.ret)
    rs = np.array(rs)
    t = rs.mean() / (rs.std() / np.sqrt(len(rs)))
    assert len(rs) > 60 and abs(t) < 3, (len(rs), t)
