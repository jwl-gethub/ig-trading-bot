import numpy as np
import pandas as pd

from bot.scalp import ScalpRules, decide, in_session, indicators, to_bars
from research.scalp_test import backtest


def _bars(c, start="2025-01-06 08:00"):
    c = np.asarray(c, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(start, periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0003, "low": np.minimum(o, c) * 0.9997, "close": c}, index=idx)


def _row(**k):
    base = dict(close=110, ema_fast=108, ema_slow=105, ema_trend=100, rsi=5, atr=1.0)
    return pd.Series({**base, **k})


def test_decide():
    r = ScalpRules()
    assert decide(_row(), r) == "long"
    assert decide(_row(rsi=20), r) is None
    assert decide(_row(ema_fast=104), r) is None                     # fast below slow: no long
    assert decide(_row(close=90, ema_fast=92, ema_slow=95, rsi=95), r) == "short"
    assert decide(_row(close=90, ema_fast=92, ema_slow=95, rsi=95), ScalpRules(long_only=True)) is None
    assert decide(_row(ema_trend=np.nan), r) is None


def test_indicators_no_lookahead_and_bars_from_samples():
    rng = np.random.default_rng(2)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 600)))
    a = indicators(_bars(c), ScalpRules())
    c2 = c.copy(); c2[400:] *= 1.1
    b = indicators(_bars(c2), ScalpRules())
    cols = ["ema_fast", "ema_slow", "ema_trend", "rsi", "atr"]
    assert np.allclose(a[cols].iloc[:400].to_numpy(), b[cols].iloc[:400].to_numpy(), equal_nan=True)
    s = pd.Series([1.0, 3.0, 2.0, 5.0], index=pd.to_datetime(
        ["2025-01-06 10:00:10", "2025-01-06 10:07:00", "2025-01-06 10:14:50", "2025-01-06 10:15:05"], utc=True))
    bars = to_bars(s)
    assert list(bars.iloc[0]) == [1.0, 3.0, 1.0, 2.0] and len(bars) == 2


def test_session():
    assert in_session(pd.Timestamp("2025-01-06 14:15", tz="UTC"), (14, 21))
    assert not in_session(pd.Timestamp("2025-01-06 21:00", tz="UTC"), (14, 21))
    assert not in_session(pd.Timestamp("2025-01-04 15:00", tz="UTC"), (14, 21))   # Saturday


def test_backtest_no_edge_on_random_walk_and_costs_hurt():
    rng = np.random.default_rng(7)
    c = 5000 * np.exp(np.cumsum(rng.normal(0, 0.0008, 60_000)))
    bars = _bars(c, "2023-01-02 00:00")
    free = backtest(bars, ScalpRules(), (0, 24), 0.0)
    paid = backtest(bars, ScalpRules(), (0, 24), 0.4)
    t = free.r.mean() / (free.r.std() / np.sqrt(len(free)))
    assert len(free) > 300 and abs(t) < 3, (len(free), t)
    assert paid.r.mean() < free.r.mean()
    assert set(free.why) <= {"stop", "target", "time"} and free.r.min() > -1.6
