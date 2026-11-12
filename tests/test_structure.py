import numpy as np
import pandas as pd

from bot.structure import StructureRules, htf_bias, signals, swings
from research.structure_test import backtest_market, simulate

R3 = StructureRules()


def _df(rows, start="2025-01-06 08:00"):
    a = np.asarray(rows, float)
    idx = pd.date_range(start, periods=len(a), freq="15min", tz="UTC")
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]}, index=idx)


def _path(points, steps=6, pad=0.2):
    """Piecewise-linear close path through waypoints; one bar per step with a small wick."""
    cl = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        cl += list(np.linspace(a, b, steps + 1)[1:])
    rows, prev = [], cl[0]
    for c in cl:
        rows.append((prev, max(prev, c) + pad * (c > prev), min(prev, c) - pad * (c < prev), c))
        prev = c
    return rows


UP = [100, 108, 102, 115, 106, 125]            # H1 108, L1 102, H2 115, L2 106 -> bullish, then BOS of 115
PULL = [(125.0, 125.2, 123.0, 123.2), (123.2, 123.4, 120.0, 120.2), (120.2, 120.4, 117.0, 117.2),
        (117.2, 117.4, 115.0, 115.2), (115.2, 115.4, 114.2, 114.4), (114.4, 114.6, 114.0, 114.2)]
CONF = (114.1, 115.6, 113.9, 115.4)


def _setup(extra=()):
    return _df(_path(UP, 8) + PULL + [CONF] + list(extra) + [(115.4, 116, 115, 115.8)] * 5)


def test_swings_confirmed_k_late_and_no_repaint():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    d = _df(np.c_[c, c + 0.5, c - 0.5, c])
    full = swings(d, 3)
    assert (full["confirm"] - full["pivot"] == 3).all()
    for t in (50, 120, 299):
        part = swings(d.iloc[:t + 1], 3)
        want = full[full.confirm <= t].reset_index(drop=True)
        assert part.equals(want)
    junk = d.copy(); junk.iloc[200:] *= 2
    assert swings(junk.iloc[:200], 3).equals(full[full.confirm <= 199].reset_index(drop=True))


def test_swing_strictness():
    d = _df([(1, 2, 1, 1.5)] * 3 + [(1, 5, 1, 2)] + [(1, 2, 1, 1.5)] * 3 + [(1, 5, 1, 2)] + [(1, 2, 1, 1.5)] * 3)
    s = swings(d, 3)
    assert list(s[s.kind == "H"]["pivot"]) == [3, 7]


def test_bullish_setup_one_entry_stop_and_target():
    d = _setup()
    e = signals(d, R3)
    assert len(e) == 1 and e[0].side == 1
    x = e[0]
    pull_low = d.low.iloc[x.bos_idx + 1: x.idx + 1].min()
    assert x.stop_level < pull_low and abs(x.stop_level - pull_low) <= 0.1 * x.atr + 1e-9
    assert abs(x.target_dist - 2 * x.stop_dist) < 1e-9
    assert abs(x.broken_level - 115.2) < 1e-9 and abs(x.leg_low - 105.8) < 1e-9 and not x.choch
    # a bar later in the series cannot produce a second entry from the same setup
    d2 = _df(_path(UP, 8) + PULL + [CONF, (115.4, 117, 115.3, 116.9), (116.9, 118, 116.5, 117.9)])
    assert len(signals(d2, R3)) == 1


def test_bearish_mirror():
    m = _setup()
    mir = pd.DataFrame({"open": 200 - m.open, "high": 200 - m.low, "low": 200 - m.high, "close": 200 - m.close}, index=m.index)
    e = signals(mir, R3)
    assert len(e) == 1 and e[0].side == -1 and e[0].stop_level > mir.high.iloc[e[0].bos_idx + 1:e[0].idx + 1].max()
    assert signals(mir, StructureRules(long_only=True)) == []


def test_bos_needs_close_through_level():
    d = _df(_path(UP[:-1] + [115.0], 8))              # never closes above 115 after L2
    assert signals(d, R3) == []


def test_invalidated_below_leg_low():
    crash = [(125.0, 125.2, 110.0, 110.5), (110.5, 110.6, 104.0, 105.0)]      # closes below L2 = 106
    d = _df(_path(UP, 8) + crash + [(105, 106, 104.5, 105.5)] + [CONF] * 1)
    assert signals(d, R3) == []


def test_setup_expires():
    drift = [(125.0, 125.4, 124.8, 125.1)] * 22         # no pullback into the zone for > 20 bars
    d = _df(_path(UP, 8) + drift + PULL[3:] + [CONF])
    assert signals(d, R3) == []
    assert len(signals(d, StructureRules(expiry_bars=60))) == 1       # positive control


def test_too_deep_invalidates():
    def run(low):
        bar = (114.4, 114.6, low, low + 0.2)
        return signals(_df(_path(UP, 8) + PULL[:4] + [bar, (low + 0.2, 114.0, low + 0.1, 113.9)] + [CONF]), R3)
    assert len(run(113.0)) == 1          # slightly below the 0.618 level but within 0.25 ATR: allowed
    assert run(112.0) == []              # too deep: setup invalidated


def _h1(close_level, n=120):
    idx = pd.date_range("2025-01-06 00:00", periods=n, freq="1h", tz="UTC")
    c = np.r_[np.full(n - 30, 100.0), np.full(30, close_level)]
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c}, index=idx)


def test_htf_filter_blocks_long_below_ema():
    d = _df(_path(UP, 8) + PULL + [CONF] + [(115.4, 116, 115, 115.8)] * 5, start="2025-01-10 00:00")
    rules = StructureRules(htf_filter=True)
    assert len(signals(d, rules, _h1(150.0))) == 1          # last 1h close above its EMA(50)
    assert signals(d, rules, _h1(60.0)) == []               # below
    assert len(signals(d, StructureRules(), None)) == 1


def test_htf_bias_uses_only_completed_hours():
    h1 = _h1(150.0, 80)
    idx = pd.date_range("2025-01-08 12:00", periods=8, freq="15min", tz="UTC")
    b = htf_bias(idx, h1, 50)
    full = htf_bias(idx, h1.iloc[:60], 50)               # hour bars after 12:00 removed
    assert (b[:3] == full[:3]).all()                      # bars ending <= 12:45 cannot see the 12:00 hour


def test_signals_no_lookahead():
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    o = np.r_[c[0], c[:-1]]
    w = rng.uniform(0, 0.003, (3000, 2))
    d = _df(np.c_[o, np.maximum(o, c) * (1 + w[:, 0]), np.minimum(o, c) * (1 - w[:, 1]), c])
    t = 2000
    a = signals(d.iloc[:t + 1], R3)
    d2 = d.copy(); d2.iloc[t + 1:] *= np.linspace(1, 3, len(d) - t - 1)[:, None]
    b = [x for x in signals(d2, R3) if x.idx <= t]
    assert len(a) > 5 and a == b


def test_random_walk_no_edge_and_costs_hurt():
    rng = np.random.default_rng(11)
    c = 5000 * np.exp(np.cumsum(rng.normal(0, 0.0008, 120_000)))
    o = np.r_[c[0], c[:-1]]
    w = rng.uniform(0, 0.001, (len(c), 2))
    d = _df(np.c_[o, np.maximum(o, c) * (1 + w[:, 0]), np.minimum(o, c) * (1 - w[:, 1]), c], start="2020-01-01 00:00")
    free = backtest_market(d, None, (0, 24), 0.0)
    paid = backtest_market(d, None, (0, 24), 0.1)
    t = free.r.mean() / (free.r.std() / np.sqrt(len(free)))
    assert len(free) > 300 and abs(t) < 3, (len(free), t)
    assert paid.r.mean() < free.r.mean()


def test_simulate_stop_first_gap_and_breakeven():
    d = _setup([(115.4, 116, 115, 115.8)] * 0)
    x = signals(d, R3)[0]
    bars = d.iloc[: x.idx + 1]
    px = 115.4
    # next bar touches both stop and target -> stop first
    nxt = pd.DataFrame([(px, px + 50, x.stop_level - 1, px)], columns=bars.columns,
                       index=[bars.index[-1] + pd.Timedelta(minutes=15)])
    t, _ = simulate(pd.concat([bars, nxt]), [x], R3, 0.0)
    assert list(t.why) == ["stop"] and t.r.iloc[0] < -0.9
    # gap through the stop fills at the open (worse than -1R)
    gap = pd.DataFrame([(px, px + 0.1, px - 0.1, px), (x.stop_level - 2, x.stop_level - 1, x.stop_level - 3, x.stop_level - 2)],
                       columns=bars.columns, index=pd.date_range(bars.index[-1] + pd.Timedelta(minutes=15), periods=2, freq="15min"))
    t, _ = simulate(pd.concat([bars, gap]), [x], R3, 0.0)
    assert t.r.iloc[0] < -1.0
