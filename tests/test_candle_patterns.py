import numpy as np
import pandas as pd

from research.candle_patterns import engulf_setups, resolve_entry, retrace_setups
from research.candle_test import simulate


def _df(rows, freq="4h"):
    a = np.asarray(rows, float)
    idx = pd.date_range("2025-01-06 00:00", periods=len(a), freq=freq, tz="UTC")
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]}, index=idx)


def _warm(n=20, px=100.0):
    """Neutral warm-up bars (open == close, range 2) so ATR is defined and no pattern fires."""
    return [(px, px + 1, px - 1, px) for k in range(n)]


BULL = [(100, 102, 99.5, 101.5), (101.5, 103, 101, 102.5)]            # two bullish context bars; prev high 103, low 101
BEAR = [(102.5, 103, 100, 101), (101, 101.5, 99, 99.5)]               # two bearish; prev low 99, high 101.5


def test_long_engulf_fires_with_high_rule_and_stop_at_signal_low():
    sig = (102.5, 103.6, 100.0, 103.4)                                 # low<101, close>103 (prev high)
    d = _df(_warm() + BULL + [sig, (103.4, 104, 103, 103.5)])
    s = engulf_setups(d, "high")
    assert len(s) == 1 and s[0].side == 1 and s[0].idx == 22
    assert 99.4 < s[0].stop_level < 100.0                       # signal low minus 0.1*ATR(~2)


def test_close_rule_is_looser_than_high_rule():
    sig = (102.5, 103.0, 100.0, 102.8)                                 # close > prev close 102.5 but < prev high 103
    d = _df(_warm() + BULL + [sig, (102.8, 103, 102, 102.5)])
    assert engulf_setups(d, "high") == []
    assert len(engulf_setups(d, "close")) == 1


def test_long_engulf_requires_sweep_and_context():
    no_sweep = (102.5, 103.6, 101.5, 103.4)                            # low not below prev low 101
    assert engulf_setups(_df(_warm() + BULL + [no_sweep]), "high") == []
    mixed = [(100, 102, 99.5, 101.5), (102.5, 103, 101, 101.2)]        # second context bar bearish
    sig = (102.5, 103.6, 100.0, 103.4)
    assert engulf_setups(_df(_warm() + mixed + [sig]), "high") == []
    assert engulf_setups(_df(_warm() + BULL[1:] + [sig]), "high") == []     # only one bullish context bar


def test_short_engulf_is_exact_mirror():
    sig = (99.5, 102.0, 98.9, 98.6)                                    # high>101.5, close<99 (prev low)
    d = _df(_warm() + BEAR + [sig, (98.6, 99, 98, 98.5)])
    s = engulf_setups(d, "high")
    assert len(s) == 1 and s[0].side == -1 and 102.0 < s[0].stop_level < 102.6
    # price mirror image of the whole series gives the long signal at the same bar
    m = 200 - d[["open", "low", "high", "close"]].rename(columns={"low": "high", "high": "low"})
    ml = engulf_setups(m[["open", "high", "low", "close"]], "high")
    assert [(x.idx, x.side) for x in ml] == [(s[0].idx, 1)]


def test_engulf_no_lookahead():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000)))
    o = np.r_[c[0], c[:-1]]
    w = rng.uniform(0, 0.006, (4000, 2))
    d = _df(np.c_[o, np.maximum(o, c) * (1 + w[:, 0]), np.minimum(o, c) * (1 - w[:, 1]), c])
    t = 3000
    d2 = d.copy()
    d2.iloc[t + 1:] *= np.linspace(1, 3, len(d) - t - 1)[:, None]
    for fn in (lambda x: engulf_setups(x, "high"), lambda x: engulf_setups(x, "close"), retrace_setups):
        a = fn(d.iloc[:t + 1])
        b = [x for x in fn(d2) if x.idx <= t]
        assert len(a) > 5 and a == b


# ---- strategy B ----
def _retrace_short(c_bar=(99.0, 100.2, 98.2, 99.8)):
    # two bearish bars then bullish C (high 100.2, low 98.2)
    return _df(_warm() + [(101.5, 102, 100, 100.5), (100.5, 101, 98.5, 99.0), c_bar], freq="1D")


def test_retrace_short_setup_levels():
    s = retrace_setups(_retrace_short())
    assert len(s) == 1 and s[0].side == -1 and s[0].idx == 22
    assert s[0].trigger == 98.2 and s[0].cancel == 100.2 and 100.2 < s[0].stop_level < 100.6


def test_retrace_needs_two_bearish_then_bullish():
    d = _df(_warm() + [(100.5, 101, 98.5, 100.7), (100.5, 101, 98.5, 99.0), (99.0, 100.2, 98.2, 99.8)], freq="1D")
    assert retrace_setups(d) == []                                      # first context bar is bullish
    d = _df(_warm() + [(101.5, 102, 100, 100.5), (100.5, 101, 98.5, 99.0), (99.0, 99.5, 97.0, 97.5)], freq="1D")
    assert [x for x in retrace_setups(d) if x.idx == 22] == []          # C is bearish after bearish bars: no setup


def _fill(extra):
    d = _retrace_short()
    d = pd.concat([d, _df(extra, freq="1D").set_axis(pd.date_range(d.index[-1] + pd.Timedelta("1D"), periods=len(extra), freq="1D"))])
    s = retrace_setups(d)[0]
    o, h, lo = (d[k].to_numpy(float) for k in ("open", "high", "low"))
    return resolve_entry(s, o, h, lo, len(d))


def test_retrace_fills_on_first_or_second_bar_only():
    assert _fill([(99.0, 99.5, 98.0, 98.5)]) == (23, 98.2)             # trades through C's low
    assert _fill([(99.0, 99.5, 98.5, 99.0), (99.0, 99.5, 98.0, 98.5)]) == (24, 98.2)
    assert _fill([(99.0, 99.5, 98.5, 99.0)] * 2 + [(99.0, 99.5, 98.0, 98.5)]) is None   # third bar: expired


def test_retrace_gap_fills_at_open_and_cancel_rules():
    assert _fill([(97.5, 98.0, 97.0, 97.8)]) == (23, 97.5)             # gap below the trigger
    assert _fill([(99.0, 100.3, 98.5, 100.0), (99.0, 99.5, 98.0, 98.5)]) is None       # C's high traded first
    assert _fill([(99.0, 100.3, 98.0, 99.0)]) is None                   # one bar touches both -> cancelled
    assert _fill([(99.0, 100.2, 98.0, 99.0)])[1] == 98.2                # high only equal to C's high: not 'above'


def test_retrace_long_is_mirror():
    d = _retrace_short()
    m = 200 - d[["open", "low", "high", "close"]].set_axis(["open", "high", "low", "close"], axis=1)
    s = retrace_setups(m)
    assert len(s) == 1 and s[0].side == 1 and s[0].trigger == 200 - 98.2 and s[0].cancel == 200 - 100.2
    ext = _df([(101.0, 102.0, 100.5, 101.5)], freq="1D")                # next bar trades above trigger 101.8
    m2 = pd.concat([m, ext.set_axis(pd.date_range(m.index[-1] + pd.Timedelta("1D"), periods=1, freq="1D"))])
    o, h, lo = (m2[k].to_numpy(float) for k in ("open", "high", "low"))
    assert resolve_entry(s[0], o, h, lo, len(m2)) == (23, 101.8)


def test_simulate_target_stop_and_one_position():
    d = _retrace_short()
    ext = [(99.0, 99.5, 98.0, 98.5), (98.5, 98.6, 90.0, 91.0), (91, 92, 90, 91)]       # fills 98.2, then hits 3R target
    d = pd.concat([d, _df(ext, freq="1D").set_axis(pd.date_range(d.index[-1] + pd.Timedelta("1D"), periods=3, freq="1D"))])
    t, _ = simulate(d, retrace_setups(d), 3.0, 0.0)
    assert list(t.why) == ["target"] and abs(t.r.iloc[0] - 3.0) < 1e-9
    # costs reduce R
    t2, _ = simulate(d, retrace_setups(d), 3.0, 0.05)
    assert t2.r.iloc[0] < t.r.iloc[0]
    # stop first when the same bar touches stop and target is impossible for B by construction; engulf case:
    sig = (102.5, 103.6, 100.0, 103.4)
    e = _df(_warm() + BULL + [sig, (103.4, 110, 90, 100)])             # next bar touches stop and 2R
    t3, _ = simulate(e, engulf_setups(e, "high"), 2.0, 0.0)
    assert list(t3.why) == ["stop"] and t3.r.iloc[0] < -0.9


def test_spread_guard_skips_tight_stops():
    sig = (102.5, 103.6, 100.0, 103.4)
    e = _df(_warm() + BULL + [sig, (103.4, 104, 103, 103.5)])
    t, sk = simulate(e, engulf_setups(e, "high"), 2.0, 5.0)           # cost huge relative to stop
    assert len(t) == 0 and sk["skipped_spread"] == 1


def test_random_walk_no_edge():
    rng = np.random.default_rng(21)
    n = 60_000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    w = rng.uniform(0, 0.004, (n, 2))
    d = _df(np.c_[o, np.maximum(o, c) * (1 + w[:, 0]), np.minimum(o, c) * (1 - w[:, 1]), c], freq="4h")
    for setups, tgt in ((engulf_setups(d, "high"), 2.0), (retrace_setups(d), 3.0)):
        free, _ = simulate(d, setups, tgt, 0.0)
        paid, _ = simulate(d, setups, tgt, 0.02)
        t = free.r.mean() / (free.r.std() / np.sqrt(len(free)))
        assert len(free) > 300 and abs(t) < 3.5, (len(free), t)
        assert paid.r.mean() < free.r.mean()
