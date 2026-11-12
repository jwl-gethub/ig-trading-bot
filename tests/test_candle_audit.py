import numpy as np
import pandas as pd

from research.candle_audit import Tape, resolve_entry_1m, entry_window, simulate_1m
from research.candle_patterns import resolve_entry, retrace_setups
from research.candle_test import simulate

N = 10                      # 1-minute bars per signal bar
TF = "10min"
TF_NS = pd.Timedelta(TF).value


def bar(rows, px=None):
    """One signal bar = N one-minute rows (o,h,l,c); padded with flat minutes at the last close."""
    rows = list(rows)
    last = rows[-1][3] if px is None else px
    return rows + [(last, last, last, last)] * (N - len(rows))


def flat(o, h, l, c):
    return bar([(o, h, l, c)])


def build(extra):
    """warm-up + two bearish bars + bullish C (high 100.2, low 98.2) + extra signal bars (lists of minute rows)."""
    bars = [bar([(100, 101, 99, 100)]) for _ in range(20)]
    bars += [bar([(101.5, 102, 100, 100.5)]), bar([(100.5, 101, 98.5, 99.0)]), bar([(99.0, 100.2, 98.2, 99.8)])]
    bars += extra
    rows = [r for b in bars for r in b]
    idx = pd.date_range("2025-01-06", periods=len(rows), freq="1min", tz="UTC")
    m1 = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    m1["volume"] = 1.0
    sig = m1.resample(TF).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "count"})
    return m1, sig


def run(extra, tgt=2.0, cost=0.0):
    m1, sig = build(extra)
    setups = retrace_setups(sig)
    assert len(setups) == 1 and setups[0].idx == 22
    s = setups[0]
    tape = Tape.from_frame(m1)
    starts = sig.index.as_unit("ns").asi8
    t, lab, cnt = simulate_1m(tape, starts, TF_NS, setups, tgt, cost)
    o, h, lo = (sig[k].to_numpy(float) for k in ("open", "high", "low"))
    orig = resolve_entry(s, o, h, lo, len(sig))
    return s, t, lab[22], orig, (m1, sig, tape, starts)


def test_entry_then_stop_in_one_signal_bar_is_a_loss_not_a_cancel():
    s0 = retrace_setups(build([])[1])[0]
    hi = s0.stop_level + 0.1
    rows = [(99.0, 99.0, 98.1, 98.1), (98.1, hi, 98.1, hi - 0.2)]       # fills 98.2, then runs to the stop
    s, t, lab, orig, _ = run([bar(rows)])
    assert orig is None                                                   # bar-level engine cancels it
    assert lab == "stop" and len(t) == 1 and abs(t.r.iloc[0] + 1.0) < 1e-9
    assert t.ts.iloc[0] == pd.Timestamp("2025-01-06", tz="UTC") + pd.Timedelta(minutes=23 * N)


def test_cancel_first_really_cancels():
    rows = [(99.0, 100.5, 99.0, 100.4), (100.4, 100.4, 98.0, 98.5)]    # cancel level first, then trigger
    s, t, lab, orig, _ = run([bar(rows), flat(99, 99.5, 98.0, 98.5)])
    assert lab == "cancelled" and len(t) == 0 and orig is None


def test_opening_beyond_cancel_level_cancels_even_if_same_minute_touches_entry():
    rows = [(100.5, 100.6, 98.0, 99.0)]
    s, t, lab, _, _ = run([bar(rows)])
    assert lab == "cancelled" and len(t) == 0


def test_both_levels_inside_one_minute_counts_filled_and_stopped():
    s0 = retrace_setups(build([])[1])[0]
    rows = [(99.0, s0.stop_level + 0.2, 98.0, 99.0)]
    s, t, lab, _, _ = run([bar(rows)])
    assert lab == "stop" and len(t) == 1 and t.r.iloc[0] < -0.99


def test_gap_fill_at_open_and_target():
    rows = [(97.5, 97.6, 97.0, 97.2), (97.2, 97.2, 80.0, 81.0)]
    s, t, lab, orig, _ = run([bar(rows)], tgt=2.0)
    assert orig == (23, 97.5)
    assert lab == "target" and abs(t.r.iloc[0] - 2.0) < 1e-9
    assert abs(s.stop_level - 97.5 - t.risk_atr.iloc[0] * s.atr) < 1e-9


def test_stop_first_when_stop_and_target_share_a_minute():
    s0 = retrace_setups(build([])[1])[0]
    rows = [(99.0, 99.0, 98.1, 98.1), (98.1, s0.stop_level + 1, 80.0, 90.0)]
    s, t, lab, _, _ = run([bar(rows)])
    assert lab == "stop" and abs(t.r.iloc[0] + 1.0) < 1e-9


def test_expires_after_two_signal_bars():
    s, t, lab, _, _ = run([flat(99, 99.5, 98.5, 99), flat(99, 99.5, 98.5, 99), flat(99, 99.5, 97.0, 98)])
    assert lab == "expired" and len(t) == 0


def test_agrees_with_bar_engine_when_unambiguous_and_costs_apply():
    rows = [(99.0, 99.0, 98.1, 98.1), (98.1, 98.1, 90.0, 91.0)]
    m1, sig = build([bar(rows), flat(91, 92, 90, 91)])
    setups = retrace_setups(sig)
    tape, starts = Tape.from_frame(m1), sig.index.as_unit("ns").asi8
    for cost in (0.0, 0.05):
        t1, _, _ = simulate_1m(tape, starts, TF_NS, setups, 3.0, cost)
        t0, _ = simulate(sig, setups, 3.0, cost)
        assert list(t1.why) == list(t0.why) == ["target"] and abs(t1.r.iloc[0] - t0.r.iloc[0]) < 1e-9


def test_long_is_mirror_of_short():
    s0 = retrace_setups(build([])[1])[0]
    rows = [(99.0, 99.0, 98.1, 98.1), (98.1, s0.stop_level + 0.1, 98.1, 99.0)]
    m1, sig = build([bar(rows)])
    mirror = 200 - m1[["open", "low", "high", "close"]].set_axis(["open", "high", "low", "close"], axis=1)
    mirror["volume"] = 1.0
    sg = mirror.resample(TF).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "count"})
    st = retrace_setups(sg)
    assert len(st) == 1 and st[0].side == 1
    t, lab, _ = simulate_1m(Tape.from_frame(mirror), sg.index.as_unit("ns").asi8, TF_NS, st, 2.0, 0.0)[0:3]
    assert lab[22] == "stop" and abs(t.r.iloc[0] + 1.0) < 1e-9


def test_causal_truncating_future_does_not_change_past_trades():
    rng = np.random.default_rng(5)
    n = 60_000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.r_[c[0], c[:-1]]
    w = rng.uniform(0, 0.0006, (n, 2))
    m1 = pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + w[:, 0]), "low": np.minimum(o, c) * (1 - w[:, 1]),
                       "close": c, "volume": 1.0}, index=pd.date_range("2025-01-06", periods=n, freq="1min", tz="UTC"))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "count"}

    def go(m):
        sig = m.resample("1h").agg(agg)
        return simulate_1m(Tape.from_frame(m), sig.index.as_unit("ns").asi8, pd.Timedelta("1h").value,
                           retrace_setups(sig), 2.0, 0.0)[0]
    full = go(m1)
    cut = m1.index[40_000]
    part = go(m1.iloc[:40_000])
    a = full[full.xts < cut - pd.Timedelta("1D")].reset_index(drop=True)
    b = part[part.xts < cut - pd.Timedelta("1D")].reset_index(drop=True)
    assert len(a) > 20 and a.equals(b)
