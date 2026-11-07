import numpy as np
import pandas as pd

from bot.data import synthetic_market
from bot.indicators import compute_all, rsi


def test_no_lookahead(cfg):
    """Indicator values at bar t must not change when future bars are appended."""
    df = synthetic_market(3, 800, 5000.0, 0.004, regimes=True)
    full = compute_all(df, cfg["strategy"])
    for cut in (320, 500, 799):
        part = compute_all(df.iloc[:cut], cfg["strategy"])
        a = full.iloc[:cut]
        num = [c for c in part.columns if part[c].dtype.kind in "fi"]
        pd.testing.assert_frame_equal(part[num], a[num], check_exact=False, rtol=1e-9, atol=1e-9)
        for c in ("regime", "vol_ok", "ready", "touched_fast_long", "touched_fast_short"):
            assert (part[c].values == a[c].values).all(), c


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up, 14).dropna().iloc[-1] == 100.0
    flat = pd.Series(np.ones(60))
    assert rsi(flat, 14).dropna().iloc[-1] == 50.0
    rnd = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=500)) + 100)
    r = rsi(rnd, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_ready_after_warmup(cfg):
    df = synthetic_market(4, 400, 1000.0, 0.003)
    out = compute_all(df, cfg["strategy"])
    assert not out["ready"].iloc[:200].any()
    assert out["ready"].iloc[-1]
    assert (out["atr"].dropna() > 0).all()


def test_bad_tick_does_not_blow_up_atr(cfg):
    df = synthetic_market(9, 400, 25_000.0, 0.004)
    clean = compute_all(df, cfg["strategy"])
    bad = df.copy()
    ts = bad.index[380]
    bad.loc[ts, "high"] = bad.loc[ts, "close"] + 7_000          # absurd spike wick
    dirty = compute_all(bad, cfg["strategy"])
    ratio = dirty["atr"].iloc[-1] / clean["atr"].iloc[-1]
    assert ratio < 1.6, f"ATR inflated x{ratio:.1f} by one bad tick"
    assert dirty["data_spike"].loc[ts]
    assert not dirty["vol_ok"].loc[ts]                          # no entries on the spike bar
    assert dirty["dc_high"].iloc[-1] < bad.loc[ts, "high"]      # Donchian not poisoned


def test_real_trend_bodies_survive_cleaning(cfg):
    from bot.indicators import clean_bars
    df = synthetic_market(10, 300, 1000.0, 0.003)
    out = clean_bars(df)
    assert (out["open"] == df["open"]).all() and (out["close"] == df["close"]).all()
