import numpy as np
import pandas as pd

from research.noise_area import Params, backtest, indicators, session_matrix

M = 390


def _sm(paths: np.ndarray, start="2024-01-01") -> dict:
    """Synthetic session matrix from close paths (days x minutes)."""
    D = paths.shape[0]
    o = np.c_[paths[:, :1], paths[:, :-1]]
    hi = np.maximum(o, paths) * 1.00002
    lo = np.minimum(o, paths) * 0.99998
    return {"open": o, "high": hi, "low": lo, "close": paths, "volume": np.ones_like(paths),
            "days": pd.bdate_range(start, periods=D)}


def _random_walk(D, seed=0, vol=0.0006):
    rng = np.random.default_rng(seed)
    out, px = [], 5000.0
    for _ in range(D):
        gap = px * (1 + rng.normal(0, 0.003))
        path = gap * np.exp(np.cumsum(rng.normal(0, vol, M)))
        out.append(path)
        px = path[-1]
    return np.array(out)


def test_session_matrix_uses_exchange_local_time():
    idx = pd.date_range("2025-01-06 14:00", "2025-01-06 21:30", freq="1min", tz="UTC")  # winter: ET = UTC-5
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(len(idx), dtype=float),
                       "volume": 1.0}, index=idx)
    sm = session_matrix(df, "US500")
    assert sm["close"].shape == (1, 390)
    assert sm["close"][0, 0] == 30            # 14:30 UTC = 09:30 ET is the first minute
    assert sm["close"][0, -1] == 30 + 389     # 20:59 UTC = 15:59 ET is the last


def test_no_lookahead_in_noise_area():
    paths = _random_walk(40, seed=1)
    p = Params()
    a = indicators(_sm(paths), p)
    changed = paths.copy()
    changed[30:] *= 1.5                       # rewrite the future
    b = indicators(_sm(changed), p)
    assert np.allclose(a["upper"][:30], b["upper"][:30], equal_nan=True)
    assert np.allclose(a["lower"][:30], b["lower"][:30], equal_nan=True)
    assert np.allclose(a["lev"][:30], b["lev"][:30], equal_nan=True)
    assert np.allclose(a["ref"][29], b["ref"][29])                  # same-day VWAP up to each minute


def test_no_edge_on_random_walk():
    paths = _random_walk(700, seed=3)
    free = backtest(_sm(paths), Params(cost_side_points=0.0, vol_target=False))[1].ret
    costly = backtest(_sm(paths), Params(cost_side_points=0.3, vol_target=False))[1].ret
    t_stat = free.mean() / (free.std() / np.sqrt(len(free)))
    assert abs(t_stat) < 3, f"suspicious edge on random data: t={t_stat:.2f}"
    assert costly.sum() < free.sum()


def test_trend_day_goes_long_and_is_flat_at_close():
    calm = _random_walk(20, seed=5, vol=0.0002)
    last = calm[-1, -1]
    trend = last * np.exp(np.linspace(0, 0.02, M))                  # +2% steady up day
    paths = np.vstack([calm, trend])
    trades, daily = backtest(_sm(paths), Params(cost_side_points=0.3))
    t = trades[trades.day == trades.day.max()]
    assert len(t) == 1 and t.iloc[0].dir == 1 and t.iloc[0].reason == "close"
    assert t.iloc[0].ret > 0 and daily.iloc[-1].ret > 0
    sm = _sm(paths)
    assert np.isclose(t.iloc[0].entry, sm["open"][-1, int(t.iloc[0].t_in)] + 0.3)   # filled next minute


def test_hard_stop_caps_the_loss():
    calm = _random_walk(20, seed=7, vol=0.0002)
    last = calm[-1, -1]
    up = last * np.exp(np.linspace(0, 0.01, 60))
    crash = up[-1] * np.exp(np.linspace(0, -0.06, M - 60))           # breakout then a 6% collapse
    paths = np.vstack([calm, np.r_[up, crash]])
    no_stop = backtest(_sm(paths), Params(cost_side_points=0.3, check_every_min=120))[0]
    stop = backtest(_sm(paths), Params(cost_side_points=0.3, check_every_min=120, hard_stop_sigma=1.0))[0]
    day = no_stop.day.max()
    assert stop[stop.day == day].reason.eq("hard_stop").any()
    assert stop[stop.day == day].ret.min() > no_stop[no_stop.day == day].ret.min()


def test_dukascopy_parser_finds_columns_and_scale():
    import lzma
    import struct
    from datetime import date
    from research.dukascopy import parse_bi5, to_frame
    rows = [(60 * i, 5000123 + i, 5000456 + i, 4999900 + i, 5000900 + i, 1.5) for i in range(5)]  # O,C,L,H
    raw = lzma.compress(b"".join(struct.pack(">5if", *r) for r in rows), format=lzma.FORMAT_ALONE)
    df = to_frame(parse_bi5(raw, date(2025, 1, 6)), (1500, 12000))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.iloc[0].open == 5000.123 and df.iloc[0].close == 5000.456
    assert df.iloc[0].low == 4999.9 and df.iloc[0].high == 5000.9
    assert str(df.index[1]) == "2025-01-06 00:01:00+00:00"


def test_last_half_hour_follows_the_day_and_ignores_the_future():
    from research.noise_area import last_half_hour
    calm = _random_walk(20, seed=9, vol=0.0002)
    up_day = calm[-1, -1] * np.exp(np.linspace(0, 0.01, M))
    paths = np.vstack([calm, up_day])
    trades, daily = last_half_hour(_sm(paths), Params(cost_side_points=0.0))
    last = trades[trades.day == trades.day.max()].iloc[0]
    assert last.dir == 1 and last.t_in == M - 30 and last.ret > 0
    changed = paths.copy()
    changed[-1, M - 30:] = changed[-1, M - 31]                       # flatten the final half hour
    t2, _ = last_half_hour(_sm(changed), Params(cost_side_points=0.0))
    assert t2[t2.day == t2.day.max()].iloc[0].dir == 1                # same signal: no lookahead
