"""Tests for research/candidates (synthetic data only): timing, no lookahead, funding, band, DST, no edge on random walks."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research import cand_s1, cand_s23, cand_s4
from research.cand_common import UNIVERSE, annual_rate, funding_frac, night_weight, stress_rate
from research.cand_data import daily_21utc
from research.cand_s1 import BAND, Panel, band_trade, build_units, report_run, target_weights
from research.cand_s23 import Hr, RipCtx, rip_market, tom_market, tom_schedule
from research.cand_s4 import Minute, ny_instants, trades_4a, trades_4b

NS = 10 ** 9
DAY = 86400 * NS


def ns(s: str) -> int:
    return pd.Timestamp(s, tz="UTC").value


def rw_hourly(seed: int, years: float = 8, start: str = "2012-01-02", price: float = 100.0, vol: float = 0.002):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(years * 365 * 24), freq="1h", tz="UTC")
    wd, hr = idx.dayofweek, idx.hour
    keep = ~((wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 21)))
    idx = idx[keep]
    c = price * np.exp(np.cumsum(rng.normal(0, vol, len(idx))))
    o = np.r_[price, c[:-1]]
    w = np.abs(rng.normal(0, vol / 2, (2, len(idx))))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + w[0]), "low": np.minimum(o, c) * (1 - w[1]),
                         "close": c, "volume": 1.0}, index=idx)


def rw_minute(seed: int, days: int = 700, start: str = "2022-01-03", vol: float = 0.0002):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    wd, hr = idx.dayofweek, idx.hour
    idx = idx[~((wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 21)))]
    c = 100 * np.exp(np.cumsum(rng.normal(0, vol, len(idx))))
    return pd.DataFrame({"open": np.r_[100, c[:-1]], "close": c}, index=idx)


# ------------------------------------------------------------------------------------------ daily bars
def test_daily_bars_close_21utc_and_weekend_to_monday():
    idx = pd.to_datetime(["2024-01-05 19:00", "2024-01-05 20:00", "2024-01-05 21:00", "2024-01-07 22:00",
                          "2024-01-08 08:00", "2024-01-08 20:00", "2024-01-08 21:00"], utc=True)
    h = pd.DataFrame({"open": [1, 2, 3, 4, 5, 6, 7.0], "high": 9.0, "low": 0.5, "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]},
                     index=idx)
    d = daily_21utc(h, min_bars=1)
    assert list(d.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09")]
    assert d.close.iloc[0] == 2.5                         # last bar starting before 21:00
    assert d.open.iloc[1] == 3 and d.close.iloc[1] == 6.5  # Friday 21:00 and Sunday evening bars belong to Monday


# ------------------------------------------------------------------------------------------ turn of month
def test_tom_indexing_holidays_and_short_months():
    days = pd.bdate_range("2023-12-01", "2024-04-30")
    days = days.drop([pd.Timestamp("2023-12-25"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15"),
                      pd.Timestamp("2024-02-19"), pd.Timestamp("2024-03-29"), pd.Timestamp("2024-03-28")])
    s = tom_schedule(days, 2)
    got = [(days[i].date().isoformat(), days[j].date().isoformat()) for i, j in s]
    # Dec: last days 28,29 -> day -2 = Dec 28; Jan trading days: 2,3,4 -> day +3 = Jan 4
    assert got[0] == ("2023-12-28", "2024-01-04")
    # Jan: ends 30,31 -> -2 = Jan 30; Feb 1,2,5 -> +3 = Feb 5
    assert got[1] == ("2024-01-30", "2024-02-05")
    # Feb (leap, 29 days): last trading days 28, 29 -> -2 = Feb 28; Mar 1,4,5 -> +3 = Mar 5
    assert got[2] == ("2024-02-28", "2024-03-05")
    # Mar: 27 is the last trading day before the Good Friday gap (28,29 missing): last two are 26?27 -> -2 = Mar 26
    assert got[3][0] == "2024-03-26" and got[3][1] == "2024-04-03"
    assert len(got) == 4                                   # April has no following month: no trade
    s5 = tom_schedule(days, 5)
    assert days[s5[0][0]] == pd.Timestamp("2023-12-22")   # day -5 with Dec 25 missing: 22,26,27,28,29


def test_tom_no_lookahead():
    days = pd.bdate_range("2015-01-01", "2020-12-31")
    full = tom_schedule(days, 2)
    cut = 900
    part = tom_schedule(days[:cut], 2)
    assert part[:-1] == full[:len(part) - 1]


def test_tom_stop_checked_on_hourly_bars_stop_first():
    m = replace(UNIVERSE["US500"], spread=0.0)
    idx = pd.bdate_range("2024-01-01", periods=40)
    n = len(idx)
    px = np.full(n, 100.0)
    d = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px, "n": 20,
                      "open_ts": [(t + pd.Timedelta(hours=-2)).tz_localize("UTC") for t in idx],
                      "close_ts": [(t + pd.Timedelta(hours=21)).tz_localize("UTC") for t in idx]}, index=idx)
    atr = np.full(n, 2.0)                                  # stop = 100 - 6 = 94
    hts = np.array([ns("2024-01-11 10:00"), ns("2024-01-12 10:00")])
    hr = Hr(pd.DataFrame({"open": [100.0, 99.0], "high": 101.0, "low": [93.0, 90.0], "close": 100.0},
                         index=pd.to_datetime(hts, utc=True)))
    # trade from day 8 (Jan 11) close to day 10 (Jan 15) close: first hourly bar after entry is Jan 12 10:00? entry close_ts Jan 11 21:00
    i = list(idx).index(pd.Timestamp("2024-01-11"))
    t = cand_s23.sim_tom(m, d, atr, hr, i + 1, i + 3)   # entry Jan 12 close: both low bars are before it
    assert t["why"] == "time"                              # Jan 11 10:00 bar is BEFORE the entry close: ignored
    t2 = cand_s23.sim_tom(m, d, atr, hr, i - 1, i + 2)     # entry Jan 10 close: Jan 11 10:00 bar low 93 <= 94 -> stop
    assert t2["why"] == "stop" and t2["r"] == pytest.approx((94 - 100) / 6, abs=0.05)


# ------------------------------------------------------------------------------------------ funding
def test_night_weights_friday_triple():
    assert list(night_weight(np.arange(7))) == [1, 1, 1, 1, 3, 0, 0]


def test_funding_friday_x3_fx():
    fx = UNIVERSE["EURUSD"]
    f = funding_frac(fx, 1, [ns("2024-03-08 15:00")], [ns("2024-03-11 10:00")])   # holds Fri 21:00 -> x3
    assert f[0] == pytest.approx(3 * 0.015 / 365)
    mon = funding_frac(fx, -1, [ns("2024-03-04 15:00")], [ns("2024-03-05 10:00")])
    assert mon[0] == pytest.approx(0.015 / 365)
    assert funding_frac(fx, 1, [ns("2024-03-04 08:00")], [ns("2024-03-04 20:00")])[0] == 0.0


def test_funding_index_long_pays_short_credit():
    us = UNIVERSE["US500"]
    L = funding_frac(us, 1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")])[0]
    S = funding_frac(us, -1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")])[0]
    assert L == pytest.approx((0.0514 + 0.025) / 365) and S == pytest.approx((0.025 - 0.0514) / 365) and S < 0
    uk = UNIVERSE["UK100"]
    assert funding_frac(uk, 1, [ns("2019-05-06 15:00")], [ns("2019-05-07 15:00")])[0] == pytest.approx((0.0075 + 0.025) / 365)
    ger = UNIVERSE["GER40"]
    assert funding_frac(ger, 1, [ns("2021-05-06 15:00")], [ns("2021-05-07 15:00")])[0] == pytest.approx((-0.0057 + 0.025) / 365)


def test_funding_gold_other_and_stress():
    g = UNIVERSE["GOLD"]
    assert funding_frac(g, 1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")])[0] == pytest.approx((0.0514 + 0.015) / 365)
    assert funding_frac(g, -1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")])[0] == pytest.approx((0.015 - 0.0514) / 365)
    assert funding_frac(UNIVERSE["WTI"], 1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")])[0] == pytest.approx(0.034 / 365)
    s = funding_frac(g, -1, [ns("2024-05-06 15:00")], [ns("2024-05-07 15:00")], 1.5)[0]
    assert s == pytest.approx((0.015 - 0.0514) / 365)       # credits are not scaled up
    assert stress_rate(np.array([0.02, -0.01]), 1.5).tolist() == pytest.approx([0.03, -0.01])
    assert annual_rate(UNIVERSE["US500"], 1, 2031) == pytest.approx(0.037 + 0.025)    # years past the table use 2026


# ------------------------------------------------------------------------------------------ TSMOM
def test_band():
    assert band_trade(1.0, 0.0)
    assert not band_trade(1.0, 0.8)           # |0.2| = 20% of target: inside the band
    assert band_trade(1.0, 0.7)               # 30% off
    assert band_trade(-1.0, 0.5)              # sign change
    assert band_trade(0.0, 0.1) and not band_trade(0.0, 0.0)
    assert not band_trade(1.0, 1.25) and band_trade(1.0, 1.3)
    assert BAND == 0.25


def daily_dict(seed0: int, names=("US500", "GER40", "EURUSD", "GOLD", "UK100", "USDJPY"), years=9):
    return {n: daily_21utc(rw_hourly(seed0 + k, years=years)) for k, n in enumerate(names)}


def test_tsmom_no_lookahead():
    daily = daily_dict(10, years=7)
    cut = pd.Timestamp("2016-06-30")
    trunc = {n: d[d.index <= cut] for n, d in daily.items()}
    pf_, pt = Panel(daily), Panel(trunc)
    pf_.features(daily)
    pt.features(trunc)
    k = len(pt.dates)
    assert np.allclose(pf_.sgn[:k], pt.sgn, equal_nan=True) and np.allclose(pf_.sig[:k], pt.sig, equal_nan=True)
    assert (pf_.dec[:k - 8] == pt.dec[:k - 8]).all()
    tw_f, tw_t = target_weights(pf_, np.nanmean(pf_.sgn, axis=2)), target_weights(pt, np.nanmean(pt.sgn, axis=2))
    assert np.allclose(tw_f[:k - 8], tw_t[:k - 8], equal_nan=True)
    uf, ut = build_units(pf_, tw_f), build_units(pt, tw_t)
    assert np.allclose(uf[:k - 8], ut[:k - 8])


def test_tsmom_execution_is_next_open_not_decision_close():
    daily = daily_dict(20, years=3)
    p = Panel(daily)
    p.features(daily)
    tw = target_weights(p, np.nanmean(p.sgn, axis=2))
    U = build_units(p, tw)
    for j in range(p.M):
        ch = np.flatnonzero(np.abs(np.diff(U[:, j], prepend=0.0)) > 0)
        for e in ch[:5]:
            assert p.dec[:e, j].any()
            prev_dec = np.flatnonzero(p.dec[:e, j])[-1]
            assert prev_dec < e                            # decided strictly before the bar it executes on


def zero_costs(p: Panel) -> None:
    p.hs[:] = 0
    p.Rl[:] = 0
    p.Rs[:] = 0


def test_tsmom_random_walk_no_edge_before_costs():
    sh = []
    for seed in range(6):
        daily = daily_dict(100 + 10 * seed, years=12)
        p = Panel(daily)
        p.features(daily)
        zero_costs(p)
        U = build_units(p, target_weights(p, np.nanmean(p.sgn, axis=2)))
        res, _ = report_run("T", "t", "x", p, U, quiet=True)
        sh.append(res["pooled"]["sharpe"])
    assert abs(np.mean(sh)) < 0.45, sh


def test_tsmom_costs_hurt():
    daily = daily_dict(5, years=8)
    p = Panel(daily)
    p.features(daily)
    U = build_units(p, target_weights(p, np.nanmean(p.sgn, axis=2)))
    base, _ = report_run("T", "t", "x", p, U, quiet=True)
    stress, _ = report_run("T", "t", "x", p, U, 1.5, quiet=True)
    assert stress["pooled"]["ann"] < base["pooled"]["ann"]


# ------------------------------------------------------------------------------------------ strategy 3
def rip_ctx(seed, name="US500", years=10):
    h = rw_hourly(seed, years=years)
    return RipCtx(UNIVERSE[name], h, cand_s23.DipRules(long_only=False, rsi_short=90.0)), h


def test_rip_no_lookahead_signals_and_entries():
    x, h = rip_ctx(3)
    cut = h.index[len(h) * 3 // 4]
    xt = RipCtx(UNIVERSE["US500"], h[h.index < cut], cand_s23.DipRules(long_only=False, rsi_short=90.0))
    k = len(xt.d) - 2
    for a in ("rsi", "sma_t", "sma_x", "atr_pct"):
        assert np.allclose(getattr(x, a)[:k], getattr(xt, a)[:k], equal_nan=True)
    tf, tt = rip_market(x, 90.0), rip_market(xt, 90.0)
    early = tt[tt.xts < cut - pd.Timedelta(days=30)]
    assert len(early) > 3
    assert early.ts.tolist() == tf[tf.xts < cut - pd.Timedelta(days=30)].ts.tolist()[:len(early)]


def test_rip_short_sign_and_rules():
    x, _ = rip_ctx(4)
    t = rip_market(x, 90.0)
    assert len(t) > 20
    cand = np.flatnonzero(x.ok & (x.c < x.sma_t) & (x.rsi > 90))
    assert len(cand) >= len(t) and (x.c[cand] < x.sma_t[cand]).all()
    assert (t.why.isin(["exit", "time", "stop"])).all()
    assert (t.xts > t.ts).all()


def test_rip_random_walk_no_edge_before_costs(monkeypatch):
    monkeypatch.setattr(cand_s23, "funding_frac", lambda *a, **k: np.zeros(1))
    allr = []
    for seed in range(8):
        for name in ("US500", "GBPUSD"):
            h = rw_hourly(300 + seed, years=14)
            m = replace(UNIVERSE[name], spread=0.0)
            x = RipCtx(m, h, cand_s23.DipRules(long_only=False, rsi_short=90.0))
            allr.append(cand_s23.rip_allhold(x))
    t = pd.concat(allr)
    tt = t.ret.mean() / (t.ret.std() / np.sqrt(len(t) / 6))     # overlapping holds: deflate n generously
    assert abs(tt) < 3.5, tt


# ------------------------------------------------------------------------------------------ TOM random walk
def test_tom_random_walk_no_edge_before_costs(monkeypatch):
    monkeypatch.setattr(cand_s23, "funding_frac", lambda *a, **k: np.zeros(1))
    rets = []
    for seed in range(6):
        h = rw_hourly(500 + seed, years=14)
        d = daily_21utc(h)
        m = replace(UNIVERSE["US500"], spread=0.0)
        _, allh = tom_market(m, d, Hr(h), 2)
        rets.append(allh)
    t = pd.concat(rets)
    tt = t.ret.mean() / (t.ret.std() / np.sqrt(len(t) / 4))
    assert abs(tt) < 3.5, tt


# ------------------------------------------------------------------------------------------ strategy 4
def ny_local(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")


def test_ny_dst_handling():
    d = pd.DatetimeIndex(["2024-01-10", "2024-07-10", "2024-03-08", "2024-03-11", "2024-11-01", "2024-11-04"])
    close, openp = ny_instants(d, 16, 0), ny_instants(d, 9, 35)
    hrs = [pd.Timestamp(x, unit="ns").hour for x in close]
    assert hrs == [21, 20, 21, 20, 20, 21]                  # EST = UTC-5, EDT = UTC-4 (US switches 10 Mar, 3 Nov 2024)
    assert [pd.Timestamp(x, unit="ns").strftime("%H:%M") for x in openp] == ["14:35", "13:35", "14:35", "13:35", "13:35", "14:35"]


def test_4a_entries_exits_at_ny_times_across_dst_and_no_friday():
    m1 = rw_minute(1, days=800, start="2022-01-03")
    m = replace(UNIVERSE["US500"], spread=0.0)
    t = trades_4a(m, Minute(m1))
    assert len(t) > 300
    tin = t.ts.dt.tz_convert("America/New_York")
    tout = t.xts.dt.tz_convert("America/New_York")
    assert (tin.dt.hour == 16).all() and (tin.dt.minute == 0).all()
    assert (tout.dt.hour == 9).all() and (tout.dt.minute == 35).all()
    assert (tin.dt.dayofweek <= 3).all()                    # Monday-Thursday nights only
    assert (tout.dt.dayofweek == (tin.dt.dayofweek + 1) % 7).all()
    assert set(tin.dt.month) == set(range(1, 13))           # includes both DST regimes


def test_4b_window_and_days():
    m1 = rw_minute(2, days=400, start="2022-01-03")
    t = trades_4b(replace(UNIVERSE["GOLD"], spread=0.0), Minute(m1))
    assert (t.ts.dt.hour == 23).all() and (t.xts.dt.hour == 7).all()
    assert set(t.ts.dt.dayofweek) == {6, 0, 1, 2, 3}        # Sunday..Thursday starts
    assert ((t.xts - t.ts) == pd.Timedelta(hours=8)).all()
    full = trades_4b(UNIVERSE["GOLD"], Minute(m1))
    assert (full.bps < t.bps + 1e-9).all()                  # spread costs reduce every trade; no funding crosses 21:00


def test_4_random_walk_no_edge_before_costs(monkeypatch):
    monkeypatch.setattr(cand_s4, "funding_frac", lambda m, d, a, b, s=1.0: np.zeros(len(a)))
    tts_a, tts_b = [], []
    for seed in range(3):
        m1 = rw_minute(600 + seed, days=1100)
        a = Minute(m1)
        tts_a.append(trades_4a(replace(UNIVERSE["US500"], spread=0.0), a))
        tts_b.append(trades_4b(replace(UNIVERSE["GOLD"], spread=0.0), a))
    for t in (pd.concat(tts_a), pd.concat(tts_b)):
        tt = t.bps.mean() / (t.bps.std() / np.sqrt(len(t)))
        assert abs(tt) < 3.5, tt
        assert t.bps.mean() == pytest.approx(t.gross.mean(), abs=1e-6)     # no costs -> net == gross


def test_4_costs_are_charged():
    m1 = rw_minute(7, days=300)
    a = Minute(m1)
    free = trades_4a(replace(UNIVERSE["US500"], spread=0.0), a)
    paid = trades_4a(UNIVERSE["US500"], a)
    assert (paid.bps.to_numpy() < free.bps.to_numpy()).all()
    assert (paid.gross.to_numpy() == pytest.approx(free.gross.to_numpy()))


# ------------------------------------------------------------------------------------------ added after review
def test_4a_holiday_without_price_skips_trade_not_multi_night():
    m1 = rw_minute(9, days=60, start="2024-01-02")
    day = pd.Timestamp("2024-01-18", tz="UTC")                    # drop a whole Thursday: no 09:35 price that day
    m1 = m1[(m1.index < day) | (m1.index >= day + pd.Timedelta(days=1))]
    t = trades_4a(replace(UNIVERSE["US500"], spread=0.0), Minute(m1))
    tin = t.ts.dt.tz_convert("America/New_York")
    tout = t.xts.dt.tz_convert("America/New_York")
    assert (tout.dt.dayofweek == (tin.dt.dayofweek + 1) % 7).all()
    assert ((t.xts - t.ts) < pd.Timedelta(hours=20)).all()          # always a single overnight hold


def test_split_mask_period_pooled():
    from research.candidates import split_mask_period
    idx = pd.period_range("2018-11", "2023-02", freq="W")
    assert split_mask_period(idx, "pooled").all()
    assert split_mask_period(idx, "development")[0] and not split_mask_period(idx, "out_of_sample")[0]


def test_reports_run_end_to_end_and_print_prefixed_lines(capsys):
    daily = daily_dict(40, names=("US500", "USTECH", "US30", "GER40", "UK100", "EURUSD"), years=9)
    hourly = {n: rw_hourly(40 + k, years=9) for k, n in enumerate(daily)}
    cand_s23.run_tom(daily, {n: Hr(h) for n, h in hourly.items()})
    cand_s23.run_rip({n: hourly[n] for n in ("US500", "UK100", "EURUSD")})
    cand_s1.run_tsmom(daily)
    out = capsys.readouterr().out.strip().splitlines()
    assert out and all(line.startswith("CAND|") and line.count("|") >= 4 for line in out)
    assert sum("|verdict|" in line for line in out) == 3
    assert any("|cost_stress|" in line for line in out) and any("|min_stake|" in line for line in out)


def test_fix_scale_breaks_repairs_tenfold_stretch_and_leaves_real_moves():
    from research.cand_data import fix_scale_breaks
    h = rw_hourly(11, years=1)
    bad = h.copy()
    sl = slice(2000, 3000)
    for c in ("open", "high", "low", "close"):
        bad.iloc[sl, bad.columns.get_loc(c)] *= 10
    fixed, rep = fix_scale_breaks(bad)
    assert len(rep) == 2
    assert np.allclose(fixed[["open", "high", "low", "close"]].to_numpy(), h[["open", "high", "low", "close"]].to_numpy())
    same, rep2 = fix_scale_breaks(h)
    assert rep2 == [] and same is h
