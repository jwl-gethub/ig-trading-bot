from bot.strategy import PositionState, Strategy

MKT = {"name": "TEST", "modules": ["trend", "meanrev"], "allow_short": False}


def trend_row(**over):
    r = dict(ready=True, vol_ok=True, regime="TREND", close=110.0, open=108.0, high=111.0, low=107.0,
             ema_fast=108.0, ema_mid=105.0, ema_slow=100.0, plus_di=30.0, minus_di=15.0, macd_hist=0.5,
             rsi=60.0, rsi_fast=60.0, dc_high=109.0, dc_low=95.0, prev_high=109.5, prev_low=106.0,
             touched_fast_long=False, touched_fast_short=False, atr=2.0, adx=30.0, atr_pctile=50.0,
             bb_lower=100.0, bb_mid=105.0, bb_upper=110.0)
    r.update(over)
    return r


def test_trend_long_breakout(cfg):
    s = Strategy(cfg["strategy"]).entry_signal(MKT, trend_row())
    assert s and s.direction == 1 and s.module == "trend"
    assert s.stop_distance == cfg["strategy"]["trend"]["stop_atr"] * 2.0
    assert s.conditions["breakout"]


def test_no_trigger_no_trade(cfg):
    assert Strategy(cfg["strategy"]).entry_signal(MKT, trend_row(dc_high=120.0)) is None


def test_bias_mandatory(cfg):
    assert Strategy(cfg["strategy"]).entry_signal(MKT, trend_row(ema_slow=115.0)) is None


def test_filters_block(cfg):
    st = Strategy(cfg["strategy"])
    assert st.entry_signal(MKT, trend_row(vol_ok=False)) is None
    assert st.entry_signal(MKT, trend_row(ready=False)) is None
    assert st.entry_signal(MKT, trend_row(regime="NEUTRAL")) is None


def test_short_blocked_when_not_allowed(cfg):
    bear = trend_row(close=90.0, ema_fast=92.0, ema_mid=95.0, ema_slow=100.0, plus_di=10.0, minus_di=30.0,
                     macd_hist=-0.5, rsi=40.0, dc_low=91.0, dc_high=110.0)
    st = Strategy(cfg["strategy"])
    assert st.entry_signal(MKT, bear) is None
    s = st.entry_signal({**MKT, "allow_short": True}, bear)
    assert s and s.direction == -1


def test_meanrev_long(cfg):
    row = trend_row(regime="RANGE", close=99.0, ema_slow=95.0, bb_lower=100.0, rsi_fast=5.0, rsi=35.0)
    s = Strategy(cfg["strategy"]).entry_signal(MKT, row)
    assert s and s.module == "meanrev" and s.direction == 1


def _pos(module="trend", stop=95.0):
    return PositionState(market="TEST", direction=1, module=module, entry_price=100.0, stop_level=stop,
                         size=1.0, point_value=1.0, initial_risk_points=5.0, entry_ts="t", extreme=100.0)


def test_trailing_stop_only_ratchets_up(cfg):
    st = Strategy(cfg["strategy"])
    p = _pos()
    d1 = st.manage(p, trend_row(high=120.0, close=119.0, atr=2.0))
    assert d1.new_stop == 120.0 - 3 * 2.0
    p.stop_level = d1.new_stop
    d2 = st.manage(p, trend_row(high=115.0, close=116.0, atr=2.0))   # lower extreme stays 120
    assert d2.new_stop is None and not d2.close


def test_trend_failure_exit(cfg):
    d = Strategy(cfg["strategy"]).manage(_pos(), trend_row(ema_fast=100.0, ema_mid=101.0))
    assert d.close and d.reason == "trend_failure"


def test_meanrev_time_stop(cfg):
    st = Strategy(cfg["strategy"])
    p = _pos("meanrev")
    row = trend_row(close=98.0, bb_mid=105.0, rsi_fast=20.0)
    for _ in range(cfg["strategy"]["meanrev"]["max_bars"] - 1):
        assert not st.manage(p, row).close
    d = st.manage(p, row)
    assert d.close and d.reason == "time_stop"


def test_excursions_tracked(cfg):
    st = Strategy(cfg["strategy"])
    p = _pos()
    st.manage(p, trend_row(high=104.0, low=97.0, close=103.0))
    st.manage(p, trend_row(high=102.0, low=99.0, close=101.0))
    assert p.mfe_pts == 4.0 and p.mae_pts == 3.0


# ---------------------------------------------------------------- sweep & reclaim
def sweep_row(**over):
    # prev candle 104-108; this candle dips to 102 (below 104) and closes 109 (above 108)
    r = trend_row(regime="NEUTRAL", open=105.0, high=109.5, low=102.0, close=109.0,
                  prev_low=104.0, prev_high=108.0, atr=4.0, ema_fast=106.0, ema_mid=104.0, ema_slow=100.0)
    r.update(over)
    return r


def test_sweep_long_fires_with_stop_under_wick_and_2r_target(cfg):
    s = Strategy(cfg["strategy"]).entry_signal({**MKT, "modules": ["sweep"]}, sweep_row())
    assert s and s.module == "sweep" and s.direction == 1
    assert s.stop_distance == 109.0 - (102.0 - 0.1 * 4.0)              # 7.4
    assert s.target_distance == cfg["strategy"]["sweep"]["target_r"] * s.stop_distance


def test_sweep_needs_reclaim_trend_and_sane_stop(cfg):
    st = Strategy(cfg["strategy"])
    mk = {**MKT, "modules": ["sweep"]}
    assert st.entry_signal(mk, sweep_row(close=107.5)) is None          # didn't close above prev high
    assert st.entry_signal(mk, sweep_row(ema_slow=112.0)) is None       # no uptrend context
    assert st.entry_signal(mk, sweep_row(low=95.0)) is None             # stop wider than 2.5 ATR
    assert st.entry_signal(mk, sweep_row(low=103.9)) is None            # sweep too shallow (<0.1 ATR)


def test_sweep_time_stop(cfg):
    st = Strategy(cfg["strategy"])
    p = _pos("sweep")
    for _ in range(cfg["strategy"]["sweep"]["max_bars"] - 1):
        assert not st.manage(p, trend_row()).close
    assert st.manage(p, trend_row()).reason == "time_stop"
