"""Market-structure pullback: pure rules for research (no broker, no clock, no lookahead).

Swing pivots (k bars each side) are confirmed k bars late. A break of structure (BOS) starts a setup;
price must pull back into the zone between the broken level and a Fibonacci retracement of the
impulse leg, then print a confirmation candle. Entry is the next bar's open (done by the backtest).
Everything here works on completed OHLC bars and uses data up to the bar being evaluated only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scalp import _wilder


@dataclass(frozen=True)
class StructureRules:
    swing_k: int = 3
    atr_n: int = 14
    zone_fib: float = 0.618
    body_ratio_min: float = 0.5
    expiry_bars: int = 20
    stop_buffer_atr: float = 0.1
    min_stop_atr: float = 0.5
    max_stop_atr: float = 3.0
    target_r: float = 2.0
    breakeven_at_r: float | None = None
    htf_filter: bool = False
    long_only: bool = False
    htf_ema: int = 50
    too_deep_atr: float = 0.25


@dataclass(frozen=True)
class Entry:
    idx: int               # confirmation bar (completed); the entry fills at the next bar's open
    side: int              # +1 long, -1 short
    stop_level: float      # absolute stop price
    stop_dist: float       # confirmation close -> stop (reference; the backtest re-measures from the fill)
    target_dist: float
    atr: float
    bos_idx: int
    broken_level: float
    leg_low: float         # for shorts: the leg's high (the far end of the impulse)
    leg_high: float        # for shorts: the leg's low (the extreme reached)
    choch: bool


def _swing_arrays(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (sh, sl) indexed by CONFIRMATION bar: the pivot level confirmed at that bar, else NaN."""
    n = len(high)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    if n < 2 * k + 1:
        return sh, sl
    from numpy.lib.stride_tricks import sliding_window_view as win
    wh, wl = win(high, 2 * k + 1), win(low, 2 * k + 1)
    ch, cl = wh[:, k], wl[:, k]
    others_h = np.maximum(wh[:, :k].max(axis=1), wh[:, k + 1:].max(axis=1))
    others_l = np.minimum(wl[:, :k].min(axis=1), wl[:, k + 1:].min(axis=1))
    is_h, is_l = ch > others_h, cl < others_l
    # window row r has its pivot at bar r + k and is confirmed at bar r + 2k
    sh[2 * k:][is_h] = ch[is_h]
    sl[2 * k:][is_l] = cl[is_l]
    return sh, sl


def swings(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Confirmed swings: columns kind ('H'/'L'), pivot (bar index), confirm (pivot + k), level."""
    sh, sl = _swing_arrays(df.high.to_numpy(float), df.low.to_numpy(float), k)
    rows = [("H", int(c - k), int(c), float(sh[c])) for c in np.flatnonzero(~np.isnan(sh))]
    rows += [("L", int(c - k), int(c), float(sl[c])) for c in np.flatnonzero(~np.isnan(sl))]
    out = pd.DataFrame(rows, columns=["kind", "pivot", "confirm", "level"])
    return out.sort_values(["confirm", "kind"], kind="stable").reset_index(drop=True)


def atr_series(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    prev = df.close.shift()
    tr = np.maximum(df.high - df.low, np.maximum((df.high - prev).abs(), (df.low - prev).abs()))
    return _wilder(tr.to_numpy(float), n)


def htf_bias(index: pd.DatetimeIndex, h1: pd.DataFrame, ema_n: int = 50, bar: str = "15min") -> np.ndarray:
    """+1 / -1 / 0 per bar: last COMPLETED 1-hour close above / below its EMA(n). A bar starting at t
    ends at t + 15 min; the usable 1-hour bar is the latest one that ended at or before that time."""
    ema = h1.close.ewm(span=ema_n, adjust=False).mean().to_numpy(float).copy()
    ema[:ema_n] = np.nan
    cl = h1.close.to_numpy(float)
    starts = h1.index.as_unit("ns").asi8
    ends_needed = index.as_unit("ns").asi8 + pd.Timedelta(bar).value - pd.Timedelta("1h").value
    pos = np.searchsorted(starts, ends_needed, side="right") - 1      # latest 1h bar with start + 1h <= T
    ok = pos >= 0
    p = np.where(ok, pos, 0)
    diff = np.where(ok, cl[p] - ema[p], np.nan)
    return np.where(np.isnan(diff), 0, np.sign(diff)).astype(int)


def signals(df: pd.DataFrame, rules: StructureRules = StructureRules(), htf: pd.DataFrame | None = None,
            session: np.ndarray | None = None) -> list[Entry]:
    """Walk the bars once and return entries (at the confirmation bar). `htf` = 1-hour OHLC bars (required
    when rules.htf_filter). `session` = bool per bar, True if a confirmation on that bar is allowed; a
    blocked bar simply cannot confirm (the setup keeps waiting)."""
    r = rules
    n = len(df)
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = atr_series(df, r.atr_n)
    sh, sl = _swing_arrays(H, L, r.swing_k)
    bias = None
    if r.htf_filter:
        if htf is None:
            raise ValueError("htf_filter needs 1-hour bars")
        bias = htf_bias(df.index, htf, r.htf_ema)
    mir = {1: (O, H, L, C), -1: (-O, -L, -H, -C)}
    out: list[Entry] = []
    h1 = h2 = l1 = l2 = np.nan          # last two confirmed swing highs / lows (h2, l2 newer)
    last_trend, setup = 0, None
    for t in range(n):
        if sh[t] == sh[t]:
            h1, h2 = h2, sh[t]
        if sl[t] == sl[t]:
            l1, l2 = l2, sl[t]
        lab = 0
        if h1 == h1 and l1 == l1:
            lab = 1 if (h2 > h1 and l2 > l1) else (-1 if (h2 < h1 and l2 < l1) else 0)
        a = atr[t]
        if t == 0 or not np.isfinite(a) or a <= 0:
            if lab:
                last_trend = lab
            continue
        # ---- break of structure -> new setup
        new = None
        for s in (1, -1):
            if s == -1 and r.long_only:
                continue
            lvl = h2 if s == 1 else l2
            leg = l2 if s == 1 else h2
            if lvl != lvl or leg != leg or lab == -s:
                continue
            c0, c1 = s * C[t], s * C[t - 1]
            if c0 > s * lvl and c1 <= s * lvl and s * (lvl - leg) > 0:
                new = (s, lvl, leg)
                break
        if new and (setup is None or setup["s"] != new[0]):
            s, lvl, leg = new
            m = mir[s]
            setup = {"s": s, "bos": t, "lvl": s * lvl, "leg_low": s * leg, "leg_high": m[1][t],
                     "pull_low": m[2][t], "touched": False, "choch": last_trend == -s}
            if lab:
                last_trend = lab
            continue
        if setup is not None:
            s = setup["s"]
            Om, Hm, Lm, Cm = mir[s]
            if Hm[t] > setup["leg_high"]:
                setup["leg_high"], setup["pull_low"] = Hm[t], Lm[t]
            else:
                setup["pull_low"] = min(setup["pull_low"], Lm[t])
            top = setup["lvl"]
            fib = setup["leg_high"] - r.zone_fib * (setup["leg_high"] - setup["leg_low"])
            bottom = min(top, fib)
            if (Cm[t] < setup["leg_low"] or t - setup["bos"] > r.expiry_bars
                    or Lm[t] < fib - r.too_deep_atr * a):
                setup = None
            else:
                if Lm[t] <= top:
                    setup["touched"] = True
                ok = setup["touched"] and (session is None or session[t])
                if ok and bias is not None and bias[t] != s:
                    ok = False
                rng = Hm[t] - Lm[t]
                if (ok and Cm[t] > Om[t] and rng > 0 and (Cm[t] - Om[t]) / rng >= r.body_ratio_min
                        and Cm[t] > Hm[t - 1]
                        and (bottom <= Lm[t] <= top or bottom <= Lm[t - 1] <= top)):
                    stop_m = setup["pull_low"] - r.stop_buffer_atr * a
                    dist = Cm[t] - stop_m
                    if r.min_stop_atr * a <= dist <= r.max_stop_atr * a:
                        out.append(Entry(t, s, s * stop_m, float(dist), float(r.target_r * dist), float(a),
                                         setup["bos"], s * top, s * setup["leg_low"], s * setup["leg_high"],
                                         setup["choch"]))
                    setup = None          # consumed either way: a setup never enters twice
        if lab:
            last_trend = lab
    return out
