"""Daily RSI(2) dip-buying in an uptrend (Connors & Alvarez, published ~2008), long only.

Entry: close above its 200-day simple average and RSI(2) below a threshold (5).
Exit A ("quick"): first close above the 5-day simple average (take the rebound).
Exit B ("ride"): chandelier trail = highest close since entry - 3 x ATR(14), checked on closes.
Protection: a server-side stop 2.5 x ATR(14) below the fill, hit intraday (fills at the worse of
open and stop). Fills either at the next session's open, or at the close using a signal computed
from the 15:50 price ("near_close", what a bot can actually do). Costs: IG spread + slippage per
fill in index points, and overnight funding on the position for every calendar night held.
Positions are marked to market daily. Unlevered (1x the account) so it compares with an ISA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DipParams:
    rsi_n: int = 2
    rsi_entry: float = 5.0
    trend_sma: int = 200
    exit_sma: int = 5
    exit: str = "quick"                  # "quick" or "ride"
    trail_atr: float = 3.0
    stop_atr: float | None = 2.5         # None = no server-side stop (Connors' original)
    atr_n: int = 14
    fill: str = "next_open"              # "next_open" or "near_close"
    pre_close_minute: int = 380          # 15:50 for a 09:30-16:00 session
    cost_side_points: float = 0.3
    funding_annual: float = 0.065        # benchmark rate + IG's 2.5% admin fee, longs pay
    leverage: float = 1.0


def daily_bars(sm: dict, pre_close_minute: int) -> pd.DataFrame:
    return pd.DataFrame({"open": sm["open"][:, 0], "high": np.nanmax(sm["high"], axis=1),
                         "low": np.nanmin(sm["low"], axis=1), "close": sm["close"][:, -1],
                         "pre": sm["close"][:, pre_close_minute]}, index=sm["days"])


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def signals(d: pd.DataFrame, p: DipParams) -> pd.DataFrame:
    """Indicators on closes. For near_close fills, today's close is replaced by the 15:50 price,
    so a decision at 15:50 uses nothing that happens after 15:50."""
    c = d.close.to_numpy()
    chg = np.r_[np.nan, np.diff(c)]
    ag, al = _wilder(np.where(chg > 0, chg, 0.0), p.rsi_n), _wilder(np.where(chg < 0, -chg, 0.0), p.rsi_n)
    tr = np.maximum(d.high - d.low, np.maximum(abs(d.high - d.close.shift()), abs(d.low - d.close.shift()))).to_numpy()
    atr = pd.Series(_wilder(tr, p.atr_n), index=d.index)
    out = pd.DataFrame(index=d.index)
    if p.fill == "near_close":
        x = d.pre.to_numpy()                                   # today's price at 15:50
        g = x - np.r_[np.nan, c[:-1]]
        ag_t = (np.r_[np.nan, ag[:-1]] * (p.rsi_n - 1) + np.where(g > 0, g, 0)) / p.rsi_n
        al_t = (np.r_[np.nan, al[:-1]] * (p.rsi_n - 1) + np.where(g < 0, -g, 0)) / p.rsi_n
        prev = pd.Series(c, index=d.index).shift()
        roll = lambda n: (prev.rolling(n - 1).sum() + x) / n   # noqa: E731
        out["px"], out["sma_trend"], out["sma_exit"] = x, roll(p.trend_sma), roll(p.exit_sma)
        out["atr"] = atr.shift()                               # yesterday's ATR (today's not known)
    else:
        ag_t, al_t = ag, al
        out["px"] = c
        out["sma_trend"] = d.close.rolling(p.trend_sma).mean()
        out["sma_exit"] = d.close.rolling(p.exit_sma).mean()
        out["atr"] = atr
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rsi"] = np.where(al_t == 0, 100.0, 100 - 100 / (1 + ag_t / al_t))
    return out


def backtest_dip(sm: dict, p: DipParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = daily_bars(sm, p.pre_close_minute)
    s = signals(d, p)
    o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    days = d.index
    cost, lev = p.cost_side_points, p.leverage
    trades, daily = [], []
    pos, entry, mark, stop, peak, t_in, pending = False, 0.0, 0.0, None, 0.0, None, None
    for i in range(len(d)):
        ret = 0.0
        nights = (days[i] - days[i - 1]).days if i else 0

        def close_trade(px: float, why: str) -> None:
            nonlocal pos, ret
            fill = px - cost
            ret += lev * (fill / mark - 1)
            fin = lev * p.funding_annual / 365 * (days[i] - t_in).days
            trades.append({"day": days[i], "dir": 1, "entry": entry, "exit": fill, "days_held": (days[i] - t_in).days,
                           "ret": fill / entry - 1, "acct_ret": lev * (fill / entry - 1) - fin, "lev": lev, "reason": why})
            pos = False

        if pos:
            ret -= lev * p.funding_annual / 365 * nights        # overnight funding
        # 1) orders decided at yesterday's close, filled at today's open
        if p.fill == "next_open" and pending:
            if pending == "exit" and pos:
                close_trade(o[i], "exit")
            elif pending == "enter" and not pos:
                pos, entry, t_in = True, o[i] + cost, days[i]
                mark, peak = entry, entry
                stop = entry - p.stop_atr * s.atr.iloc[i - 1] if p.stop_atr else None
            pending = None
        # 2) server-side stop during the session
        if pos and stop is not None and lo[i] <= stop:
            close_trade(min(o[i], stop), "stop")
        # 3) mark open position to the close
        if pos:
            ret += lev * (c[i] / mark - 1)
            mark = c[i]
            peak = max(peak, c[i])
        # 4) decisions on today's close (or at 15:50 for near_close)
        px, atr = s.px.iloc[i], s.atr.iloc[i]
        if not np.isfinite(s.sma_trend.iloc[i]) or not np.isfinite(atr):
            daily.append({"day": days[i], "ret": ret, "lev": lev if pos else 0.0})
            continue
        if pos:
            if p.exit == "quick":
                go_out = px > s.sma_exit.iloc[i]
            else:
                trail = peak - p.trail_atr * atr
                stop = max(stop, trail) if stop is not None else trail
                go_out = False                                   # the trail is a resting stop
            if go_out:
                if p.fill == "near_close":
                    close_trade(c[i], "exit")                    # already marked to c[i]; pays the spread
                else:
                    pending = "exit"
        elif px > s.sma_trend.iloc[i] and s.rsi.iloc[i] < p.rsi_entry:
            if p.fill == "near_close":
                pos, entry, t_in = True, c[i] + cost, days[i]
                mark, peak = entry, entry
                ret += lev * (c[i] / entry - 1)                 # pays the spread today
                mark = c[i]
                stop = entry - p.stop_atr * atr if p.stop_atr else None
            else:
                pending = "enter"
        daily.append({"day": days[i], "ret": ret, "lev": lev if pos else 0.0})
    return pd.DataFrame(trades), pd.DataFrame(daily)
