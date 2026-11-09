"""Intraday momentum "noise area" strategy, published rules, with IG-style costs.

Source: Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective Intraday Momentum
Strategy for S&P500 ETF (SPY)".
- Noise area: for each minute of the session, sigma = average absolute move from the day's
  open at that minute over the previous 14 days. Upper = max(open, prev close) x (1 + sigma),
  lower = min(open, prev close) x (1 - sigma).
- Checked every 30 minutes (HH:00 / HH:30) from 30 minutes after the open. Above the upper
  bound (and above VWAP) -> long; below the lower bound (and below VWAP) -> short.
- Trailing exit at the checks: long exits below max(upper, VWAP), short above min(lower, VWAP).
- Flat at the close. Size = equity x min(4, 2% / 14-day daily volatility).

Research only: nothing here can place an order. Signals use data up to the decision minute;
fills happen at the next minute's open, costs are charged in index points on every fill.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSIONS = {                        # local exchange cash session
    "US500": ("America/New_York", "09:30", "16:00"),
    "USTECH": ("America/New_York", "09:30", "16:00"),
    "GER40": ("Europe/Berlin", "09:00", "17:30"),
    "UK100": ("Europe/London", "08:00", "16:30"),
}


@dataclass
class Params:
    lookback_days: int = 14
    min_days: int = 10
    first_check_min: int = 30
    check_every_min: int = 30
    target_daily_vol: float = 0.02
    max_leverage: float = 4.0
    vol_target: bool = True
    ref: str = "vwap"                       # "vwap", "twap" or "none"
    cost_side_points: float = 0.3           # half spread + slippage per fill, index points
    hard_stop_sigma: float | None = None    # server-side stop: k x 14-day daily sigma from entry


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def session_matrix(df: pd.DataFrame, market: str, min_coverage: float = 0.9) -> dict:
    """1-min UTC bars -> per-day arrays (days x session minutes) in exchange local time."""
    tz, start, end = SESSIONS[market]
    s, e = _minutes(start), _minutes(end)
    loc = df.tz_convert(tz)
    mod = loc.index.hour * 60 + loc.index.minute
    loc = loc[(mod >= s) & (mod < e)]
    m = (loc.index.hour * 60 + loc.index.minute - s).to_numpy()
    d = loc.index.normalize().tz_localize(None)
    days = pd.Index(sorted(d.unique()))
    di = days.get_indexer(d)
    M = e - s
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        a = np.full((len(days), M), np.nan)
        a[di, m] = loc[col].to_numpy(dtype=float)
        out[col] = a
    keep = np.isfinite(out["close"]).mean(axis=1) >= min_coverage
    for col in out:
        out[col] = out[col][keep]
    days = days[keep]
    c = pd.DataFrame(out["close"]).ffill(axis=1).bfill(axis=1).to_numpy()
    for col in ("open", "high", "low"):
        a = out[col]
        out[col] = np.where(np.isfinite(a), a, c)
    out["close"] = c
    out["volume"] = np.nan_to_num(out["volume"])
    out["days"] = days
    return out


def indicators(sm: dict, p: Params) -> dict:
    """Everything a decision at (day, minute) may use. Rows only look backwards in time."""
    o, h, l, c, v = (sm[k] for k in ("open", "high", "low", "close", "volume"))
    D, M = c.shape
    day_open = o[:, 0]
    day_close = c[:, -1]
    prev_close = np.r_[np.nan, day_close[:-1]]
    move = np.abs(c / day_open[:, None] - 1)
    sigma = np.full((D, M), np.nan)
    for i in range(D):
        lo = max(0, i - p.lookback_days)
        if i - lo >= p.min_days:
            sigma[i] = np.nanmean(move[lo:i], axis=0)          # previous days only
    upper = np.fmax(day_open, prev_close)[:, None] * (1 + sigma)
    lower = np.fmin(day_open, prev_close)[:, None] * (1 - sigma)
    tp = (h + l + c) / 3
    cv = np.cumsum(v, axis=1)
    twap = np.cumsum(c, axis=1) / np.arange(1, M + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cv > 0, np.cumsum(tp * v, axis=1) / cv, twap)
    ret = day_close / prev_close - 1
    dvol = pd.Series(ret).rolling(p.lookback_days, min_periods=p.min_days).std().shift(1).to_numpy()
    if p.vol_target:
        lev = np.minimum(p.max_leverage, p.target_daily_vol / dvol)
    else:
        lev = np.where(np.isfinite(dvol), 1.0, np.nan)
    ref = {"vwap": vwap, "twap": twap, "none": None}[p.ref]
    return {"upper": upper, "lower": lower, "ref": ref, "lev": lev, "dvol": dvol,
            "day_close": day_close, "prev_close": prev_close}


def backtest(sm: dict, p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trades, daily). Daily 'ret' is the account return that day (leverage applied)."""
    ind = indicators(sm, p)
    o, h, l, c = sm["open"], sm["high"], sm["low"], sm["close"]
    D, M = c.shape
    checks = set(range(p.first_check_min, M - 1, p.check_every_min))
    cost = p.cost_side_points
    trades, daily = [], []
    for i in range(D):
        lev = ind["lev"][i]
        up, dn = ind["upper"][i], ind["lower"][i]
        if not np.isfinite(lev) or not np.isfinite(up[0]):
            continue
        ref = ind["ref"][i] if ind["ref"] is not None else None
        pos, entry, stop, t_in = 0, 0.0, None, 0
        day_ret = 0.0

        def close_at(px: float, t: int, why: str) -> None:
            nonlocal pos, day_ret
            fill = px - pos * cost
            r = pos * (fill - entry) / entry
            day_ret += lev * r
            trades.append({"day": sm["days"][i], "dir": pos, "t_in": t_in, "t_out": t, "entry": entry,
                           "exit": fill, "ret": r, "acct_ret": lev * r, "lev": lev, "reason": why})
            pos = 0

        for t in range(M):
            if pos and stop is not None and t >= t_in:             # server-side stop, intrabar
                if pos == 1 and l[i, t] <= stop:
                    close_at(min(o[i, t], stop), t, "hard_stop")
                elif pos == -1 and h[i, t] >= stop:
                    close_at(max(o[i, t], stop), t, "hard_stop")
            if t in checks:
                px = c[i, t]
                r_t = ref[t] if ref is not None else None
                if pos == 1 and px < (max(up[t], r_t) if r_t is not None else up[t]):
                    close_at(o[i, t + 1], t + 1, "trail")
                elif pos == -1 and px > (min(dn[t], r_t) if r_t is not None else dn[t]):
                    close_at(o[i, t + 1], t + 1, "trail")
                if pos == 0:
                    go = 0
                    if px > up[t] and (r_t is None or px > r_t):
                        go = 1
                    elif px < dn[t] and (r_t is None or px < r_t):
                        go = -1
                    if go:
                        pos, t_in = go, t + 1
                        entry = o[i, t + 1] + go * cost
                        stop = None
                        if p.hard_stop_sigma:
                            stop = o[i, t + 1] - go * p.hard_stop_sigma * ind["dvol"][i] * o[i, t + 1]
        if pos:
            close_at(c[i, M - 1], M - 1, "close")
        daily.append({"day": sm["days"][i], "ret": day_ret, "lev": lev})
    return pd.DataFrame(trades), pd.DataFrame(daily)


def metrics(trades: pd.DataFrame, daily: pd.DataFrame, start=None, end=None) -> dict:
    d = daily.copy()
    t = trades.copy()
    if start is not None:
        d, t = d[d.day >= pd.Timestamp(start)], t[t.day >= pd.Timestamp(start)] if len(t) else t
    if end is not None:
        d, t = d[d.day < pd.Timestamp(end)], t[t.day < pd.Timestamp(end)] if len(t) else t
    if not len(d):
        return {"days": 0}
    eq = (1 + d.ret).cumprod()
    years = len(d) / 252
    dd = float((eq / eq.cummax() - 1).min())
    sd = d.ret.std()
    out = {
        "from": str(d.day.iloc[0].date()), "to": str(d.day.iloc[-1].date()), "days": int(len(d)),
        "total_return_pct": round(100 * (eq.iloc[-1] - 1), 1),
        "annual_return_pct": round(100 * (eq.iloc[-1] ** (1 / years) - 1), 1) if years > 0.2 else None,
        "annual_vol_pct": round(100 * sd * np.sqrt(252), 1),
        "sharpe": round(float(d.ret.mean() / sd * np.sqrt(252)), 2) if sd > 0 else None,
        "max_drawdown_pct": round(-100 * dd, 1),
        "days_traded_pct": round(100 * float((d.ret != 0).mean()), 1),
        "positive_months_pct": round(100 * float((d.groupby(d.day.dt.to_period("M")).ret
                                                   .apply(lambda x: (1 + x).prod() - 1) > 0).mean()), 1),
        "worst_day_pct": round(100 * float(d.ret.min()), 2),
        "best_day_pct": round(100 * float(d.ret.max()), 2),
        "avg_leverage": round(float(d.lev.mean()), 2),
    }
    if len(t):
        a = t.acct_ret
        wins, losses = a[a > 0], a[a <= 0]
        out.update({
            "trades": int(len(t)),
            "win_rate_pct": round(100 * len(wins) / len(t), 1),
            "avg_win_pct": round(100 * float(wins.mean()), 3) if len(wins) else None,
            "avg_loss_pct": round(100 * float(losses.mean()), 3) if len(losses) else None,
            "payoff_ratio": round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) else None,
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
            "exit_reasons": t.reason.value_counts().to_dict(),
            "long_share_pct": round(100 * float((t.dir == 1).mean()), 1),
        })
    out["by_year_pct"] = {str(y): round(100 * float((1 + g.ret).prod() - 1), 1)
                          for y, g in d.groupby(d.day.dt.year)}
    return out


def buy_and_hold(sm: dict, start=None, end=None) -> float | None:
    days = sm["days"]
    mask = np.ones(len(days), bool)
    if start is not None:
        mask &= days >= pd.Timestamp(start)
    if end is not None:
        mask &= days < pd.Timestamp(end)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None
    return round(100 * float(sm["close"][idx[-1], -1] / sm["close"][idx[0], 0] - 1), 1)


def last_half_hour(sm: dict, p: Params, signal: str = "rest_of_day", minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market intraday momentum (Gao et al. 2018; Baltussen et al. 2021): trade the final half hour
    in the direction of the day so far. signal='rest_of_day': previous close -> 30 min before the
    close; 'first_half_hour': previous close -> 30 min after the open. Flat at the close."""
    ind = indicators(sm, p)
    o, c = sm["open"], sm["close"]
    D, M = c.shape
    t_in = M - minutes
    trades, daily = [], []
    for i in range(D):
        lev, pc = ind["lev"][i], ind["prev_close"][i]
        if not (np.isfinite(lev) and np.isfinite(pc)):
            continue
        ref_px = c[i, t_in - 1] if signal == "rest_of_day" else c[i, 29]
        go = int(np.sign(ref_px / pc - 1))
        r = 0.0
        if go:
            entry = o[i, t_in] + go * p.cost_side_points
            exit_ = c[i, M - 1] - go * p.cost_side_points
            r = go * (exit_ - entry) / entry
            trades.append({"day": sm["days"][i], "dir": go, "t_in": t_in, "t_out": M - 1, "entry": entry,
                           "exit": exit_, "ret": r, "acct_ret": lev * r, "lev": lev, "reason": "close"})
        daily.append({"day": sm["days"][i], "ret": lev * r, "lev": lev})
    return pd.DataFrame(trades), pd.DataFrame(daily)
