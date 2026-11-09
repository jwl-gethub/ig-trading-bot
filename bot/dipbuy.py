"""Daily RSI(2) dip strategy ("quick exit"), shared by the research backtest and the live bot.

Rules (Connors & Alvarez, published ~2008; validated in research/ before use):
- Each market has a fixed daily close: the hourly bar that ends at `close_hour` local time.
- Long: close above its 200-day average and RSI(2) below 5.
  Short (only where allowed, e.g. FX/commodities): close below the 200-day average and RSI(2) above 95.
- Exit: the first close back beyond the 5-day average (above it for longs, below for shorts).
- Orders decided on a close are filled at the next weekday's `fill_hour` local time, a liquid hour.
- Server-side stop 2.5 x ATR(14) from the fill. The stop is static; the exit rule does the rest.
Everything here is pure computation: no broker calls, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DipSchedule:
    close_tz: str
    close_hour: int
    fill_tz: str
    fill_hour: int


@dataclass(frozen=True)
class DipRules:
    rsi_n: int = 2
    rsi_long: float = 5.0
    rsi_short: float = 95.0
    trend_sma: int = 200
    exit_sma: int = 5
    atr_n: int = 14
    stop_atr: float = 2.5
    long_only: bool = True


def daily_bars(hourly: pd.DataFrame, s: DipSchedule) -> pd.DataFrame:
    """Hourly bars (UTC index = bar start) -> one row per trading day, indexed by the close time (UTC).
    A day runs from the previous close to this close. Days without the closing bar are dropped."""
    if hourly.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    end = (hourly.index + pd.Timedelta(hours=1)).tz_convert(s.close_tz)
    after_close = (end.hour * 60 + end.minute) > s.close_hour * 60
    label = end.normalize().tz_localize(None) + pd.to_timedelta(after_close.astype(int), unit="D")
    is_close = (end.hour == s.close_hour) & (end.minute == 0)
    df = hourly.assign(label=label, is_close=is_close)
    g = df.groupby("label")
    out = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min()})
    closes = df[df.is_close].groupby("label").close.last()
    out["close"] = closes
    out = out.dropna(subset=["close"])
    out = out[out.index.dayofweek < 5]
    idx = (out.index + pd.Timedelta(hours=s.close_hour)).tz_localize(s.close_tz, ambiguous="NaT",
                                                                        nonexistent="shift_forward")
    out.index = idx.tz_convert("UTC")
    return out[out.index.notna()]


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def indicators(d: pd.DataFrame, r: DipRules) -> pd.DataFrame:
    """Row t uses closes up to and including t only."""
    c = d.close.to_numpy(dtype=float)
    chg = np.r_[np.nan, np.diff(c)]
    ag = _wilder(np.where(chg > 0, chg, 0.0), r.rsi_n)
    al = _wilder(np.where(chg < 0, -chg, 0.0), r.rsi_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(al == 0, 100.0, 100 - 100 / (1 + ag / al))
    prev = d.close.shift()
    tr = np.maximum(d.high - d.low, np.maximum((d.high - prev).abs(), (d.low - prev).abs())).to_numpy(dtype=float)
    atr = _wilder(tr, r.atr_n)
    out = d.copy()
    out["sma_trend"] = d.close.rolling(r.trend_sma).mean()
    out["sma_exit"] = d.close.rolling(r.exit_sma).mean()
    out["rsi"] = rsi
    out["atr"] = atr
    out["atr_pct"] = atr / c
    return out


def decide(row, pos_dir: int, r: DipRules) -> str | None:
    """'long', 'short', 'exit' or None for one completed daily close."""
    vals = (row.close, row.sma_trend, row.sma_exit, row.rsi, row.atr)
    if not all(np.isfinite(v) for v in vals):
        return None
    if pos_dir > 0:
        return "exit" if row.close > row.sma_exit else None
    if pos_dir < 0:
        return "exit" if row.close < row.sma_exit else None
    if row.close > row.sma_trend and row.rsi < r.rsi_long:
        return "long"
    if not r.long_only and row.close < row.sma_trend and row.rsi > r.rsi_short:
        return "short"
    return None


def fill_time(close_ts: pd.Timestamp, s: DipSchedule) -> pd.Timestamp:
    """The next weekday at fill_hour (fill_tz) strictly after the close."""
    local = close_ts.tz_convert(s.fill_tz)
    day = local.normalize()
    cand = day + pd.Timedelta(hours=s.fill_hour)
    if cand <= local:
        day = day + pd.Timedelta(days=1)
    while day.dayofweek >= 5:
        day = day + pd.Timedelta(days=1)
    naive = day.tz_localize(None) + pd.Timedelta(hours=s.fill_hour)
    return naive.tz_localize(s.fill_tz, nonexistent="shift_forward").tz_convert("UTC")


def stop_distance_pct(row, r: DipRules) -> float:
    """Stop distance as a fraction of price, so it converts to any broker's price units."""
    return float(r.stop_atr * row.atr_pct)
