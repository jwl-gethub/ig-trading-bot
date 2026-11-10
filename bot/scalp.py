"""15-minute pullback scalper: pure rules shared by the research backtest and the live bot.

- Trend: close above EMA(200) and EMA(20) above EMA(50) -> longs only; the mirror image -> shorts only.
- Entry: RSI(2) below 10 in an uptrend (above 90 in a downtrend), decided on a closed 15-minute bar.
- Stop: 1.0 x ATR(14) from the fill. Take-profit: 1.5 x ATR(14). Both rest on the broker's side.
- Time stop: closed at market after 8 bars (2 hours).
- Only inside each market's liquid session (UTC hours), so the spread is small next to the stop.
No broker calls and no clock in here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalpRules:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_n: int = 2
    rsi_long: float = 10.0
    rsi_short: float = 90.0
    atr_n: int = 14
    stop_atr: float = 1.0
    target_atr: float = 1.5
    max_bars: int = 8
    long_only: bool = False


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def to_bars(prices: pd.Series, minutes: int = 15) -> pd.DataFrame:
    """Price samples (UTC index) -> OHLC bars labelled by their start. The last bar may be forming."""
    if prices.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    return prices.resample(f"{minutes}min").ohlc().dropna()


def indicators(d: pd.DataFrame, r: ScalpRules) -> pd.DataFrame:
    """Row t uses bars up to and including t only."""
    c = d.close.to_numpy(dtype=float)
    chg = np.r_[np.nan, np.diff(c)]
    ag = _wilder(np.where(chg > 0, chg, 0.0), r.rsi_n)
    al = _wilder(np.where(chg < 0, -chg, 0.0), r.rsi_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(al == 0, 100.0, 100 - 100 / (1 + ag / al))
    prev = d.close.shift()
    tr = np.maximum(d.high - d.low, np.maximum((d.high - prev).abs(), (d.low - prev).abs())).to_numpy(dtype=float)
    out = d.copy()
    for name, n in (("ema_fast", r.ema_fast), ("ema_slow", r.ema_slow), ("ema_trend", r.ema_trend)):
        e = d.close.ewm(span=n, adjust=False).mean()
        e.iloc[:n] = np.nan
        out[name] = e
    out["rsi"] = rsi
    out["atr"] = _wilder(tr, r.atr_n)
    return out


def decide(row, r: ScalpRules) -> str | None:
    vals = (row.close, row.ema_fast, row.ema_slow, row.ema_trend, row.rsi, row.atr)
    if not all(np.isfinite(v) for v in vals) or row.atr <= 0:
        return None
    if row.close > row.ema_trend and row.ema_fast > row.ema_slow and row.rsi < r.rsi_long:
        return "long"
    if not r.long_only and row.close < row.ema_trend and row.ema_fast < row.ema_slow and row.rsi > r.rsi_short:
        return "short"
    return None


def in_session(ts: pd.Timestamp, hours: tuple[int, int]) -> bool:
    """UTC hours [start, end); weekdays only. The entry bar must close inside the session."""
    return ts.dayofweek < 5 and hours[0] <= ts.hour < hours[1]
