"""Indicator library.

Every function is causal: the value at row t uses only rows <= t.
tests/test_indicators.py enforces this (no lookahead).

Indicators are grouped so each one has a distinct job — piling on more
indicators that measure the same thing (RSI, Stochastic, CCI, Williams %R
are all momentum) adds parameters to overfit, not information.

  Trend:       EMA 20/50/200, ADX + DI
  Momentum:    MACD histogram, RSI(14)
  Volatility:  ATR(14), ATR percentile, Bollinger Bands
  Structure:   Donchian channel (breakouts)
  Mean-rev:    RSI(2)
Volume is deliberately excluded: IG CFD/spread-bet volume is tick volume
and not a reliable signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    out = out.mask((dn == 0) & (up > 0), 100.0)
    out = out.mask((dn == 0) & (up == 0), 50.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return wilder(dx, n), plus_di, minus_di


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int, k: float):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def donchian_prev(df: pd.DataFrame, n: int):
    """Channel of the n bars BEFORE the current bar, so a breakout = close beyond it."""
    hi = df["high"].rolling(n, min_periods=n).max().shift(1)
    lo = df["low"].rolling(n, min_periods=n).min().shift(1)
    return hi, lo


def rolling_percentile(s: pd.Series, n: int) -> pd.Series:
    """Percentile rank (0-100) of the current value within the trailing n values."""

    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        ranks = 100.0 * (w <= w[:, -1:]).mean(axis=1)
        ranks[np.isnan(w).any(axis=1)] = np.nan
        out[n - 1:] = ranks
    return pd.Series(out, index=s.index)


REQUIRED_COLS = [
    "ema_fast", "ema_mid", "ema_slow", "atr", "adx", "plus_di", "minus_di", "macd_hist",
    "rsi", "rsi_fast", "bb_lower", "bb_mid", "bb_upper", "dc_high", "dc_low", "atr_pctile",
    "prev_high", "prev_low",
]


def compute_all(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Add every indicator column used by the strategy. `p` = cfg['strategy']."""
    t, m, r, v = p["trend"], p["meanrev"], p["regime"], p["volatility"]
    base = df[["open", "high", "low", "close"]].astype(float)
    c = base["close"]
    col: dict[str, pd.Series] = {k: base[k] for k in ("open", "high", "low", "close")}

    col["ema_fast"] = ema(c, t["ema_fast"])
    col["ema_mid"] = ema(c, t["ema_mid"])
    col["ema_slow"] = ema(c, t["ema_slow"])
    col["atr"] = atr(base, v["atr_len"])
    col["adx"], col["plus_di"], col["minus_di"] = adx(base, r["adx_len"])
    _, _, col["macd_hist"] = macd(c, t["macd_fast"], t["macd_slow"], t["macd_signal"])
    col["rsi"] = rsi(c, t["rsi_len"])
    col["rsi_fast"] = rsi(c, m["rsi_fast_len"])
    col["bb_lower"], col["bb_mid"], col["bb_upper"] = bollinger(c, m["bb_len"], m["bb_std"])
    col["dc_high"], col["dc_low"] = donchian_prev(base, t["donchian_len"])
    col["atr_pctile"] = rolling_percentile(col["atr"] / c, v["lookback"])
    col["prev_high"] = base["high"].shift(1)
    col["prev_low"] = base["low"].shift(1)

    k = t["pullback_lookback"]
    col["touched_fast_long"] = (
        (base["low"] <= col["ema_fast"]).astype(float).rolling(k, min_periods=1).max() > 0
    )
    col["touched_fast_short"] = (
        (base["high"] >= col["ema_fast"]).astype(float).rolling(k, min_periods=1).max() > 0
    )

    out = pd.DataFrame(col, index=base.index)
    adx_v = out["adx"]
    regime = np.select(
        [adx_v >= r["adx_trend"], adx_v <= r["adx_range"]], ["TREND", "RANGE"], default="NEUTRAL"
    )
    vol_ok = (out["atr_pctile"] >= v["low_pctile"]) & (out["atr_pctile"] <= v["high_pctile"])
    ready = out[REQUIRED_COLS].notna().all(axis=1)
    return out.assign(regime=regime, vol_ok=vol_ok, ready=ready)
