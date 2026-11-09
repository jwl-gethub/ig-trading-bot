"""Historical data: CSV loader + synthetic generator (synthetic = code smoke tests ONLY).

CSV format (one file per market, named <MARKET_NAME>.csv, e.g. EURUSD.csv):
    timestamp,open,high,low,close
    2024-01-02 00:00:00,1.10412,1.10530,1.10201,1.10288
Timestamps are bar OPEN times in UTC. Prices are mid. `price_scale` in config
converts them to IG quote units (FX spread bets: x10000).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path, price_scale: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = next(c for c in df.columns if c in ("timestamp", "time", "date", "datetime"))
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = df[["open", "high", "low", "close"]].astype(float) * float(price_scale)
    return out.dropna()


def load_dir(directory: str | Path, markets: list[dict]) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    data = {}
    for m in markets:
        f = directory / f"{m['name']}.csv"
        if f.exists():
            data[m["name"]] = load_csv(f, m.get("price_scale", 1.0))
    if not data:
        raise FileNotFoundError(f"no <MARKET>.csv files found in {directory}")
    return data


def synthetic_market(seed: int, n_bars: int = 3000, start_price: float = 100.0,
                     bar_vol: float = 0.003, start: str = "2023-01-02",
                     bar_hours: int = 4, regimes: bool = False) -> pd.DataFrame:
    """Synthetic OHLC bars.

    regimes=False -> pure random walk (no edge exists; a correct backtester must
                     show the strategy losing roughly its trading costs).
    regimes=True  -> regime-switching walk (trend up / down / range) that
                     exercises every strategy branch.

    This is NOT market data and says nothing about profitability. It exists so
    the backtester, risk engine and tests can be exercised offline.
    """
    rng = np.random.default_rng(seed)
    closes: list[float] = []
    price = start_price
    while len(closes) < n_bars:
        length = int(rng.integers(80, 300))
        kind = rng.choice(["up", "down", "range"], p=[0.4, 0.25, 0.35]) if regimes else "flat"
        anchor = price
        for _ in range(min(length, n_bars - len(closes))):
            if kind == "up":
                drift = 0.25 * bar_vol
            elif kind == "down":
                drift = -0.25 * bar_vol
            elif kind == "range":
                drift = -0.08 * np.log(price / anchor)
            else:
                drift = 0.0
            price *= float(np.exp(drift + bar_vol * rng.standard_normal()))
            closes.append(price)
    close = np.array(closes)
    open_ = np.r_[start_price, close[:-1]] * np.exp(rng.normal(0, bar_vol * 0.1, n_bars))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, bar_vol * 0.6, n_bars)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, bar_vol * 0.6, n_bars)))

    idx = pd.date_range(start, periods=int(n_bars * 1.6) + 10, freq=f"{bar_hours}h", tz="UTC")
    idx = idx[idx.weekday < 5][:n_bars]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


SYNTHETIC_PROFILES = {
    # name: (start_price in IG quote units, 4H volatility)
    "EURUSD": (11600.0, 0.0018),
    "GBPUSD": (13300.0, 0.0020),
    "UK100": (9500.0, 0.0030),
    "US500": (6600.0, 0.0035),
    "GER40": (23500.0, 0.0035),
    "GOLD": (3900.0, 0.0045),
    "SILVER": (4800.0, 0.0070),
    "COPPER": (13800.0, 0.0050),   # IG quote units (~13,870 on 2026-11-09)
}


def synthetic_universe(markets: list[dict], n_bars: int = 3000, seed: int = 7,
                       regimes: bool = False) -> dict[str, pd.DataFrame]:
    data = {}
    for i, m in enumerate(markets):
        price, vol = SYNTHETIC_PROFILES.get(m["name"], (1000.0, 0.003))
        data[m["name"]] = synthetic_market(seed + i, n_bars, price, vol, regimes=regimes)
    return data
