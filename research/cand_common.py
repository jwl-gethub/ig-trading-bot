"""Shared pieces for the four candidate strategies (research only): markets, costs, funding, splits, reporting.

Every report line is `CAND|<strategy>|<variant>|<section>|<text>` so lines stay identifiable even when the log API
returns them out of order. Nothing here talks to a broker.

Cost conventions (fixed by the brief):
- Spread: full spread in price units per market (config.yaml backtest_spread_points where present, else the IG-like
  figures listed below). Dukascopy BID prices are treated as mid; each fill pays half the spread against the trader.
- Overnight funding per night held through 21:00 UTC (Friday night counts x3, Saturday/Sunday 0), on notional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CAPITAL = 10_000.0
STRESS = 1.5
SPLIT_NAMES = ["development", "validation", "out_of_sample"]
SPLIT_BOUNDS = {"development": (None, "2018-12-31"), "validation": ("2019-01-01", "2022-12-31"),
                "out_of_sample": ("2023-01-01", None)}

# Approximate annual average benchmark rates (%), by calendar year. Approximate, supplied with the brief.
USD = {2010: .18, 2011: .10, 2012: .14, 2013: .11, 2014: .09, 2015: .13, 2016: .40, 2017: 1.00, 2018: 1.83,
       2019: 2.16, 2020: .38, 2021: .08, 2022: 1.68, 2023: 5.02, 2024: 5.14, 2025: 4.20, 2026: 3.70}
GBP = {**{y: .50 for y in range(2010, 2016)}, 2016: .40, 2017: .30, 2018: .60, 2019: .75, 2020: .20, 2021: .10,
       2022: 1.50, 2023: 4.70, 2024: 5.10, 2025: 4.30, 2026: 3.80}
EUR = {2010: .40, 2011: .90, 2012: .20, 2013: .10, 2014: .00, 2015: -.20, 2016: -.35, 2017: -.36, 2018: -.36,
       2019: -.40, 2020: -.47, 2021: -.57, 2022: .00, 2023: 3.20, 2024: 3.60, 2025: 2.20, 2026: 2.00}
BENCH = {"USD": USD, "GBP": GBP, "EUR": EUR}
Y0, Y1 = 2010, 2035


def bench_rate(ccy: str, year: int) -> float:
    t = BENCH[ccy]
    return t[min(max(year, 2010), 2026)] / 100.0


@dataclass(frozen=True)
class Mkt:
    name: str
    inst: str                      # Dukascopy instrument
    probe: tuple[float, float]     # plausible price range NOW (ratio < 10), used to detect the integer price scale
    kind: str                      # index | fx | metal | other   (funding class)
    ccy: str | None                # benchmark currency for indices
    spread: float                  # FULL spread, price units
    sched: str | None = None       # dip schedule key (NY_INDEX, EU_INDEX, UK_INDEX, FX_DAY) when used in strategy 3
    pscale: float = 1.0            # IG price scale (stake GBP/point = units / pscale)
    min_size: float | None = None  # IG minimum stake GBP/point from config.yaml (None = unknown)
    scale: int = 1000              # default integer scale if detection fails


M = Mkt
UNIVERSE: dict[str, Mkt] = {m.name: m for m in [
    M("US500", "USA500IDXUSD", (1500, 12000), "index", "USD", 0.8, "NY_INDEX", 1, 0.04),
    M("USTECH", "USATECHIDXUSD", (4000, 40000), "index", "USD", 1.0, "NY_INDEX", 1, 0.04),
    M("US30", "USA30IDXUSD", (15000, 70000), "index", "USD", 2.4),
    M("GER40", "DEUIDXEUR", (7000, 40000), "index", "EUR", 2.0, "EU_INDEX", 1, 0.04),
    M("UK100", "GBRIDXGBP", (4000, 15000), "index", "GBP", 1.5, "UK_INDEX", 1, 0.04),
    M("FRA40", "FRAIDXEUR", (3000, 12000), "index", "EUR", 2.0),
    M("JPN225", "JPNIDXJPY", (20000, 90000), "index", "USD", 7.0),
    M("AUS200", "AUSIDXAUD", (4000, 14000), "index", "USD", 2.0),
    M("HK50", "HKGIDXHKD", (8000, 50000), "index", "USD", 8.0),
    M("EURUSD", "EURUSD", (0.9, 1.5), "fx", None, 0.9e-4, "FX_DAY", 10000, 0.04, 100000),
    M("GBPUSD", "GBPUSD", (1.0, 1.8), "fx", None, 1.4e-4, "FX_DAY", 10000, 0.03, 100000),
    M("USDJPY", "USDJPY", (90, 200), "fx", None, 0.007, None, 100, None, 1000),
    M("AUDUSD", "AUDUSD", (0.5, 1.0), "fx", None, 0.6e-4, None, 10000, None, 100000),
    M("USDCAD", "USDCAD", (1.1, 1.6), "fx", None, 1.7e-4, None, 10000, None, 100000),
    M("USDCHF", "USDCHF", (0.7, 1.1), "fx", None, 1.5e-4, None, 10000, None, 100000),
    M("NZDUSD", "NZDUSD", (0.5, 0.8), "fx", None, 1.6e-4, "FX_DAY", 10000, 0.04, 100000),
    M("EURJPY", "EURJPY", (100, 200), "fx", None, 0.015, "FX_DAY", 100, 0.04, 1000),
    M("EURGBP", "EURGBP", (0.7, 1.0), "fx", None, 0.9e-4, None, 10000, None, 100000),
    M("GOLD", "XAUUSD", (1000, 6000), "metal", "USD", 0.6, "FX_DAY", 1, 1.0),
    M("SILVER", "XAGUSD", (8, 120), "metal", "USD", 0.04, None, 1, 1.0),   # config.yaml 4.0 points = 0.04 USD (IG demo, scale x100 assumed; > harness 4e-4/fill)
    M("WTI", "LIGHTCMDUSD", (15, 150), "other", None, 0.028, None, 1),
    M("BRENT", "BRENTCMDUSD", (15, 150), "other", None, 0.028, None, 1),
    M("NATGAS", "GASCMDUSD", (1, 15), "other", None, 0.06, None, 1),
    M("COPPER", "COPPERCMDUSD", (1.5, 8), "other", None, 12e-4 * 2 * 4.5, None, 1, 1.0),  # harness 12e-4/fill of ~4.5
    M("USTBOND", "USTBONDTRUSD", (80, 200), "other", None, 0.0625, None, 1),
    M("BUND", "DEUBONDTREUR", (100, 200), "other", None, 0.03, None, 1),
    M("GILT", "GBRBONDTRGBP", (70, 150), "other", None, 0.03, None, 1),
]}
TSMOM_ORDER = list(UNIVERSE)
TOM_PRIMARY = ["US500", "USTECH", "US30", "GER40", "UK100"]
TOM_EXTRA = ["FRA40", "JPN225", "AUS200"]
RIP_MARKETS = ["US500", "USTECH", "GER40", "UK100", "GBPUSD", "EURUSD", "NZDUSD", "EURJPY", "GOLD"]


# ----------------------------------------------------------------------------------------------- funding
def night_weight(weekday: np.ndarray) -> np.ndarray:
    """Monday-Thursday nights 1, Friday night 3 (covers the weekend), Saturday/Sunday 0."""
    w = np.zeros(len(weekday))
    w[weekday <= 3] = 1.0
    w[weekday == 4] = 3.0
    return w


def annual_rate(m: Mkt, direction: int, year: np.ndarray | int) -> np.ndarray | float:
    """Annual funding rate as a fraction; positive = the position PAYS, negative = it RECEIVES a credit."""
    y = np.asarray(year)
    if m.kind == "fx":
        r = np.full(y.shape, 0.015)
    elif m.kind == "other":
        r = np.full(y.shape, 0.034)
    else:
        tab = np.array([bench_rate(m.ccy, int(v)) for v in range(Y0, Y1)])
        b = tab[np.clip(y, Y0, Y1 - 1) - Y0]
        fee = 0.025 if m.kind == "index" else 0.015
        r = b + fee if direction > 0 else fee - b
    return r if y.shape else float(r)


def stress_rate(r, stress: float):
    """Costs x stress: only charges are scaled; credits (negative) are left alone."""
    return np.where(np.asarray(r) > 0, np.asarray(r) * stress, r)


def funding_frac(m: Mkt, direction: int, t_in_ns: np.ndarray, t_out_ns: np.ndarray, stress: float = 1.0) -> np.ndarray:
    """Funding as a fraction of notional for holding from t_in to t_out (UTC ns arrays).
    One charge per weekday 21:00 UTC boundary B with t_in <= B < t_out (Friday x3), at that year's rate / 365."""
    t_in = np.atleast_1d(np.asarray(t_in_ns, dtype="int64"))
    t_out = np.atleast_1d(np.asarray(t_out_ns, dtype="int64"))
    day = 86_400 * 10 ** 9
    cut0 = (t_in // day) * day + 21 * 3600 * 10 ** 9
    out = np.zeros(len(t_in))
    span = int(np.ceil((t_out - t_in).max() / day)) + 2 if len(t_in) else 0
    for k in range(span):
        cut = cut0 + k * day
        ok = (cut >= t_in) & (cut < t_out)
        if not ok.any():
            continue
        dt = pd.to_datetime(cut[ok], unit="ns")
        w = night_weight(np.asarray(dt.dayofweek))
        rate = np.asarray(annual_rate(m, direction, np.asarray(dt.year)), dtype=float)
        out[ok] += w * np.asarray(stress_rate(rate, stress)) / 365.0
    return out


# ----------------------------------------------------------------------------------------------- stats
def split_mask(ts: pd.Series | pd.DatetimeIndex, name: str) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    a, b = SPLIT_BOUNDS[name]
    m = np.ones(len(idx), dtype=bool)
    if a:
        m &= idx >= pd.Timestamp(a)
    if b:
        m &= idx < pd.Timestamp(b) + pd.Timedelta(days=1)
    return m


def pf(x: np.ndarray) -> float | None:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else None


def max_dd(x: np.ndarray) -> float:
    if not len(x):
        return 0.0
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max())


def f2(v, nd=2) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


# ----------------------------------------------------------------------------------------------- reporting
def say(strategy: str, variant: str, section: str, text: str) -> None:
    print(f"CAND|{strategy}|{variant}|{section}|{text}", flush=True)
