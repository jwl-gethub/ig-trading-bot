"""Backtest of the 15-minute scalper (bot/scalp.py) on Dukascopy 1-minute data, with IG-like costs.

Fill at the next bar's open plus cost. If a bar touches both the stop and the target, the stop is
assumed first (conservative). Time stop at the open after max_bars. Results in R (1R = the stop).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from bot.scalp import ScalpRules, decide, in_session, indicators

from .dukascopy import load_range

# name: (instrument, price range, UTC session hours, cost per fill in price units, data start)
MARKETS = {
    "US500": ("USA500IDXUSD", (1500, 12000), (14, 21), 0.3, date(2018, 11, 1)),
    "USTECH": ("USATECHIDXUSD", (4000, 40000), (14, 21), 0.75, date(2018, 11, 1)),
    "GER40": ("DEUIDXEUR", (7000, 40000), (8, 16), 0.9, date(2018, 11, 1)),
    "UK100": ("GBRIDXGBP", (4000, 15000), (8, 16), 0.75, date(2018, 11, 1)),
    "EURUSD": ("EURUSD", (0.9, 1.5), (7, 17), 0.00004, date(2024, 1, 1)),
    "GBPUSD": ("GBPUSD", (1.0, 1.8), (7, 17), 0.00006, date(2024, 1, 1)),
    "GOLD": ("XAUUSD", (1000, 6000), (7, 20), 0.25, date(2024, 1, 1)),
}
END = date(2026, 11, 6)


def log(obj: dict) -> None:
    print("RESEARCH " + json.dumps(obj, default=str), flush=True)


def backtest(bars: pd.DataFrame, r: ScalpRules, hours: tuple[int, int], cost: float) -> pd.DataFrame:
    d = indicators(bars, r)
    o, h, lo = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    idx = d.index
    trades, i, n = [], 0, len(d)
    while i < n - 1:
        row = d.iloc[i]
        act = decide(row, r) if in_session(idx[i] + pd.Timedelta(minutes=15), hours) else None
        if not act:
            i += 1
            continue
        side = 1 if act == "long" else -1
        risk = r.stop_atr * row.atr
        entry = o[i + 1] + side * cost
        stop, target = o[i + 1] - side * risk, o[i + 1] + side * r.target_atr * row.atr
        exit_px, why, j = None, None, i + 1
        while j < n:
            hit_stop = lo[j] <= stop if side == 1 else h[j] >= stop
            hit_tgt = h[j] >= target if side == 1 else lo[j] <= target
            if hit_stop:
                exit_px, why = (min(o[j], stop) if side == 1 else max(o[j], stop)), "stop"
            elif hit_tgt:
                exit_px, why = target, "target"
            elif j - i >= r.max_bars or (idx[j] - idx[i + 1]) > pd.Timedelta(hours=6):
                exit_px, why = d.close.iloc[j], "time"
            if why:
                break
            j += 1
        if exit_px is None:
            break
        fill = exit_px - side * cost
        trades.append({"ts": idx[i + 1], "side": side, "r": side * (fill - entry) / risk, "why": why,
                       "cost_r": 2 * cost / risk})
        i = j + 1
    return pd.DataFrame(trades)


def summary(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    wins, losses = t.r[t.r > 0], t.r[t.r <= 0]
    eq = t.r.cumsum()
    return {"trades": int(len(t)), "win_rate_pct": round(100 * len(wins) / len(t), 1),
            "avg_r": round(float(t.r.mean()), 3), "total_r": round(float(t.r.sum()), 1),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
            "max_drawdown_r": round(float((eq.cummax() - eq).max()), 1),
            "avg_cost_r": round(float(t.cost_r.mean()), 3), "exits": t.why.value_counts().to_dict(),
            "long_pct": round(100 * float((t.side == 1).mean()), 1),
            "by_year_r": {str(y): round(float(g.r.sum()), 1) for y, g in t.groupby(t.ts.dt.year)}}


def run(cache: Path) -> None:
    r = ScalpRules()
    for name, (inst, pr, hours, cost, start) in MARKETS.items():
        try:
            m1 = load_range(inst, start, END, cache, pr, log=lambda *_: None)
            bars = m1.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            t = backtest(bars, r, hours, cost)
            free = backtest(bars, r, hours, 0.0)
            log({"event": "SCALP_TEST", "market": name, "from": str(bars.index[0].date()), **summary(t),
                 "no_cost_avg_r": round(float(free.r.mean()), 3) if len(free) else None,
                 "recent_2025_26": summary(t[t.ts >= pd.Timestamp("2025-01-01", tz="UTC")]) if len(t) else None})
        except Exception as exc:
            log({"event": "SCALP_TEST_ERROR", "market": name, "error": repr(exc)})
    log({"event": "SCALP_TEST_DONE"})

