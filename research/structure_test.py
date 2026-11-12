"""Backtest of the market-structure pullback (bot/structure.py) on Dukascopy 1-minute data, IG-like costs.

Rules are fixed in advance; 8 variants, no tuning. Entry at the next 15-minute bar's open plus cost.
If a bar touches both stop and target the stop is assumed first; a gap through the stop fills at the open.
Results in R (1R = initial stop distance from the fill, costs included). One position per market.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from bot.structure import StructureRules, signals

from .dukascopy import load_range
from .scalp_test import END, MARKETS

HOLD_BARS = 48           # 12 hours of 15-minute bars
SPREAD_GUARD = 0.10      # skip if 2 * cost > 10% of the stop distance
PERIODS_FULL = {"development": ("2019-01-01", "2022-12-31"), "validation": ("2023-01-01", "2024-12-31"),
                "out_of_sample": ("2025-01-01", "2100-01-01")}
PERIODS_SHORT = {"validation": ("2024-01-01", "2024-12-31"), "out_of_sample": ("2025-01-01", "2100-01-01")}
VARIANTS = {f"t{int(t)}_htf{int(h)}_be{0 if b is None else 1}": (t, h, b)
            for t in (2.0, 3.0) for h in (False, True) for b in (None, 1.0)}


def log(obj: dict) -> None:
    print("RESEARCH " + json.dumps(obj, default=str), flush=True)


def simulate(bars: pd.DataFrame, entries: list, r: StructureRules, cost: float) -> tuple[pd.DataFrame, dict]:
    """Exit simulation for entries (already computed) -> (trades, skip counters)."""
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx, n = bars.index, len(bars)
    trades, free_at = [], -1
    skipped = {"skipped_spread": 0, "skipped_range": 0}
    for e in entries:
        i, side = e.idx, e.side
        if i < free_at or i + 1 >= n:
            continue
        px = o[i + 1]
        risk = side * (px - e.stop_level)                     # measured from the actual fill
        if risk <= 0 or not (r.min_stop_atr * e.atr <= risk <= r.max_stop_atr * e.atr):
            skipped["skipped_range"] += 1
            continue
        if 2 * cost > SPREAD_GUARD * risk:
            skipped["skipped_spread"] += 1
            continue
        stop, target = e.stop_level, px + side * r.target_r * risk
        be_px = None if r.breakeven_at_r is None else px + side * r.breakeven_at_r * risk
        moved, exit_px, why, j = False, None, None, i + 1
        while j < n:
            if (lo[j] <= stop) if side == 1 else (h[j] >= stop):
                exit_px = min(o[j], stop) if side == 1 else max(o[j], stop)
                why = "breakeven" if moved else "stop"
            elif (h[j] >= target) if side == 1 else (lo[j] <= target):
                exit_px, why = target, "target"
            elif j - i >= HOLD_BARS:
                exit_px, why = c[j], "time"
            if why:
                break
            if be_px is not None and not moved and ((h[j] >= be_px) if side == 1 else (lo[j] <= be_px)):
                stop, moved = px, True                        # applies from the next bar
            j += 1
        if exit_px is None:
            break
        trades.append({"ts": idx[i + 1], "xts": idx[j], "side": side,
                       "r": (side * (exit_px - px) - 2 * cost) / risk, "why": why, "cost_r": 2 * cost / risk})
        free_at = j
    return pd.DataFrame(trades, columns=["ts", "xts", "side", "r", "why", "cost_r"]), skipped


def metrics(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    t = t.sort_values("xts")
    wins, losses = t.r[t.r > 0], t.r[t.r <= 0]
    eq = t.r.cumsum()
    lose = (t.r <= 0).to_numpy()
    streak = best = 0
    for x in lose:
        streak = streak + 1 if x else 0
        best = max(best, streak)
    return {"trades": int(len(t)), "win_rate_pct": round(100 * len(wins) / len(t), 1),
            "avg_r": round(float(t.r.mean()), 3),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
            "max_drawdown_r": round(float((eq.cummax() - eq).max()), 1), "max_consec_losses": int(best)}


def by_period(t: pd.DataFrame, periods: dict) -> dict:
    out = {}
    for name, (a, b) in periods.items():
        sel = t[(t.ts >= pd.Timestamp(a, tz="UTC")) & (t.ts <= pd.Timestamp(b + " 23:59:59", tz="UTC"))] \
            if len(t) else t
        out[name] = metrics(sel)
    out["all"] = metrics(t)
    return out


def session_mask(index: pd.DatetimeIndex, hours: tuple[int, int]) -> np.ndarray:
    """Same rule as bot.scalp.in_session, vectorised, evaluated at the bar close (start + 15 min)."""
    end = index + pd.Timedelta(minutes=15)
    return np.asarray((end.dayofweek < 5) & (end.hour >= hours[0]) & (end.hour < hours[1]))


def backtest_market(bars: pd.DataFrame, h1: pd.DataFrame, hours: tuple[int, int], cost: float,
                    rules: StructureRules = StructureRules()) -> pd.DataFrame:
    """One run of a single variant (used by the tests)."""
    e = signals(bars, rules, h1 if rules.htf_filter else None, session_mask(bars.index, hours))
    return simulate(bars, e, rules, cost)[0]


def pooled_row(frames: list[pd.DataFrame], periods: dict) -> dict:
    t = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame(
        columns=["ts", "xts", "side", "r", "why", "cost_r"])
    return by_period(t, periods)


def run(cache: Path) -> None:
    base = StructureRules()
    pool: dict[str, dict[str, list]] = {v: {"development": [], "validation": [], "out_of_sample": [], "all": []}
                                        for v in VARIANTS}
    for name, (inst, pr, hours, cost, start) in MARKETS.items():
        try:
            m1 = load_range(inst, start, END, cache, pr, log=lambda *_: None)
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            bars = m1.resample("15min").agg(agg).dropna()
            h1 = m1.resample("1h").agg(agg).dropna()
            del m1
            sess = session_mask(bars.index, hours)
            periods = PERIODS_FULL if start.year < 2024 else PERIODS_SHORT
            ent = {flag: signals(bars, replace(base, htf_filter=flag), h1 if flag else None, sess)
                   for flag in (False, True)}
            for vname, (tgt, flag, be) in VARIANTS.items():
                rules = replace(base, target_r=tgt, htf_filter=flag, breakeven_at_r=be)
                t, skipped = simulate(bars, ent[flag], rules, cost)
                free, _ = simulate(bars, ent[flag], rules, 0.0)
                res = by_period(t, periods)
                log({"event": "STRUCT_TEST", "market": name, "variant": vname, **res,
                     "avg_cost_r": round(float(t.cost_r.mean()), 3) if len(t) else None, **skipped,
                     "long_pct": round(100 * float((t.side == 1).mean()), 1) if len(t) else None,
                     "exits": t.why.value_counts().to_dict() if len(t) else {},
                     "no_cost_avg_r": round(float(free.r.mean()), 3) if len(free) else None,
                     "signals": len(ent[flag])})
                if len(t):
                    for per, (a, b) in periods.items():
                        pool[vname][per].append(t[(t.ts >= pd.Timestamp(a, tz="UTC")) &
                                                  (t.ts <= pd.Timestamp(b + " 23:59:59", tz="UTC"))])
                    pool[vname]["all"].append(t)
        except Exception as exc:
            log({"event": "STRUCT_TEST_ERROR", "market": name, "error": repr(exc)})
    passed = []
    for vname in VARIANTS:
        row = {per: metrics(pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()) for per, fr in
               pool[vname].items()}
        oos, val = row["out_of_sample"], row["validation"]
        ok = bool((oos.get("profit_factor") or 0) >= 1.2 and (val.get("profit_factor") or 0) >= 1.2
                  and oos.get("trades", 0) >= 100)
        if ok:
            passed.append(vname)
        log({"event": "STRUCT_SUMMARY", "variant": vname, **row, "pass": ok})
    log({"event": "STRUCT_DONE", "passed": passed})
