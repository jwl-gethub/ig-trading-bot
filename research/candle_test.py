"""Backtest of two candle-pattern strategies (research/candle_patterns.py) on Dukascopy 1-minute data.

Template: research/structure_test.py. Same markets, costs, spread guard, time stop, splits, pass criteria.
Rules and variants are fixed in advance; no tuning. Results in R (1R = stop distance from the fill, costs
included). If a bar touches both stop and target the stop is assumed first; a gap through the stop fills
at the open. One position per market. The ATR 0.5-3.0 stop-size filter of structure_test is NOT applied.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from bot.structure import atr_series

from .candle_patterns import Setup, engulf_setups, resolve_entry, retrace_setups
from .dukascopy import load_range
from .scalp_test import END, MARKETS
from .structure_test import HOLD_BARS, PERIODS_FULL, PERIODS_SHORT, SPREAD_GUARD, metrics

# Pass criteria: copied unchanged from structure_test.run (pooled across markets)
PF_MIN, OOS_MIN_TRADES = 1.2, 100
CRITERIA = (f"PASS = out-of-sample profit factor >= {PF_MIN} AND validation profit factor >= {PF_MIN} "
            f"AND out-of-sample trades >= {OOS_MIN_TRADES} (pooled over markets, after costs)")
SEEDS = 20
AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}
TF = {"1H": "1h", "4H": "4h", "1D": "1D"}
# name: (strategy, timeframe, engulf rule, target R, primary)
VARIANTS = {
    "A_4H_high_2R": ("A", "4H", "high", 2.0, True),
    "A_4H_high_3R": ("A", "4H", "high", 3.0, False),
    "A_4H_close_2R": ("A", "4H", "close", 2.0, False),
    "A_4H_close_3R": ("A", "4H", "close", 3.0, False),
    "A_1H_high_2R_secondary": ("A", "1H", "high", 2.0, False),
    "A_1D_high_2R_secondary": ("A", "1D", "high", 2.0, False),
    "B_1D_3R": ("B", "1D", None, 3.0, True),
    "B_1D_2R": ("B", "1D", None, 2.0, False),
    "B_4H_3R_secondary": ("B", "4H", None, 3.0, False),
}
PRIMARIES = [v for v, s in VARIANTS.items() if s[4]]
SPLITS = ["development", "validation", "out_of_sample", "all"]
COLS = ["ts", "xts", "side", "r", "why", "cost_r"]


def log(obj: dict) -> None:
    print("RESEARCH " + json.dumps(obj, default=str), flush=True)


def say(line: str = "") -> None:
    print("REPORT " + line, flush=True)


def simulate(bars: pd.DataFrame, setups: list[Setup], target_r: float, cost: float) -> tuple[pd.DataFrame, dict]:
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx, n = bars.index, len(bars)
    trades, free_at = [], -1
    sk = {"skipped_spread": 0, "skipped_risk": 0, "unfilled": 0}
    for s in setups:
        f = resolve_entry(s, o, h, lo, n)
        if f is None:
            sk["unfilled"] += 1
            continue
        fi, px = f
        if fi <= free_at:
            continue
        side = s.side
        risk = side * (px - s.stop_level)
        if risk <= 0:
            sk["skipped_risk"] += 1
            continue
        if 2 * cost > SPREAD_GUARD * risk:
            sk["skipped_spread"] += 1
            continue
        stop, target = s.stop_level, px + side * target_r * risk
        exit_px, why, j = None, None, fi
        while j < n:
            if (lo[j] <= stop) if side == 1 else (h[j] >= stop):
                exit_px, why = (min(o[j], stop) if side == 1 else max(o[j], stop)), "stop"
            elif (h[j] >= target) if side == 1 else (lo[j] <= target):
                exit_px, why = target, "target"
            elif j - s.idx >= HOLD_BARS:
                exit_px, why = c[j], "time"
            if why:
                break
            j += 1
        if exit_px is None:
            break
        trades.append({"ts": idx[fi], "xts": idx[j], "side": side,
                       "r": (side * (exit_px - px) - 2 * cost) / risk, "why": why, "cost_r": 2 * cost / risk})
        free_at = j
    return pd.DataFrame(trades, columns=COLS), sk


def stat(t: pd.DataFrame) -> dict:
    m = metrics(t)
    if m["trades"]:
        m["total_r"] = round(float(t.r.sum()), 1)
    return m


def split(t: pd.DataFrame, periods: dict, name: str) -> pd.DataFrame:
    if name == "all":
        return t
    a, b = periods[name]
    return t[(t.ts >= pd.Timestamp(a, tz="UTC")) & (t.ts <= pd.Timestamp(b + " 23:59:59", tz="UTC"))]


def random_setups(bars: pd.DataFrame, real: list[Setup], rng: np.random.Generator) -> list[Setup]:
    """No-edge baseline: same number of entries at random bars and random sides, stop distance drawn from the
    real setups' stop distances (in ATR, measured from the signal close); filled at the next open."""
    c = bars.close.to_numpy(float)
    atr = atr_series(bars, 14)
    ok = np.flatnonzero(np.isfinite(atr) & (atr > 0))
    ok = ok[ok < len(bars) - 1]
    mult = np.array([abs(c[s.idx] - s.stop_level) / s.atr for s in real])
    pick = np.sort(rng.choice(ok, size=len(real), replace=len(real) > len(ok)))
    side = rng.choice([-1, 1], size=len(real))
    m = rng.choice(mult, size=len(real))
    return [Setup("engulf", int(i), int(sd), float(c[i] - sd * mm * atr[i]), float(atr[i]))
            for i, sd, mm in zip(pick, side, m)]


def fmt(m: dict) -> str:
    if not m.get("trades"):
        return "trades 0"
    return (f"trades {m['trades']:>5} win% {m['win_rate_pct']:>5} PF {m['profit_factor']} "
            f"avgR {m['avg_r']} totR {m.get('total_r')} maxDD_R {m['max_drawdown_r']}")


def run(cache: Path) -> None:
    say("CANDLE PATTERN BACKTEST (research only)")
    say("PRE-COMMITTED PASS CRITERIA: " + CRITERIA)
    say(f"Conventions: spread guard skip if 2*cost > {SPREAD_GUARD:.0%} of stop; time stop {HOLD_BARS} bars of the "
        "signal timeframe; stop buffer 0.1*ATR(14); entry A next-bar open, entry B stop order; R net of 2*cost")
    log({"event": "CANDLE_CRITERIA", "criteria": CRITERIA, "variants": list(VARIANTS)})
    pool = {v: {s: [] for s in SPLITS} for v in VARIANTS}
    base = {v: [{s: [] for s in SPLITS} for _ in range(SEEDS)] for v in VARIANTS}
    permkt = {v: {} for v in PRIMARIES}
    for name, (inst, pr, _hours, cost, start) in MARKETS.items():
        try:
            m1 = load_range(inst, start, END, cache, pr, log=lambda *_: None)
            frames = {}
            for tf, rule in TF.items():
                b = m1.resample(rule).agg({**AGG, "volume": "count"}).dropna(subset=["open"])
                frames[tf] = b
            del m1
            periods = PERIODS_FULL if start.year < 2024 else PERIODS_SHORT
            cache_setups: dict = {}
            for vname, (strat, tf, rule, tgt, prim) in VARIANTS.items():
                bars = frames[tf]
                key = (strat, tf, rule)
                if key not in cache_setups:
                    cache_setups[key] = (engulf_setups(bars, rule) if strat == "A" else retrace_setups(bars))
                setups = cache_setups[key]
                t, sk = simulate(bars, setups, tgt, cost)
                thin = int((bars.volume < 0.5 * bars.volume.median()).sum())
                log({"event": "CANDLE_TEST", "market": name, "variant": vname, "bars": len(bars),
                     "thin_bars_lt_half_median_minutes": thin, "setups": len(setups), **sk,
                     **{s: stat(split(t, periods, s)) for s in SPLITS},
                     "avg_cost_r": round(float(t.cost_r.mean()), 3) if len(t) else None,
                     "exits": t.why.value_counts().to_dict() if len(t) else {}})
                for s in SPLITS:
                    pool[vname][s].append(split(t, periods, s))
                if prim:
                    permkt[vname][name] = stat(split(t, periods, "out_of_sample"))
                if setups:
                    for k in range(SEEDS):
                        rng = np.random.default_rng([k, zlib.crc32((name + vname).encode())])
                        bt, _ = simulate(bars, random_setups(bars, setups, rng), tgt, cost)
                        for s in SPLITS:
                            base[vname][k][s].append(split(bt, periods, s))
        except Exception as exc:
            log({"event": "CANDLE_TEST_ERROR", "market": name, "error": repr(exc)})

    def cat(fr):
        fr = [f for f in fr if len(f)]
        return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=COLS)

    say()
    say("RESULTS BY VARIANT (pooled over markets, after costs; R units)")
    passed = []
    for vname, (strat, tf, rule, tgt, prim) in VARIANTS.items():
        row = {s: stat(cat(pool[vname][s])) for s in SPLITS}
        oos, val = row["out_of_sample"], row["validation"]
        ok = bool((oos.get("profit_factor") or 0) >= PF_MIN and (val.get("profit_factor") or 0) >= PF_MIN
                  and oos.get("trades", 0) >= OOS_MIN_TRADES)
        if ok:
            passed.append(vname)
        say(f"== {vname}{' [PRIMARY]' if prim else ''}  -> {'PASS' if ok else 'FAIL'}")
        for s in SPLITS:
            say(f"   {s if s != 'all' else 'pooled':<14} {fmt(row[s])}")
        brow = []
        for s in ("validation", "out_of_sample"):
            pfs, avs = [], []
            for k in range(SEEDS):
                m = stat(cat(base[vname][k][s]))
                if m.get("profit_factor") is not None:
                    pfs.append(m["profit_factor"]); avs.append(m["avg_r"])
            real = row[s].get("profit_factor")
            if pfs:
                beat = sum(p >= (real or 0) for p in pfs)
                brow.append(f"{s}: random-entry PF mean {np.mean(pfs):.2f} p95 {np.percentile(pfs, 95):.2f} "
                            f"avgR mean {np.mean(avs):.3f}; random >= real PF in {beat}/{len(pfs)} seeds")
        say("   baseline " + " | ".join(brow))
        log({"event": "CANDLE_SUMMARY", "variant": vname, **row, "pass": ok, "baseline": brow})
    say()
    say("PER-MARKET OUT-OF-SAMPLE (primary variants)")
    for vname in PRIMARIES:
        say(f"== {vname}")
        for mk, m in permkt[vname].items():
            say(f"   {mk:<8} {fmt(m)}")
    say()
    say("PASSED: " + (", ".join(passed) or "none"))
    log({"event": "CANDLE_DONE", "passed": passed})
