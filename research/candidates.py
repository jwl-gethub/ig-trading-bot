"""Four candidate strategies + a gold-scalper side check, as RESEARCH ONLY (job RESEARCH_JOB=candidates).

Order (cheap first, each report printed as it finishes): 2 turn-of-month, 3 RSI(2) rip-sell, 1 TSMOM, 4 time-of-day,
side check. Pass criteria and splits are fixed in advance (see the module docstrings); nothing is tuned.
Splits: development <= 2018-12-31, validation 2019-2022, out_of_sample 2023-01-01 onward, by entry time.
Daily bars close at 21:00 UTC for every market (except strategy 3, which uses the dip module's own schedules).
Limitations: FX funding ignores the rate differential; commodity/bond funding ignores basis; benchmark rates are
approximate annual averages; price P&L is not FX-translated to GBP.
"""
from __future__ import annotations

import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .cand_common import TSMOM_ORDER, UNIVERSE, say
from .cand_data import Hourly, daily_21utc, data_quality
from .cand_s1 import run_tsmom
from .cand_s23 import Hr, run_rip, run_tom
from .cand_s4 import (Minute, bars15, load_minutes, run_variant, scalp_split, scalp_stats, trades_4a, trades_4b)

START_HOURLY = date(2010, 1, 1)
START_MINUTE = date(2012, 1, 1)


def guarded(name: str, fn, *a):
    try:
        return fn(*a) or []
    except Exception as exc:                                   # one strategy failing must not stop the rest
        say(name, "ERROR", "error", repr(exc) + " | " + traceback.format_exc().replace("\n", " // ")[-900:])
        return []


def load_hourly(cache: Path, today: date):
    src = Hourly(cache.parent / "hourly")
    hourly, daily, scales = {}, {}, {}
    for name in TSMOM_ORDER:
        m = UNIVERSE[name]
        try:
            h, failed, scale = src.load(m, START_HOURLY, today, lambda s: say("ALL", "data", "note", s))
            d = daily_21utc(h)
            if len(d) < 300:
                raise RuntimeError(f"only {len(d)} daily bars")
            hourly[name], daily[name], scales[name] = h, d, scale
            say("ALL", "data", f"market:{name}", f"{data_quality(name, d)} scale={scale} hourly_bars={len(h)} "
                f"failed_files={len(failed)}{(' ' + ','.join(failed[:5])) if failed else ''}")
        except Exception as exc:
            print(f"CAND|ALL|data|skip:{name}|{exc!r}", flush=True)
    return hourly, daily, scales


def run_tod(cache: Path, end: date, scales: dict, gold_keep: dict) -> list[str]:
    S, passed = "TOD", []
    say(S, "ALL", "criteria", "PASS each of 4a and 4b separately = validation PF >= 1.2 AND OOS PF >= 1.2 after costs AND OOS "
        "trades >= 100 AND OOS avg bps > random-window p95 avg bps")
    say(S, "ALL", "conventions", "4a: long US500+USTECH from the last 1-min bar before 16:00 New York (price = its close) "
        "to the first bar at/after 09:35 New York next trading day (price = its open), Mon-Thu entries, funding for the actual "
        "nights held (normally one); 4b: long GOLD from the 23:00 UTC bar open to the 07:00 UTC bar open, Sun-Thu starts, "
        "no funding; a price must exist within 5 minutes of the wanted time else the trade is skipped; spread both ends; "
        "PF on net bps; random windows enter at a random minute bar open and exit at the bar open after the same length")
    mk = {}
    for name in ("US500", "USTECH", "GOLD"):
        if name not in scales:
            print(f"CAND|{S}|data|skip:{name}|no hourly scale/data", flush=True)
            continue
        m = UNIVERSE[name]
        try:
            m1 = load_minutes(m.inst, scales[name], START_MINUTE, end, cache,
                              cols=("open", "high", "low", "close") if name == "GOLD" else ("open", "close"))
            a = Minute(m1)
            if name == "GOLD":
                gold_keep["bars"] = bars15(m1)
            del m1
            mk[name] = (m, a)
            say(S, "ALL", "data", f"market={name} minute_bars={a.n} first={pd.Timestamp(a.ts[0], unit='ns')} "
                f"last={pd.Timestamp(a.ts[-1], unit='ns')}")
        except Exception as exc:
            print(f"CAND|{S}|data|skip:{name}|{exc!r}", flush=True)
    a4 = {k: v for k, v in mk.items() if k in ("US500", "USTECH")}
    if a4:
        passed += guarded(S, run_variant, S, "4a", a4, trades_4a, "overnight")
    if "GOLD" in mk:
        passed += guarded(S, run_variant, S, "4b", {"GOLD": mk["GOLD"]}, trades_4b, "asia")
    return passed


def run_side(cache: Path, end: date, gold_bars: pd.DataFrame | None) -> None:
    from bot.scalp import ScalpRules

    from .dukascopy import load_range
    from .scalp_test import MARKETS, backtest
    S = "SCALP"
    say(S, "ALL", "conventions", "research/scalp_test.py rules unchanged (ScalpRules defaults); informational only; splits by "
        "entry time; 'gold_ext' uses GOLD history from 2012 (not in scalp_test.MARKETS, which starts GOLD in 2024)")
    r = ScalpRules()
    inst, pr, hours, cost, start = MARKETS["GOLD"]
    tabs = {}
    if gold_bars is not None:
        tg = backtest(gold_bars, r, hours, cost)
        tabs["gold_ext"] = tg
        for s in ("development", "validation", "out_of_sample", "pooled"):
            say(S, "gold_ext", "splits", f"split={s} {scalp_stats(scalp_split(tg, s))}")
        wk = gold_bars.close.groupby(gold_bars.index.tz_localize(None).to_period("W")).last()
        mv = (wk / wk.shift(1) - 1).dropna()
        big = set(mv.index[mv.abs() > 0.05])
        tw = pd.PeriodIndex(pd.DatetimeIndex(tg.ts).tz_localize(None), freq="W") if len(tg) else []
        isbig = np.array([w in big for w in tw]) if len(tg) else np.array([], bool)
        for s in ("development", "validation", "out_of_sample", "pooled"):
            wsel = mv.index[(split_mask_period(mv.index, s))]
            nb = sum(1 for w in wsel if w in big)
            sub = scalp_split(tg, s)
            ib = np.array([w in big for w in pd.PeriodIndex(pd.DatetimeIndex(sub.ts).tz_localize(None), freq="W")]) \
                if len(sub) else np.array([], bool)
            say(S, "gold_ext", "weekly_move", f"split={s} weeks={len(wsel)} weeks_with_move_gt_5pct={nb}")
            say(S, "gold_ext", "weekly_move", f"split={s} big_weeks(|move|>5%): {scalp_stats(sub[ib]) if len(sub) else 'trades=0'}")
            say(S, "gold_ext", "weekly_move", f"split={s} other_weeks: {scalp_stats(sub[~ib]) if len(sub) else 'trades=0'}")
        wk_in = (pd.DatetimeIndex(tg.ts).tz_localize(None) >= pd.Timestamp("2026-11-16")) & \
                (pd.DatetimeIndex(tg.ts).tz_localize(None) < pd.Timestamp("2026-11-21")) if len(tg) else np.array([], bool)
        w1 = gold_bars.close[(gold_bars.index >= pd.Timestamp("2026-11-13 21:00", tz="UTC")) &
                             (gold_bars.index < pd.Timestamp("2026-11-20 21:00", tz="UTC"))]
        say(S, "gold_ext", "week_2026-11-16", f"backtest scalp trades entered 2026-11-16..20: "
            f"{scalp_stats(tg[wk_in]) if len(tg) else 'trades=0'}; gold move Fri 13 Nov 21:00 -> Fri 20 Nov 20:45 UTC: "
            f"{100 * (w1.iloc[-1] / w1.iloc[0] - 1):.2f}%" if len(w1) else "no data for that week")
    allt = []
    for name, (inst, pr, hours, cost, st) in MARKETS.items():
        try:
            m1 = load_range(inst, st, end, cache, pr, log=lambda *_: None)
            t = backtest(bars15(m1), r, hours, cost)
            del m1
            if name == "GOLD":
                for s in ("development", "validation", "out_of_sample", "pooled"):
                    say(S, "gold_window", "splits", f"(scalp_test window from {st}) split={s} {scalp_stats(scalp_split(t, s))}")
            if len(t):
                allt.append(t.assign(market=name))
            say(S, "all_markets", "per_market", f"market={name} from={st} {scalp_stats(t)}")
        except Exception as exc:
            print(f"CAND|{S}|data|skip:{name}|{exc!r}", flush=True)
    if allt:
        p = pd.concat(allt, ignore_index=True)
        for s in ("development", "validation", "out_of_sample", "pooled"):
            say(S, "all_markets", "splits", f"split={s} {scalp_stats(scalp_split(p, s))}")


def split_mask_period(idx: pd.PeriodIndex, s: str) -> np.ndarray:
    from .cand_common import split_mask
    return np.ones(len(idx), bool) if s == "pooled" else split_mask(idx.to_timestamp(), s)


def run(cache: Path) -> None:
    cache = Path(cache)
    today = date.today()
    end = today - timedelta(days=1)
    t0 = time.time()
    say("ALL", "run", "start", f"today={today} hourly_start={START_HOURLY} minute_start={START_MINUTE} universe={len(UNIVERSE)}")
    say("ALL", "run", "conventions", "daily bars close at 21:00 UTC for every market (strategy 3 uses the dip module's own "
        "close/fill schedules); signals use closed bars only; daily decisions execute at the open of the next daily bar "
        "(first price after the decision close) plus half spread each side; Dukascopy BID treated as mid; splits by entry "
        "time: development <=2018-12-31, validation 2019-2022, out_of_sample 2023-01-01 on; capital GBP 10000 fixed")
    say("ALL", "run", "spreads", "full spread in price units per market: " + ", ".join(
        f"{n}={m.spread:g}" for n, m in UNIVERSE.items()))
    say("ALL", "run", "limitations", "benchmark rates are approximate annual averages by calendar year; FX funding ignores the "
        "rate differential; commodity/bond funding ignores basis; price P&L is not FX-translated to GBP; spreads are constant "
        "through time (real spreads were wider in 2012-2019 and at night); Dukascopy data is not IG's own price feed")
    hourly, daily, scales = load_hourly(cache, today)
    say("ALL", "run", "data_loaded", f"markets={','.join(daily)} seconds={time.time() - t0:.0f}")
    passed: list[str] = []
    tom_h = {k: Hr(hourly[k]) for k in ("US500", "USTECH", "US30", "GER40", "UK100", "FRA40", "JPN225", "AUS200") if k in hourly}
    passed += guarded("TOM", run_tom, daily, tom_h)
    passed += guarded("RIPSELL", run_rip, hourly)
    passed += guarded("TSMOM", run_tsmom, daily)
    gold: dict = {}
    passed += guarded("TOD", run_tod, cache, end, scales, gold)
    guarded("SCALP", run_side, cache, end, gold.get("bars"))
    print(f"CAND|ALL|summary|PASSED:{','.join(passed) or 'none'}", flush=True)
    print("CAND|ALL|DONE", flush=True)
