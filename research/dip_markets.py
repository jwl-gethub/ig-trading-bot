"""Screen the daily RSI(2) dip strategy ("quick exit") across indices, FX and commodities.

Same rules module as the live bot (bot/dipbuy.py). Hourly Dukascopy data from 2018. Fills at the
open of the first hourly bar at the market's fill time; the server-side stop is checked on every
hourly bar (IG markets trade ~24h, so overnight moves count); costs are a per-fill fraction of
price; overnight funding per calendar night. 1x notional, no leverage.

Pass criteria, fixed before running: at least 20 trades, profit factor >= 1.3, and a positive total
return in both 2019-2022 and 2023-2026. Indices are long-only (equity drift); FX and commodities
trade both ways (no natural drift).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import pandas as pd

from bot.dipbuy import DipRules, DipSchedule, daily_bars, decide, fill_time, indicators
from bot.dukascopy import HourlySource

from .noise_area import metrics

NY_INDEX = DipSchedule("America/New_York", 16, "America/New_York", 10)
EU_INDEX = DipSchedule("Europe/Berlin", 17, "Europe/Berlin", 10)
UK_INDEX = DipSchedule("Europe/London", 16, "Europe/London", 9)
FX_DAY = DipSchedule("America/New_York", 17, "Europe/London", 8)

# name: (dukascopy instrument, price range, schedule, long_only, cost per fill as fraction, group)
UNIVERSE = {
    "US500": ("USA500IDXUSD", (1500, 12000), NY_INDEX, True, 0.5e-4, "index"),
    "USTECH": ("USATECHIDXUSD", (4000, 40000), NY_INDEX, True, 0.5e-4, "index"),
    "US30": ("USA30IDXUSD", (15000, 70000), NY_INDEX, True, 0.5e-4, "index"),
    "GER40": ("DEUIDXEUR", (7000, 40000), EU_INDEX, True, 0.5e-4, "index"),
    "FRA40": ("FRAIDXEUR", (3000, 12000), EU_INDEX, True, 0.9e-4, "index"),
    "EU50": ("EUSIDXEUR", (2000, 8000), EU_INDEX, True, 2.5e-4, "index"),
    "UK100": ("GBRIDXGBP", (4000, 15000), UK_INDEX, True, 0.8e-4, "index"),
    "EURUSD": ("EURUSD", (0.9, 1.5), FX_DAY, False, 0.5e-4, "fx"),
    "GBPUSD": ("GBPUSD", (1.0, 1.8), FX_DAY, False, 0.6e-4, "fx"),
    "USDJPY": ("USDJPY", (90, 200), FX_DAY, False, 0.5e-4, "fx"),
    "AUDUSD": ("AUDUSD", (0.5, 1.0), FX_DAY, False, 0.7e-4, "fx"),
    "USDCAD": ("USDCAD", (1.1, 1.6), FX_DAY, False, 0.75e-4, "fx"),
    "USDCHF": ("USDCHF", (0.7, 1.1), FX_DAY, False, 1.1e-4, "fx"),
    "NZDUSD": ("NZDUSD", (0.5, 0.8), FX_DAY, False, 1.6e-4, "fx"),
    "EURGBP": ("EURGBP", (0.7, 1.0), FX_DAY, False, 0.8e-4, "fx"),
    "EURJPY": ("EURJPY", (100, 200), FX_DAY, False, 0.8e-4, "fx"),
    "GBPJPY": ("GBPJPY", (120, 230), FX_DAY, False, 0.9e-4, "fx"),
    "GOLD": ("XAUUSD", (1000, 6000), FX_DAY, False, 0.9e-4, "commodity"),
    "SILVER": ("XAGUSD", (8, 120), FX_DAY, False, 4e-4, "commodity"),
    "BRENT": ("BRENTCMDUSD", (15, 150), FX_DAY, False, 2.5e-4, "commodity"),
    "COPPER": ("COPPERCMDUSD", (1.5, 8), FX_DAY, False, 12e-4, "commodity"),
    "NATGAS": ("GASCMDUSD", (1, 15), FX_DAY, False, 12e-4, "commodity"),
}
START, TODAY = date(2018, 1, 1), date(2026, 11, 7)
HALVES = (("2019-01-01", "2023-01-01"), ("2023-01-01", None))
FUND_LONG = {"index": 0.065, "fx": 0.025, "commodity": 0.065}
FUND_SHORT = {"index": 0.0, "fx": 0.025, "commodity": 0.0}


def log(obj: dict) -> None:
    print("RESEARCH " + json.dumps(obj, default=str), flush=True)


def backtest(hourly: pd.DataFrame, s: DipSchedule, r: DipRules, cost: float, f_long: float, f_short: float):
    d = indicators(daily_bars(hourly, s), r)
    hs = hourly.index.as_unit("ns").asi8                         # UTC nanoseconds
    ho, hh, hl = (hourly[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    ptr, n = 0, len(hourly)
    pos, entry, mark, stop, t_in, pending = 0, 0.0, 0.0, None, None, None
    trades, daily = [], []
    prev_close_ts = None
    for ts, row in d.iterrows():
        ret = 0.0

        def close_out(px: float, why: str, when) -> None:
            nonlocal pos, ret
            fill = px * (1 - pos * cost)
            ret += pos * (fill / mark - 1)
            rate = f_long if pos > 0 else f_short
            nights = max(0, (when - t_in).days)
            trades.append({"day": ts.tz_convert(None), "dir": pos, "entry": entry, "exit": fill, "reason": why,
                           "days_held": nights, "ret": pos * (fill / entry - 1),
                           "acct_ret": pos * (fill / entry - 1) - rate * nights / 365, "lev": 1.0})
            pos = 0

        if pos and prev_close_ts is not None:                        # funding for nights held
            ret -= (f_long if pos > 0 else f_short) * (ts - prev_close_ts).days / 365
        while ptr < n and hs[ptr] < ts.as_unit("ns").value:
            t = pd.Timestamp(hs[ptr], unit="ns", tz="UTC")
            if pending and t >= pending[1]:
                act = pending[0]
                if act == "exit" and pos:
                    close_out(ho[ptr], "exit", t)
                elif act in ("long", "short") and not pos:
                    pos = 1 if act == "long" else -1
                    entry = ho[ptr] * (1 + pos * cost)
                    mark, t_in = entry, t
                    stop = ho[ptr] * (1 - pos * pending[2])
                pending = None
            if pos and stop is not None:
                if pos > 0 and hl[ptr] <= stop:
                    close_out(min(ho[ptr], stop), "stop", t)
                elif pos < 0 and hh[ptr] >= stop:
                    close_out(max(ho[ptr], stop), "stop", t)
            ptr += 1
        if pos:
            ret += pos * (row.close / mark - 1)
            mark = row.close
        act = decide(row, pos, r)
        if act and not pending:
            pending = (act, fill_time(ts, s), r.stop_atr * row.atr_pct)
        daily.append({"day": ts.tz_convert(None).normalize(), "ret": ret, "lev": float(pos != 0)})
        prev_close_ts = ts
    return pd.DataFrame(trades), pd.DataFrame(daily)


def screen(cache: str) -> dict:
    src = HourlySource(cache, log=log)
    verdicts = {}
    for name, (inst, pr, sched, long_only, cost, group) in UNIVERSE.items():
        try:
            hourly, failed = src.hourly(inst, START, TODAY, pr)
        except Exception as exc:                                     # one market's data problem
            log({"event": "DIP_DATA_ERROR", "market": name, "error": repr(exc)})
            continue
        r = DipRules(long_only=long_only)
        trades, daily = backtest(hourly, sched, r, cost, FUND_LONG[group], FUND_SHORT[group])
        full = metrics(trades, daily, "2019-01-01", None)
        halves = [metrics(trades, daily, a, b) for a, b in HALVES]
        ok = (full.get("trades", 0) >= 20 and (full.get("profit_factor") or 0) >= 1.3
              and all((h.get("total_return_pct") or -1) > 0 for h in halves))
        verdicts[name] = ok
        hold = None
        if len(hourly):
            dd = daily_bars(hourly, sched)
            dd = dd[dd.index >= pd.Timestamp("2019-01-01", tz="UTC")]
            hold = round(100 * float(dd.close.iloc[-1] / dd.close.iloc[0] - 1), 1) if len(dd) > 1 else None
        long_share = full.get("long_share_pct")
        log({"event": "DIP_SCREEN", "market": name, "group": group, "pass": ok, "long_only": long_only,
             "hourly_bars": int(len(hourly)), "failed_files": failed[:10] or None, "buy_and_hold_pct": hold,
             "full": {k: full.get(k) for k in ("trades", "win_rate_pct", "profit_factor", "annual_return_pct",
                                               "max_drawdown_pct", "days_traded_pct", "avg_win_pct",
                                               "avg_loss_pct", "by_year_pct")},
             "long_share_pct": long_share,
             "first_half_pct": halves[0].get("total_return_pct"), "second_half_pct": halves[1].get("total_return_pct"),
             "rules": asdict(r), "schedule": asdict(sched), "cost_per_fill": cost})
    log({"event": "DIP_SCREEN_DONE", "passed": [m for m, ok in verdicts.items() if ok],
         "failed": [m for m, ok in verdicts.items() if not ok]})
    return verdicts
