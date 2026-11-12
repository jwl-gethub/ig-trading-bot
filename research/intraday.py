"""Validation job for the intraday noise-area strategy (runs as the Railway 'research' service).

Downloads free 1-minute index history from Dukascopy, runs the published rules with IG costs,
and logs one JSON line per result. Periods: before the paper (2019 to 10 May 2024, overlaps the
authors' own sample) and after it (the honest out-of-sample test). No parameters are tuned.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pandas as pd

from .dip import DipParams, backtest_dip
from .dukascopy import load_range
from .noise_area import Params, backtest, buy_and_hold, last_half_hour, metrics, session_matrix

MARKETS = {   # IG in-hours spread / 2 + slippage per fill, in index points
    "US500": {"inst": "USA500IDXUSD", "range": (1500, 12000), "cost_side": 0.2 + 0.1},
    "USTECH": {"inst": "USATECHIDXUSD", "range": (4000, 40000), "cost_side": 0.5 + 0.25},
    "GER40": {"inst": "DEUIDXEUR", "range": (7000, 40000), "cost_side": 0.6 + 0.3},
    "UK100": {"inst": "GBRIDXGBP", "range": (4000, 15000), "cost_side": 0.5 + 0.25},
}
START, END = date(2018, 11, 1), date(2026, 11, 6)
PERIODS = {"full": ("2019-01-01", None), "before_paper": ("2019-01-01", "2024-05-10"),
           "after_paper": ("2024-05-10", None)}


def log(obj: dict) -> None:
    print("RESEARCH " + json.dumps(obj, default=str), flush=True)


def variants(cost: float) -> dict[str, Params]:
    base = Params(cost_side_points=cost)
    return {
        "paper_rules": base,
        "live_vwap_stop": replace(base, hard_stop_sigma=1.0),
        "live_twap_stop": replace(base, ref="twap", hard_stop_sigma=1.0),
        "double_costs": replace(base, cost_side_points=2 * cost),
        "no_leverage": replace(base, vol_target=False),
    }


def portfolio(dailies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal capital per market, decided in advance (no picking winners): each market's own
    vol-targeted return, averaged across the markets on every trading day."""
    wide = pd.concat({m: d.set_index("day").ret for m, d in dailies.items()}, axis=1).fillna(0.0)
    lev = pd.concat({m: d.set_index("day").lev for m, d in dailies.items()}, axis=1).mean(axis=1)
    return pd.DataFrame({"day": wide.index, "ret": wide.mean(axis=1).to_numpy(), "lev": lev.to_numpy()})


def run(markets: list[str], cache: Path) -> None:
    collected: dict[str, dict] = {}
    for name in markets:
        spec = MARKETS[name]
        t0 = time.time()
        df = load_range(spec["inst"], START, END, cache, spec["range"], log=log)
        sm = session_matrix(df, name)
        vol_share = float((sm["volume"] > 0).mean())
        log({"event": "DATA", "market": name, "rows": int(len(df)), "sessions": int(len(sm["days"])),
             "first": str(sm["days"][0].date()), "last": str(sm["days"][-1].date()),
             "minutes_with_volume_pct": round(100 * vol_share, 1), "seconds": round(time.time() - t0)})
        vs = variants(spec["cost_side"])
        if name != "US500":
            vs = {k: vs[k] for k in ("paper_rules", "live_twap_stop")}
        runs = {k: (lambda p=p: backtest(sm, p)) for k, p in vs.items()}
        base = Params(cost_side_points=spec["cost_side"])
        runs["last30_rest_of_day"] = lambda: last_half_hour(sm, base, "rest_of_day")
        runs["last30_first_half_hour"] = lambda: last_half_hour(sm, base, "first_half_hour")
        runs["last30_rest_of_day_unlevered"] = lambda: last_half_hour(sm, replace(base, vol_target=False))
        if os.environ.get("RESEARCH_ONLY"):
            runs = {k: v for k, v in runs.items() if k.startswith(os.environ["RESEARCH_ONLY"])}
        for vname, fn in runs.items():
            trades, daily = fn()
            p = vs.get(vname, base)
            collected.setdefault(vname, {})[name] = (trades, daily)
            for per, (a, b) in PERIODS.items():
                m = metrics(trades, daily, a, b)
                log({"event": "RESULT", "market": name, "variant": vname, "period": per,
                     "buy_and_hold_pct": buy_and_hold(sm, a, b), **m,
                     "params": asdict(p) if per == "full" else None})
    for vname in ("paper_rules", "live_twap_stop"):
        got = collected.get(vname, {})
        if len(got) < 2:
            continue
        daily = portfolio({m: d for m, (_, d) in got.items()})
        trades = pd.concat([t.assign(acct_ret=t.acct_ret / len(got)) for t, _ in got.values() if len(t)])
        for per, (a, b) in PERIODS.items():
            log({"event": "RESULT", "market": "PORTFOLIO_" + "+".join(got), "variant": vname,
                 "period": per, **metrics(trades, daily, a, b)})


def dip_variants(cost: float) -> dict[str, DipParams]:
    b = DipParams(cost_side_points=cost)
    return {
        "dip_quick_next_open": b,                                   # Connors exit: close > 5-day SMA
        "dip_quick_near_close": replace(b, fill="near_close"),
        "dip_ride_next_open": replace(b, exit="ride"),              # let winners run: 3x ATR trail
        "dip_ride_near_close": replace(b, exit="ride", fill="near_close"),
        "dip_quick_no_stop": replace(b, stop_atr=None),             # Connors original (no stop)
        "dip_quick_rsi10": replace(b, rsi_entry=10.0),              # sensitivity only
        "dip_quick_2x": replace(b, leverage=2.0),                   # what leverage does to it
    }


def run_dip(markets: list[str], cache: Path) -> None:
    for name in markets:
        spec = MARKETS[name]
        sm = session_matrix(load_range(spec["inst"], START, END, cache, spec["range"], log=log), name)
        for vname, p in dip_variants(spec["cost_side"]).items():
            trades, daily = backtest_dip(sm, p)
            extra = {}
            if len(trades):
                extra = {"avg_days_held": round(float(trades.days_held.mean()), 1),
                         "avg_trade_pct": round(100 * float(trades.acct_ret.mean()), 3),
                         "worst_trade_pct": round(100 * float(trades.acct_ret.min()), 2)}
            for per, (a, b) in PERIODS.items():
                m = metrics(trades, daily, a, b)
                log({"event": "RESULT", "market": name, "variant": vname, "period": per,
                     "buy_and_hold_pct": buy_and_hold(sm, a, b), **m, **(extra if per == "full" else {}),
                     "params": asdict(p) if per == "full" else None})


def main() -> None:
    cache = Path(os.environ.get("RESEARCH_CACHE", "/data/duka"))
    markets = [m for m in os.environ.get("RESEARCH_MARKETS", ",".join(MARKETS)).split(",") if m]
    log({"event": "START", "markets": markets, "cache": str(cache)})
    try:
        if os.environ.get("RESEARCH_JOB") == "scalp":
            from .scalp_test import run as run_scalp
            run_scalp(cache)
            log({"event": "DONE"})
            while True:                       # stay up so the service does not restart and re-run
                time.sleep(3600)
        if os.environ.get("RESEARCH_JOB") == "candle":
            from .candle_test import run as run_candle
            run_candle(cache)
            log({"event": "DONE"})
            while True:
                time.sleep(3600)
        if os.environ.get("RESEARCH_JOB") == "candle_audit":
            from .candle_audit import run as run_candle_audit
            run_candle_audit(cache)
            log({"event": "DONE"})
            while True:
                time.sleep(3600)
        if os.environ.get("RESEARCH_JOB") == "structure":
            from .structure_test import run as run_structure
            run_structure(cache)
            log({"event": "DONE"})
            while True:
                time.sleep(3600)
        if os.environ.get("RESEARCH_DIP_SCREEN", "1") == "1":
            from .dip_markets import screen
            screen(str(cache.parent / "hourly"))
        dip_markets = [m for m in os.environ.get("RESEARCH_DIP_MARKETS", "").split(",") if m]
        if dip_markets:
            run_dip(dip_markets, cache)
            log({"event": "DIP_DONE", "markets": dip_markets})
        run(markets, cache)
    except Exception as exc:
        log({"event": "ERROR", "error": repr(exc)})
        raise
    log({"event": "DONE"})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
