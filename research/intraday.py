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

from .dukascopy import load_range
from .noise_area import Params, backtest, buy_and_hold, metrics, session_matrix

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


def run(markets: list[str], cache: Path) -> None:
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
        for vname, p in vs.items():
            trades, daily = backtest(sm, p)
            for per, (a, b) in PERIODS.items():
                m = metrics(trades, daily, a, b)
                log({"event": "RESULT", "market": name, "variant": vname, "period": per,
                     "buy_and_hold_pct": buy_and_hold(sm, a, b), **m,
                     "params": asdict(p) if per == "full" else None})


def main() -> None:
    cache = Path(os.environ.get("RESEARCH_CACHE", "/data/duka"))
    markets = [m for m in os.environ.get("RESEARCH_MARKETS", ",".join(MARKETS)).split(",") if m]
    log({"event": "START", "markets": markets, "cache": str(cache)})
    try:
        run(markets, cache)
    except Exception as exc:
        log({"event": "ERROR", "error": repr(exc)})
        raise
    log({"event": "DONE"})
    sys.stdout.flush()


if __name__ == "__main__":
    main()
