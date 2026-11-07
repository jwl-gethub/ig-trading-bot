"""Can your account size each market correctly? Run this BEFORE anything else.

With small capital the binding constraint is IG's minimum stake. The bot never
rounds a position up to the minimum — if min_size x stop distance would risk more
than risk_per_trade_pct of equity, the trade is skipped. This report shows, per
market, the risk at minimum stake and the equity needed to trade it.

    python -m bot.feasibility                         # IG DEMO account (needs .env)
    python -m bot.feasibility --equity 1500           # override equity
    python -m bot.feasibility --data data/history --equity 1500   # offline from CSVs
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .config import load_config
from .indicators import atr


def feasibility_row(name: str, equity: float, risk_pct: float, min_size: float,
                    stop_points: float, point_value: float, spread: float | None = None) -> dict:
    risk_at_min = min_size * stop_points * point_value
    pct_at_min = 100 * risk_at_min / equity if equity else float("inf")
    need = risk_at_min / (risk_pct / 100)
    return {
        "market": name,
        "stop_pts": round(stop_points, 1),
        "min_size": min_size,
        "risk_at_min_£": round(risk_at_min, 2),
        "risk_at_min_%": round(pct_at_min, 2),
        "equity_needed_£": round(need, 0),
        "spread_%_of_stop": round(100 * spread / stop_points, 1) if spread else None,
        "verdict": "OK" if pct_at_min <= risk_pct else "SKIP (too small)",
    }


def report(cfg: dict, equity: float, bars_by_market: dict[str, pd.DataFrame],
           min_sizes: dict[str, float], spreads: dict[str, float] | None = None) -> pd.DataFrame:
    rp = cfg["risk"]["risk_per_trade_pct"]
    t, m = cfg["strategy"]["trend"], cfg["strategy"]["meanrev"]
    rows = []
    for mk in cfg["markets"]:
        df = bars_by_market.get(mk["name"])
        if df is None or len(df) < 20:
            continue
        a = float(atr(df, cfg["strategy"]["volatility"]["atr_len"]).iloc[-1])
        stop = min(t["stop_atr"], m["stop_atr"] if "meanrev" in mk["modules"] else t["stop_atr"]) * a
        rows.append(feasibility_row(mk["name"], equity, rp, min_sizes.get(mk["name"], mk["min_size"]),
                                    stop, mk["point_value"], (spreads or {}).get(mk["name"],
                                                                                 mk["backtest_spread_points"])))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--equity", type=float)
    ap.add_argument("--data", help="offline: directory of <MARKET>.csv")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.data:
        from .data import load_dir
        bars = load_dir(args.data, cfg["markets"])
        bars = {k: v.tail(200) for k, v in bars.items()}
        equity = args.equity or cfg["backtest"]["initial_equity"]
        mins, spreads = {}, {}
    else:
        from .ig_client import IGClient
        from .live import load_dotenv
        load_dotenv()
        client = IGClient("DEMO", os.environ["IG_API_KEY"], os.environ["IG_USERNAME"],
                          os.environ["IG_PASSWORD"], os.environ.get("IG_ACCOUNT_ID"))
        client.login()
        equity = args.equity or client.equity()
        bars, mins, spreads = {}, {}, {}
        for mk in cfg["markets"]:
            info = client.market_info(mk["epic"])
            mins[mk["name"]], spreads[mk["name"]] = info.min_size, info.spread
            bars[mk["name"]] = client.bars(mk["epic"], cfg["timeframe"], 40)  # small: saves allowance

    df = report(cfg, equity, bars, mins, spreads)
    pd.set_option("display.width", 160)
    print(f"\nEquity £{equity:,.0f} | risk per trade {cfg['risk']['risk_per_trade_pct']}%\n")
    print(df.to_string(index=False))
    print("\nTrade only markets marked OK. 'spread_%_of_stop' above ~5% means costs eat the edge.")


if __name__ == "__main__":
    main()
