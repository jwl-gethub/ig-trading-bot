"""Portfolio backtester — same Strategy and RiskManager objects as live trading.

Execution model (deliberately conservative):
  * Signals are evaluated on a CLOSED bar; orders fill at the NEXT bar's open.
  * Every fill pays half the spread + slippage (slippage_frac_of_spread x spread).
  * Stops are checked intrabar against high/low. If the bar gaps through the stop,
    the fill is the open (worse than the stop), as in real trading.
  * Overnight funding is charged on notional for the calendar time a position is
    held (weekends included).
  * Stop updates (trailing) decided on bar t apply from bar t+1.

Usage:
  python -m bot.backtest --synthetic                 # smoke test, NOT evidence of edge
  python -m bot.backtest --data data/history         # real CSVs, see bot/data.py
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, market_map
from .data import load_dir, synthetic_universe
from .indicators import compute_all
from .risk import RiskManager
from .learning import journal_record, review
from .strategy import PositionState, Signal, Strategy, entry_features


@dataclass
class Trade:
    market: str
    module: str
    direction: int
    entry_ts: str
    entry_price: float
    exit_ts: str
    exit_price: float
    size: float
    pnl: float
    funding: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    mfe_r: float = 0.0
    mae_r: float = 0.0
    features: dict | None = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    rejections: Counter
    signals: int
    halted: bool
    halt_reason: str
    start_equity: float


def _reason_key(reason: str) -> str:
    return reason.split(":")[0].split(" ")[0]


class Backtester:
    def __init__(self, cfg: dict, data: dict[str, pd.DataFrame], initial_equity: float | None = None):
        self.cfg = cfg
        self.mkts = {k: v for k, v in market_map(cfg).items() if k in data}
        self.strategy = Strategy(cfg["strategy"])
        self.risk = RiskManager(cfg["risk"], cfg["safety"])
        self.start_equity = float(initial_equity or cfg["backtest"]["initial_equity"])
        self.slip_frac = float(cfg["backtest"]["slippage_frac_of_spread"])
        self.bar_delta = pd.Timedelta(hours=cfg.get("bar_hours", 4))
        self.ind = {k: compute_all(df, cfg["strategy"]) for k, df in data.items() if k in self.mkts}
        self.rows = {k: df.to_dict("index") for k, df in self.ind.items()}

    def _cost(self, mkt: dict) -> float:
        spread = float(mkt["backtest_spread_points"])
        return spread / 2 + self.slip_frac * spread

    def run(self) -> BacktestResult:
        cash = self.start_equity
        positions: dict[str, PositionState] = {}
        pending_entry: dict[str, tuple[Signal, PositionState]] = {}
        pending_exit: dict[str, str] = {}
        last_close: dict[str, float] = {}
        last_fund_ts: dict[str, pd.Timestamp] = {}
        trades: list[Trade] = []
        rejections: Counter = Counter()
        n_signals = 0
        eq_points: list[tuple[pd.Timestamp, float]] = []

        def close_trade(name: str, px: float, ts, reason: str) -> None:
            nonlocal cash
            pos = positions.pop(name)
            gross = (px - pos.entry_price) * pos.direction * pos.size * pos.point_value
            pnl = gross - pos.funding
            cash += pnl
            init_risk = pos.initial_risk_points * pos.size * pos.point_value
            rec = journal_record(pos, px, ts, reason, pnl)
            trades.append(Trade(
                market=name, module=pos.module, direction=pos.direction, entry_ts=pos.entry_ts,
                entry_price=round(pos.entry_price, 5), exit_ts=str(ts), exit_price=round(px, 5),
                size=pos.size, pnl=round(pnl, 2), funding=round(pos.funding, 2),
                r_multiple=round(pnl / init_risk, 3) if init_risk else 0.0,
                bars_held=pos.bars_held, exit_reason=reason,
                mfe_r=rec["mfe_r"], mae_r=rec["mae_r"], features=pos.features,
            ))
            self.risk.on_trade_closed(pnl, ts)

        timeline = sorted(set().union(*[set(r.keys()) for r in self.rows.values()]))
        for ts in timeline:
            # ---- 1) fills at this bar's open, then intrabar stops, funding
            for name, rows in self.rows.items():
                row = rows.get(ts)
                if row is None:
                    continue
                mkt = self.mkts[name]
                cost = self._cost(mkt)

                if name in pending_exit and name in positions:
                    pos = positions[name]
                    close_trade(name, row["open"] - pos.direction * cost, ts, pending_exit[name])
                pending_exit.pop(name, None)

                if name in pending_entry and name not in positions:
                    sig, prov = pending_entry[name]
                    fill = row["open"] + sig.direction * cost
                    prov.entry_price = fill
                    prov.stop_level = fill - sig.direction * sig.stop_distance
                    prov.target_level = (fill + sig.direction * sig.target_distance
                                         if sig.target_distance else None)
                    prov.extreme = fill
                    prov.entry_ts = str(ts)
                    positions[name] = prov
                    last_fund_ts[name] = ts
                    self.risk.on_order_sent()
                pending_entry.pop(name, None)

                if name in positions:
                    pos = positions[name]
                    hit = (row["low"] <= pos.stop_level) if pos.direction == 1 else (row["high"] >= pos.stop_level)
                    tgt = pos.target_level
                    tgt_hit = tgt is not None and (
                        row["high"] >= tgt if pos.direction == 1 else row["low"] <= tgt)
                    if hit:   # stop checked first: if both touched in one bar, assume the worse outcome
                        raw = min(row["open"], pos.stop_level) if pos.direction == 1 else max(row["open"], pos.stop_level)
                        close_trade(name, raw - pos.direction * cost, ts, "stop")
                    elif tgt_hit:   # limit order fills at the target, paying half the spread
                        close_trade(name, tgt - pos.direction * mkt["backtest_spread_points"] / 2, ts, "target")
                    else:
                        hours = (ts - last_fund_ts[name]).total_seconds() / 3600
                        rate = mkt["funding_long_pct"] if pos.direction == 1 else mkt["funding_short_pct"]
                        notional = pos.size * row["close"] * pos.point_value
                        pos.funding += notional * rate / 100 * hours / 8760
                        last_fund_ts[name] = ts
                last_close[name] = row["close"]

            # ---- 2) mark to market (at liquidation value)
            unreal = 0.0
            for name, pos in positions.items():
                px = last_close[name] - pos.direction * self._cost(self.mkts[name])
                unreal += (px - pos.entry_price) * pos.direction * pos.size * pos.point_value - pos.funding
            equity = cash + unreal
            decision_ts = ts + self.bar_delta
            self.risk.on_bar(decision_ts, equity)
            eq_points.append((ts, equity))

            if self.risk.state.halted:
                for name in positions:
                    pending_exit[name] = "halt"
                pending_entry.clear()

            # ---- 3) decisions on the closed bar
            for name, rows in self.rows.items():
                row = rows.get(ts)
                if row is None:
                    continue
                mkt = self.mkts[name]
                if name in positions:
                    dec = self.strategy.manage(positions[name], row)
                    if dec.close:
                        pending_exit[name] = dec.reason
                    elif dec.new_stop is not None:
                        positions[name].stop_level = dec.new_stop
                    continue
                if self.risk.state.halted or name in pending_entry:
                    continue
                sig = self.strategy.entry_signal(mkt, row)
                if not sig:
                    continue
                n_signals += 1
                sz = self.risk.size(equity, sig.stop_distance, mkt["point_value"], mkt["min_size"], mkt["size_step"])
                if not sz.ok:
                    rejections[_reason_key(sz.reason)] += 1
                    continue
                margin = sz.size * row["close"] * mkt["point_value"] * mkt["margin_pct"] / 100
                prov = PositionState(
                    market=name, direction=sig.direction, module=sig.module,
                    entry_price=row["close"], stop_level=row["close"] - sig.direction * sig.stop_distance,
                    size=sz.size, point_value=mkt["point_value"], initial_risk_points=sig.stop_distance,
                    entry_ts=str(decision_ts), extreme=row["close"], margin=margin, group=mkt["group"],
                    features=entry_features(sig, row, decision_ts),
                )
                exposures = list(positions.values()) + [p for _, p in pending_entry.values()]
                ok, why = self.risk.check_entry(decision_ts, mkt, exposures, equity, sz.risk_amount, margin)
                if not ok:
                    rejections[_reason_key(why)] += 1
                    continue
                pending_entry[name] = (sig, prov)

        # close anything still open at the final close
        end_ts = timeline[-1] if timeline else None
        for name in list(positions):
            pos = positions[name]
            close_trade(name, last_close[name] - pos.direction * self._cost(self.mkts[name]), end_ts, "end_of_data")
        if eq_points:
            eq_points[-1] = (eq_points[-1][0], cash)

        equity_series = pd.Series({t: v for t, v in eq_points}, name="equity")
        return BacktestResult(trades, equity_series, rejections, n_signals,
                              self.risk.state.halted, self.risk.state.halt_reason, self.start_equity)


# ---------------------------------------------------------------- metrics
def compute_metrics(equity: pd.Series, trades: list[Trade], start_equity: float) -> dict:
    pnl = np.array([t.pnl for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = math.inf if gross_win > 0 else 0.0
    end = float(equity.iloc[-1]) if len(equity) else start_equity
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0
    cagr = (end / start_equity) ** (1 / years) - 1 if years > 0.25 and end > 0 else float("nan")
    daily = equity.resample("1D").last().dropna().pct_change().dropna() if len(equity) > 1 else pd.Series(dtype=float)
    sharpe = float(daily.mean() / daily.std() * math.sqrt(252)) if len(daily) > 5 and daily.std() > 0 else float("nan")
    r = np.array([t.r_multiple for t in trades], dtype=float)
    return {
        "trades": int(len(trades)),
        "net_profit": round(end - start_equity, 2),
        "total_return_pct": round(100 * (end / start_equity - 1), 2),
        "cagr_pct": round(100 * cagr, 2) if not math.isnan(cagr) else None,
        "max_drawdown_pct": round(-100 * dd, 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else None,
        "win_rate_pct": round(100 * len(wins) / len(pnl), 1) if len(pnl) else None,
        "avg_r": round(float(r.mean()), 3) if len(r) else None,
        "sharpe_daily": round(sharpe, 2) if not math.isnan(sharpe) else None,
        "total_funding": round(sum(t.funding for t in trades), 2),
    }


def split_report(res: BacktestResult, oos_fraction: float) -> dict:
    eq = res.equity
    if len(eq) < 10:
        return {"full": compute_metrics(eq, res.trades, res.start_equity)}
    split_ts = eq.index[int(len(eq) * (1 - oos_fraction))]
    is_tr = [t for t in res.trades if pd.Timestamp(t.entry_ts) < split_ts]
    oos_tr = [t for t in res.trades if pd.Timestamp(t.entry_ts) >= split_ts]
    is_eq, oos_eq = eq[eq.index < split_ts], eq[eq.index >= split_ts]
    return {
        "split_at": str(split_ts),
        "full": compute_metrics(eq, res.trades, res.start_equity),
        "in_sample": compute_metrics(is_eq, is_tr, res.start_equity),
        "out_of_sample": compute_metrics(oos_eq, oos_tr, float(is_eq.iloc[-1])),
    }


def promotion_check(report: dict, promo: dict) -> list[dict]:
    full, oos = report["full"], report.get("out_of_sample", report["full"])
    checks = [
        ("trades", full["trades"], ">=", promo["backtest_min_trades"]),
        ("profit_factor", full["profit_factor"] or 0, ">=", promo["backtest_min_profit_factor"]),
        ("max_drawdown_pct", full["max_drawdown_pct"], "<=", promo["backtest_max_drawdown_pct"]),
        ("oos_profit_factor", oos["profit_factor"] or 0, ">=", promo["oos_min_profit_factor"]),
    ]
    out = []
    for name, val, op, thr in checks:
        ok = val >= thr if op == ">=" else val <= thr
        out.append({"check": name, "value": val, "rule": f"{op} {thr}", "pass": bool(ok)})
    return out


def run_cli(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Backtest the IG strategy")
    ap.add_argument("--config", default="config.yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="directory of <MARKET>.csv files")
    src.add_argument("--synthetic", action="store_true",
                     help="random-walk smoke test: a correct engine should LOSE ~costs here")
    src.add_argument("--synthetic-regimes", action="store_true",
                     help="trend/range synthetic data to exercise all code paths (meaningless P&L)")
    ap.add_argument("--bars", type=int, default=3000)
    ap.add_argument("--equity", type=float, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.synthetic or args.synthetic_regimes:
        data = synthetic_universe(cfg["markets"], args.bars, regimes=args.synthetic_regimes)
    else:
        data = load_dir(args.data, cfg["markets"])
    res = Backtester(cfg, data, args.equity).run()
    report = split_report(res, cfg["backtest"]["oos_fraction"])
    report["signals"] = res.signals
    report["rejections"] = dict(res.rejections.most_common())
    report["halted"] = res.halted
    report["halt_reason"] = res.halt_reason
    report["promotion"] = promotion_check(report, cfg["promotion"])
    journal = pd.DataFrame([{**asdict(t), "r": t.r_multiple, "segment": f"{t.market}:{t.module}"}
                            for t in res.trades])
    report["learning"] = review(journal, cfg)
    if args.synthetic:
        report["data_source"] = "SYNTHETIC RANDOM WALK (no edge exists; expect a loss ~= costs)"
    elif args.synthetic_regimes:
        report["data_source"] = "SYNTHETIC REGIMES (code-path test only; P&L is meaningless)"
    else:
        report["data_source"] = str(args.data)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(t) for t in res.trades]).to_csv(out / "trades.csv", index=False)
    res.equity.to_csv(out / "equity.csv", header=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps({k: report[k] for k in ("data_source", "full", "in_sample", "out_of_sample",
                                             "signals", "rejections", "halted", "promotion")
                      if k in report}, indent=2, default=str))
    return report


if __name__ == "__main__":
    run_cli()
