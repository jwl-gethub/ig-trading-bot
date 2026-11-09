"""Budgeted download of real IG price history, and a backtest on it. No trading effect.

The live bot calls `step()` only when it is idle (no bar due soon). Each call fetches at
most one chunk (~30 days) for the highest-priority market that is not yet complete, and
only if IG's weekly historical-data allowance stays above `reserve_points` afterwards, so
live trading never runs short of price data (guardrail 11). Downloads resume next week
when the allowance resets.

When the set of fully downloaded markets changes, `maybe_backtest()` runs the exact live
rules (same Strategy, RiskManager and Sleeves via `Backtester`) over them and logs the
report. Raw files live in <state_dir>/history/<MARKET>.csv, in IG quote units.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import Backtester, promotion_check, sleeve_report, split_report
from .config import market_map
from .data import load_csv

DEFAULTS = {
    "enabled": False,
    "markets": [],               # download order (priority)
    "target_days": 365,
    "chunk_days": 30,
    "reserve_points": 1500,      # never let IG's weekly allowance fall below this
    "backtest_equity": [10000],
}
BARS_PER_DAY = 6                 # 4H bars, used only to estimate a chunk's allowance cost


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("history") or {})}


def segment_table(trades) -> list[dict]:
    """Per sleeve/market/module: trades, win rate, average R, total R, profit factor."""
    if not trades:
        return []
    df = pd.DataFrame([{"sleeve": t.sleeve, "market": t.market, "module": t.module,
                        "pnl": t.pnl, "r": t.r_multiple} for t in trades])
    rows = []
    for (sl, mk, mo), g in df.groupby(["sleeve", "market", "module"]):
        win, loss = g.pnl[g.pnl > 0].sum(), -g.pnl[g.pnl <= 0].sum()
        rows.append({"sleeve": sl, "market": mk, "module": mo, "trades": int(len(g)),
                     "win_rate_pct": round(100 * float((g.pnl > 0).mean()), 1),
                     "avg_r": round(float(g.r.mean()), 3), "total_r": round(float(g.r.sum()), 2),
                     "pnl": round(float(g.pnl.sum()), 2),
                     "profit_factor": round(float(win / loss), 2) if loss > 0 else None})
    return sorted(rows, key=lambda x: -x["trades"])


class HistoryDownloader:
    def __init__(self, cfg: dict, broker, state_dir: str | Path, log_dir: str | Path,
                 event: Callable[..., None]):
        self.cfg = cfg
        self.s = settings(cfg)
        self.broker = broker
        self.dir = Path(state_dir) / "history"
        self.log_dir = Path(log_dir)
        self.event = event
        self.mkts = market_map(cfg)
        self.bar_delta = pd.Timedelta(hours=cfg.get("bar_hours", 4))
        self.disabled = not self.s["enabled"] or not hasattr(broker, "bars_range")
        self.status_path = self.dir / "status.json"
        self.status: dict = {}
        if self.status_path.exists():
            try:
                self.status = json.loads(self.status_path.read_text())
            except ValueError:
                self.status = {}

    # ------------------------------------------------------------ files
    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.csv"

    def load(self, name: str) -> pd.DataFrame:
        return load_csv(self._path(name), 1.0) if self._path(name).exists() else pd.DataFrame()

    def _save_status(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.status, indent=2))

    def priority(self) -> list[str]:
        return [m for m in self.s["markets"] if m in self.mkts]

    def complete(self) -> list[str]:
        return [m for m in self.priority() if self.status.get(m, {}).get("complete")]

    def pending(self) -> list[str]:
        return [m for m in self.priority() if not self.status.get(m, {}).get("complete")]

    # ------------------------------------------------------------ download
    def step(self, now: pd.Timestamp) -> bool:
        """Fetch at most one chunk. Returns True if a request was made."""
        if self.disabled:
            return False
        todo = self.pending()
        if not todo:
            return False
        remaining = getattr(self.broker, "allowance_remaining", None)
        cost = int(self.s["chunk_days"]) * BARS_PER_DAY
        if remaining is None or float(remaining) - cost < float(self.s["reserve_points"]):
            return False                      # unknown or too low: wait (resets weekly)
        name = todo[0]
        mkt = self.mkts[name]
        have = self.load(name)
        target_start = now - pd.Timedelta(days=int(self.s["target_days"]))
        # walk backwards from the oldest bar we have; overlap by a day (IG may read times as UK time)
        end = (have.index[0] + pd.Timedelta(days=1)) if len(have) else now
        start = max(end - pd.Timedelta(days=int(self.s["chunk_days"])), target_start)
        fresh = self.broker.bars_range(mkt["epic"], self.cfg["timeframe"], start, end)
        if len(fresh):
            fresh = fresh[fresh.index + self.bar_delta <= now]          # closed bars only
        df = pd.concat([have, fresh]) if len(have) else fresh
        if len(df):
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self.dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._path(name), index_label="timestamp")
        no_more = len(fresh) == 0 or (len(have) and fresh.index[0] >= have.index[0])
        done = start <= target_start or no_more
        self.status[name] = {"complete": bool(done), "bars": int(len(df)),
                             "first": str(df.index[0]) if len(df) else None,
                             "last": str(df.index[-1]) if len(df) else None,
                             "ig_has_no_older_data": bool(no_more and start > target_start)}
        self._save_status()
        self.event("HISTORY_CHUNK", market=name, fetched=int(len(fresh)), bars=int(len(df)),
                   first=self.status[name]["first"], complete=bool(done),
                   allowance_left=getattr(self.broker, "allowance_remaining", None))
        if done:
            self.event("HISTORY_COMPLETE", market=name, **self.status[name])
        return True

    # ------------------------------------------------------------ backtest
    def maybe_backtest(self) -> dict | None:
        """Backtest the fully downloaded markets once each time that set changes."""
        if self.disabled:
            return None
        ready = self.complete()
        key = ",".join(ready)
        if not ready or key == self.status.get("_last_backtest"):
            return None
        data = {m: self.load(m) for m in ready}
        data = {m: df for m, df in data.items() if len(df) > int(self.cfg["warmup_bars"]) + 50}
        if not data:
            return None
        reports = {}
        for i, eq in enumerate(self.s["backtest_equity"]):
            res = Backtester(copy.deepcopy(self.cfg), data, float(eq)).run()
            rep = split_report(res, self.cfg["backtest"]["oos_fraction"])
            rep["sleeves"] = sleeve_report(res)
            rep["signals"] = res.signals
            rep["rejections"] = dict(res.rejections.most_common(10))
            rep["halted"], rep["halt_reason"] = res.halted, res.halt_reason
            rep["promotion"] = promotion_check(rep, self.cfg["promotion"])
            rep["by_segment"] = segment_table(res.trades)
            rep["data"] = {m: {"first": str(df.index[0]), "last": str(df.index[-1]), "bars": int(len(df))}
                           for m, df in data.items()}
            reports[str(int(eq))] = rep
            self.event("BACKTEST_REPORT", equity=float(eq), markets=list(data), **_jsonable(rep))
            if i == 0:                         # trade list for the main run, for later analysis
                for t in res.trades:
                    self.event("BACKTEST_TRADE", equity=float(eq), sleeve=t.sleeve, market=t.market,
                               module=t.module, dir=t.direction, entry_ts=t.entry_ts, exit_ts=t.exit_ts,
                               entry=round(t.entry_price, 2), exit=round(t.exit_price, 2), size=t.size,
                               pnl=round(t.pnl, 2), r=round(t.r_multiple, 3), bars=t.bars_held,
                               reason=t.exit_reason, mfe_r=round(t.mfe_r, 2), mae_r=round(t.mae_r, 2))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "backtest_report.json").write_text(json.dumps(_jsonable(reports), indent=2))
        self.status["_last_backtest"] = key
        self._save_status()
        return reports


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x
