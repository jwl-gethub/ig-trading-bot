"""Kronos in SHADOW mode: forecast, record, score. Never trades on it.

Every time a new 4H bar closes, Kronos forecasts each market ~24h ahead (6 bars). The
forecast is appended to logs/kronos.jsonl and, if a trade opened on that bar, attached to
the trade's features (kronos_exp_ret_pct, kronos_agree) so the learning loop can compare
trades Kronos agreed with against trades it didn't.

`evaluate()` scores past forecasts against what price actually did: directional hit rate
with a confidence range, and rank correlation (IC). Verdicts need 200+ scored forecasts.
Only if Kronos shows real skill does it become a candidate filter, via the strategy-change
protocol and human approval (AGENTS.md guardrails 9 and 15).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "enabled": False,
    "model": "NeoQuasar/Kronos-small",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "lookback_bars": 400,
    "horizon_bars": 6,
    "samples": 5,
    "timeout_s": 300,
    "min_forecasts_for_verdict": 200,
}


class KronosError(RuntimeError):
    pass


def settings(cfg: dict) -> dict:
    k = {**DEFAULTS, **(cfg.get("kronos") or {})}
    env = os.environ.get("BOT_KRONOS", "").strip().lower()
    if env:
        k["enabled"] = env in ("1", "true", "yes", "on")
    return k


class KronosShadow:
    def __init__(self, cfg: dict, log_dir: Path, state_dir: Path):
        self.k = settings(cfg)
        self.enabled = bool(self.k["enabled"])
        self.bar_hours = float(cfg.get("bar_hours", 4))
        self.log_path = Path(log_dir) / "kronos.jsonl"
        self.hf_home = os.environ.get("HF_HOME") or str(Path(state_dir) / "hf")
        self.cmd = [sys.executable, "-m", "bot.kronos_worker"]   # tests swap in a fake worker

    def forecast(self, bars: dict[str, pd.DataFrame]) -> dict:
        """Forecast every market given. Raises KronosError on any worker failure."""
        if not self.enabled or not bars:
            return {}
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in"
            inp.mkdir()
            for name, df in bars.items():
                if len(df) >= 50:
                    df.tail(self.k["lookback_bars"])[["open", "high", "low", "close"]].to_csv(inp / f"{name}.csv")
            out = Path(tmp) / "out.json"
            args = ["--inputs", str(inp), "--out", str(out), "--model", self.k["model"],
                    "--tokenizer", self.k["tokenizer"], "--pred-len", str(self.k["horizon_bars"]),
                    "--samples", str(self.k["samples"]), "--lookback", str(self.k["lookback_bars"]),
                    "--bar-hours", str(self.bar_hours)]
            env = {**os.environ, "HF_HOME": self.hf_home, "TOKENIZERS_PARALLELISM": "false"}
            try:
                proc = subprocess.run(self.cmd + args, capture_output=True, text=True,
                                      timeout=self.k["timeout_s"], env=env)
            except subprocess.TimeoutExpired as exc:
                raise KronosError(f"timeout after {self.k['timeout_s']}s") from exc
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                raise KronosError(f"worker exit {proc.returncode}: {' | '.join(tail)}")
            data = json.loads(out.read_text())
        results = data.get("results", {})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as fh:
            for name, r in results.items():
                fh.write(json.dumps({"market": name, "horizon_bars": self.k["horizon_bars"],
                                     "model": self.k["model"], **r}) + "\n")
        return {"seconds": data.get("seconds"), "load_seconds": data.get("load_seconds"), "results": results}

    # ---------------------------------------------------------------- scoring
    def evaluate(self, bars: dict[str, pd.DataFrame], confidence_z: float = 1.6449) -> dict:
        return evaluate_forecasts(self.log_path, bars, self.k["min_forecasts_for_verdict"], confidence_z)


def evaluate_forecasts(log_path: Path, bars: dict[str, pd.DataFrame], min_n: int = 200,
                       z: float = 1.6449) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        return {"scored": 0, "verdict": "no forecasts yet"}
    rows = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        f = json.loads(line)
        if "error" in f or f.get("market") not in bars:
            continue
        df = bars[f["market"]]
        t = pd.Timestamp(f["bar"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        if t not in df.index:
            continue
        i = df.index.get_loc(t) + int(f.get("horizon_bars", 6))
        if i >= len(df):
            continue                                   # outcome not known yet
        real = 100 * (float(df["close"].iloc[i]) / float(f["last_close"]) - 1)
        rows.append({"market": f["market"], "bar": str(t), "pred": float(f["exp_ret_pct"]), "real": real})
    rows = list({(r["market"], r["bar"]): r for r in reversed(rows)}.values())   # one per market+bar
    n = len(rows)
    if n == 0:
        return {"scored": 0, "verdict": "no forecasts old enough to score yet"}
    d = pd.DataFrame(rows)
    d = d[(d["pred"] != 0) & (d["real"] != 0)]
    hits = (np.sign(d["pred"]) == np.sign(d["real"])).astype(float)
    p = float(hits.mean()) if len(hits) else float("nan")
    half = z * math.sqrt(p * (1 - p) / len(hits)) if len(hits) else float("nan")
    varied = len(d) > 2 and d["pred"].nunique() > 1 and d["real"].nunique() > 1
    ic = float(d["pred"].rank().corr(d["real"].rank())) if varied else float("nan")
    up, down = d[d["pred"] > 0]["real"], d[d["pred"] < 0]["real"]
    out = {
        "scored": int(len(d)),
        "hit_rate_pct": round(100 * p, 1),
        "hit_rate_range_pct": [round(100 * (p - half), 1), round(100 * (p + half), 1)],
        "rank_ic": round(ic, 3) if ic == ic else None,
        "avg_move_when_kronos_up_pct": round(float(up.mean()), 3) if len(up) else None,
        "avg_move_when_kronos_down_pct": round(float(down.mean()), 3) if len(down) else None,
        "by_market": {m: {"n": int(len(g)), "hit_rate_pct": round(100 * float(
            (np.sign(g["pred"]) == np.sign(g["real"])).mean()), 1)} for m, g in d.groupby("market")},
    }
    if len(d) < min_n:
        out["verdict"] = f"not enough data ({len(d)}/{min_n} scored forecasts)"
    elif p - half > 0.5:
        out["verdict"] = "SKILL: right more often than chance. Candidate filter; test via backtest + approval."
    elif p + half < 0.5:
        out["verdict"] = "WORSE than chance on these markets."
    else:
        out["verdict"] = "NO EDGE: indistinguishable from a coin flip so far."
    return out
