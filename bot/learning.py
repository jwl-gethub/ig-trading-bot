"""Learning loop: measure what works, report it, and switch OFF what provably doesn't.

Design rules (see AGENTS.md guardrail 15):
  * Every closed trade is journalled with its entry context (features) and outcome
    (R multiple, max favourable / adverse excursion in R, exit reason).
  * The weekly review groups trades into segments and reports average R with a
    confidence interval. Nothing is called a finding until a segment has enough trades
    AND its interval excludes zero; fewer trades = "not enough data", never a verdict.
  * The ONLY automatic action is risk-reducing: a market+module segment whose interval
    sits entirely below zero is disabled. The loop can never enable anything, raise size,
    or change strategy parameters; those become written suggestions for a human.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}
DEFAULTS = {
    "enabled": True,
    "confidence": 0.90,
    "min_trades_for_findings": 30,
    "min_trades_to_disable": 30,
    "auto_disable_losing_segments": True,
    "disable_confidence": 0.95,   # stricter than reporting: many segments are tested at once
    "sentiment_crowd_threshold": 60,
}
SEGMENTS = [
    ("sleeve",), ("segment",), ("market",), ("module",), ("regime",), ("score",), ("crowd",), ("kronos",),
    ("session",), ("weekday",), ("exit_reason",), ("module", "crowd"),
]


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("learning") or {})}


# ---------------------------------------------------------------- journal
def journal_record(pos, exit_px: float, exit_ts, reason: str, pnl: float) -> dict:
    """One closed trade, with everything needed to learn from it."""
    risk_pts = pos.initial_risk_points or float("nan")
    move = (exit_px - pos.entry_price) * pos.direction
    mfe = max(pos.mfe_pts, move, 0.0)
    mae = max(pos.mae_pts, -move, 0.0)
    init_risk = risk_pts * pos.size * pos.point_value
    return {
        "sleeve": getattr(pos, "sleeve", ""), "market": pos.market, "module": pos.module,
        "direction": pos.direction,
        "segment": f"{pos.market}:{pos.module}",
        "entry_ts": str(pos.entry_ts), "exit_ts": str(exit_ts),
        "entry": round(pos.entry_price, 5), "exit": round(exit_px, 5), "size": pos.size,
        "pnl": round(pnl, 2), "r": round(pnl / init_risk, 3) if init_risk else None,
        "mfe_r": round(mfe / risk_pts, 3), "mae_r": round(mae / risk_pts, 3),
        "bars_held": pos.bars_held, "exit_reason": reason, "deal_id": pos.deal_id,
        "features": pos.features,
    }


def append_journal(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def load_journal(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis
def _session(hour) -> str:
    if hour is None or (isinstance(hour, float) and math.isnan(hour)):
        return "unknown"
    h = int(hour)
    if h < 7:
        return "asia"
    if h < 12:
        return "london"
    if h < 16:
        return "london_ny_overlap"
    return "ny_late"


def _crowd(direction: int, long_pct, threshold: float) -> str:
    if long_pct is None or (isinstance(long_pct, float) and math.isnan(long_pct)):
        return "unknown"
    crowd_long = long_pct >= threshold
    crowd_short = long_pct <= 100 - threshold
    if not (crowd_long or crowd_short):
        return "crowd_split"
    with_crowd = (direction == 1 and crowd_long) or (direction == -1 and crowd_short)
    return "with_crowd" if with_crowd else "against_crowd"


def enrich(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "sleeve" not in df:
        df["sleeve"] = ""
    df["sleeve"] = df["sleeve"].fillna("")
    feats = df["features"].apply(lambda f: f if isinstance(f, dict) else {})
    for key in ("regime", "score", "hour_utc", "weekday", "crowd_long_pct"):
        df[key] = feats.apply(lambda f, k=key: f.get(k))
    df["session"] = df["hour_utc"].apply(_session)
    df["crowd"] = [_crowd(d, p, s["sentiment_crowd_threshold"]) for d, p in zip(df["direction"], df["crowd_long_pct"])]
    agree = feats.apply(lambda f: f.get("kronos_agree"))
    df["kronos"] = agree.map({True: "kronos_agreed", False: "kronos_disagreed"}).fillna("no_forecast")
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    return df.dropna(subset=["r"])


def stats(r: pd.Series, confidence: float) -> dict:
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = int(len(r))
    if n == 0:
        return {"n": 0}
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    half = Z.get(confidence, 1.6449) * sd / math.sqrt(n) if n > 1 and sd == sd else float("nan")
    gross_w, gross_l = float(r[r > 0].sum()), float(-r[r <= 0].sum())
    return {
        "n": n, "avg_r": round(mean, 3), "total_r": round(float(r.sum()), 2),
        "win_rate": round(100 * float((r > 0).mean()), 1),
        "ci_low": round(mean - half, 3) if half == half else None,
        "ci_high": round(mean + half, 3) if half == half else None,
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
    }


def review(journal: pd.DataFrame, cfg: dict, already_disabled: list[str] | None = None) -> dict:
    s = settings(cfg)
    df = enrich(journal, s)
    conf = s["confidence"]
    n_total = int(len(df))
    out: dict = {"trades": n_total, "confidence": conf, "findings": [], "suggestions": [],
                 "disable": [], "segments": {}}
    from .montecarlo import sleeve_report
    out["montecarlo"] = sleeve_report(df if n_total else None, cfg)
    if n_total == 0:
        out["findings"].append("No closed trades yet.")
        return out

    out["overall"] = stats(df["r"], conf)
    for keys in SEGMENTS:
        name = "+".join(keys)
        rows = []
        for val, grp in df.groupby(list(keys), dropna=False):
            label = val if isinstance(val, str) else "|".join(str(v) for v in np.atleast_1d(val))
            rows.append({"value": label, **stats(grp["r"], conf)})
        out["segments"][name] = sorted(rows, key=lambda x: -x["n"])

    need = s["min_trades_for_findings"]
    if n_total < need:
        out["findings"].append(
            f"Only {n_total} closed trades; need {need}+ before any segment result means anything.")

    # statistically clear segments (CI excludes zero, enough trades)
    for name, rows in out["segments"].items():
        for row in rows:
            if row.get("n", 0) < need or row.get("ci_low") is None:
                continue
            if row["ci_low"] > 0:
                out["findings"].append(
                    f"{name}={row['value']}: working, avg {row['avg_r']:+.2f}R over {row['n']} trades "
                    f"({int(conf * 100)}% range {row['ci_low']:+.2f} to {row['ci_high']:+.2f}).")
            elif row["ci_high"] < 0:
                out["findings"].append(
                    f"{name}={row['value']}: losing, avg {row['avg_r']:+.2f}R over {row['n']} trades "
                    f"({int(conf * 100)}% range {row['ci_low']:+.2f} to {row['ci_high']:+.2f}).")

    # exit efficiency: how much open profit is given back, how often losers were winners first
    trend = df[df["module"] == "trend"]
    if len(trend) >= need:
        giveback = float((trend["mfe_r"] - trend["r"]).mean())
        losers = trend[trend["r"] <= 0]
        were_up = float((losers["mfe_r"] >= 1.0).mean()) if len(losers) else 0.0
        out["exits"] = {"trend_avg_giveback_r": round(giveback, 2),
                        "trend_losers_that_reached_1r_pct": round(100 * were_up, 1)}
        if were_up > 0.35:
            out["suggestions"].append(
                f"{100 * were_up:.0f}% of losing trend trades were +1R first. Test (backtest, IS/OOS) "
                f"moving the stop to breakeven at +1R before changing anything.")
    winners = df[df["r"] > 0]
    if len(winners) >= need:
        p80 = float(winners["mae_r"].quantile(0.8))
        if p80 < 0.5:
            out["suggestions"].append(
                f"80% of winners never went more than {p80:.2f}R against you. Stops may be wider than needed; "
                f"test a tighter stop in the backtest before changing it.")

    # crowd positioning (from IG client sentiment): suggestion only, never automatic
    crowd = {r["value"]: r for r in out["segments"].get("crowd", [])}
    a, w = crowd.get("against_crowd"), crowd.get("with_crowd")
    if a and w and a.get("n", 0) >= need and w.get("n", 0) >= need:
        if a.get("ci_low") is not None and w.get("ci_high") is not None and a["ci_low"] > w["ci_high"]:
            out["suggestions"].append(
                "Trades against the crowd clearly beat trades with the crowd. Candidate filter: skip "
                "signals where most IG clients are already on the same side (needs your approval).")

    # the only automatic action: switch OFF clearly losing market+module segments
    if s["auto_disable_losing_segments"]:
        disabled = set(already_disabled or [])
        for seg, grp in df.groupby("segment"):
            st = stats(grp["r"], s["disable_confidence"])
            if (st.get("n", 0) >= s["min_trades_to_disable"] and st.get("ci_high") is not None
                    and st["ci_high"] < 0 and seg not in disabled):
                out["disable"].append(seg)
    return out
