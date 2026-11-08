"""Replay ("time machine") risk test: what can losing streaks do to a sleeve?

Takes a sleeve's actual trade results in R (profit / initial risk), reshuffles them
10,000 times into new sequences, and compounds them at the sleeve's risk per trade.
Reports the chance of hitting the sleeve's drawdown stop-out and the spread of
drawdowns and returns. When there are too few real trades, `scenario()` does the same
from an assumed win rate and payoff (e.g. sweep: +2R target vs -1R stop).

Slightly pessimistic on purpose: it ignores the risk-halving after 4 losses in a row.

    python -m bot.montecarlo --risk 10 --dd 50 --win-rate 0.40 --payoff 2.0
    python -m bot.montecarlo --journal logs/trades.jsonl --sleeve leveraged
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _paths(r_draws: np.ndarray, risk_pct: float) -> np.ndarray:
    growth = np.clip(1 + r_draws * risk_pct / 100, 0, None)
    eq = np.cumprod(growth, axis=1)
    return np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)


def _summarise(eq: np.ndarray, dd_limit_pct: float) -> dict:
    peak = np.maximum.accumulate(eq, axis=1)
    dd = 1 - eq / peak
    max_dd = dd.max(axis=1)
    final = eq[:, -1] - 1
    stopped = (dd >= dd_limit_pct / 100).any(axis=1) if dd_limit_pct < 100 else np.zeros(len(eq), bool)
    pct = lambda a, q: round(float(np.percentile(a, q)) * 100, 1)  # noqa: E731
    return {
        "p_stopout_pct": round(100 * float(stopped.mean()), 1),
        "p_losing_pct": round(100 * float((final < 0).mean()), 1),
        "max_drawdown_median_pct": pct(max_dd, 50),
        "max_drawdown_worst5_pct": pct(max_dd, 95),
        "return_median_pct": pct(final, 50),
        "return_worst5_pct": pct(final, 5),
        "return_best5_pct": pct(final, 95),
    }


def bootstrap(r: np.ndarray, risk_pct: float, n_trades: int, dd_limit_pct: float,
              sims: int = 10_000, seed: int = 0) -> dict:
    """Resample the sleeve's own trade results (in R) into `sims` new sequences."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {"error": "no trades"}
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(sims, n_trades), replace=True)
    return {"source": f"{len(r)} real trades", "trades_ahead": n_trades, "sims": sims,
            **_summarise(_paths(draws, risk_pct), dd_limit_pct)}


def scenario(win_rate: float, payoff_r: float, risk_pct: float, n_trades: int, dd_limit_pct: float,
             cost_r: float = 0.05, sims: int = 10_000, seed: int = 0) -> dict:
    """Assumed outcomes: win = +payoff_r, loss = -1R, both minus costs (in R)."""
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, n_trades)) < win_rate
    draws = np.where(wins, payoff_r - cost_r, -1 - cost_r)
    return {"source": f"assumed {win_rate:.0%} win rate, +{payoff_r}R / -1R", "trades_ahead": n_trades,
            "sims": sims, **_summarise(_paths(draws, risk_pct), dd_limit_pct)}


def sleeve_report(journal, cfg: dict, min_trades: int = 20, horizons=(30, 100)) -> dict:
    """Per sleeve: replay real results when there are enough; otherwise a scenario grid
    for sleeves whose strategy has a fixed target (the sweep's 2R)."""
    spec = cfg.get("sleeves") or {}
    out = {}
    for name, s in spec.items():
        risk, dd = float(s["risk_per_trade_pct"]), float(s["max_drawdown_pct"])
        r = np.array([], dtype=float)
        if journal is not None and len(journal) and "sleeve" in journal:
            r = journal.loc[journal["sleeve"] == name, "r"].astype(float).to_numpy()
        if len(r) >= min_trades:
            out[name] = {str(h): bootstrap(r, risk, h, dd) for h in horizons}
            continue
        entry = {"note": f"{len(r)} real trades; need {min_trades}+ to replay real results"}
        if "sweep" in s["modules"]:
            payoff = float(cfg["strategy"]["sweep"]["target_r"])
            entry["scenarios_30_trades"] = {
                f"{int(w * 100)}%_win": scenario(w, payoff, risk, 30, dd) for w in (0.30, 0.35, 0.40, 0.45)
            }
        out[name] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo replay of sleeve risk")
    ap.add_argument("--risk", type=float, default=10.0, help="risk per trade, % of sleeve")
    ap.add_argument("--dd", type=float, default=50.0, help="sleeve stop-out drawdown, %")
    ap.add_argument("--trades", type=int, default=30)
    ap.add_argument("--win-rate", type=float)
    ap.add_argument("--payoff", type=float, default=2.0)
    ap.add_argument("--journal")
    ap.add_argument("--sleeve")
    a = ap.parse_args(argv)
    if a.journal:
        from .learning import load_journal
        j = load_journal(a.journal)
        r = j.loc[j["sleeve"] == a.sleeve, "r"].astype(float).to_numpy() if a.sleeve else j["r"].to_numpy()
        print(json.dumps(bootstrap(r, a.risk, a.trades, a.dd), indent=2))
    else:
        print(json.dumps(scenario(a.win_rate or 0.4, a.payoff, a.risk, a.trades, a.dd), indent=2))


if __name__ == "__main__":
    main()
