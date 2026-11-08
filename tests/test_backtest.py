import copy

import numpy as np

from bot.backtest import Backtester, compute_metrics, split_report
from bot.data import synthetic_universe


def test_runs_and_reports(cfg):
    data = synthetic_universe(cfg["markets"], 1500, regimes=True)
    res = Backtester(cfg, data, 50_000).run()
    assert len(res.equity) > 1000 and res.trades
    rep = split_report(res, 0.3)
    assert {"full", "in_sample", "out_of_sample"} <= set(rep)
    for t in res.trades:
        assert t.size > 0


def test_small_account_skips_unsizeable(cfg):
    data = synthetic_universe(cfg["markets"], 1200, regimes=True)
    res = Backtester(cfg, data, 300).run()
    assert res.rejections.get("UNSIZEABLE", 0) > 0
    assert len(res.trades) == 0


def test_costs_reduce_pnl(cfg):
    data = synthetic_universe(cfg["markets"], 1500, regimes=True)
    with_cost = Backtester(cfg, data, 50_000).run()
    free = copy.deepcopy(cfg)
    for m in free["markets"]:
        m["backtest_spread_points"] = 0.0
        m["funding_long_pct"] = m["funding_short_pct"] = 0.0
    no_cost = Backtester(free, data, 50_000).run()
    assert sum(t.pnl for t in with_cost.trades) < sum(t.pnl for t in no_cost.trades)


def test_no_edge_on_random_walk(cfg):
    """On pure random walks there is no edge. A profitable result here would mean lookahead bias."""
    cfg = copy.deepcopy(cfg)
    cfg["risk"]["max_drawdown_halt_pct"] = 25
    rs = []
    for seed in range(6):
        data = synthetic_universe(cfg["markets"], 3000, seed=500 + seed * 10, regimes=False)
        rs += [t.r_multiple for t in Backtester(cfg, data, 1_000_000).run().trades]
    r = np.array(rs)
    assert len(r) > 200
    assert r.mean() < 0.05, f"suspicious edge on random data: avg R {r.mean():.3f}"


def test_drawdown_halt_stops_trading(cfg):
    cfg = copy.deepcopy(cfg)
    data = synthetic_universe(cfg["markets"], 6000, seed=7, regimes=False)
    res = Backtester(cfg, data, 20_000).run()
    if res.halted:
        last_entry = max(t.entry_ts for t in res.trades)
        halt_exits = [t for t in res.trades if t.exit_reason == "halt"]
        assert all(t.exit_ts >= last_entry for t in halt_exits)
        assert compute_metrics(res.equity, res.trades, 20_000)["max_drawdown_pct"] < 25


def test_backtest_trades_carry_learning_features(cfg):
    import pandas as pd
    from bot.learning import review
    data = synthetic_universe(cfg["markets"], 1500, regimes=True)
    res = Backtester(cfg, data, 50_000).run()
    t = res.trades[0]
    assert t.features and "score" in t.features and t.mfe_r >= 0 and t.mae_r >= 0
    j = pd.DataFrame([{**t.__dict__, "r": t.r_multiple, "segment": f"{t.market}:{t.module}"} for t in res.trades])
    assert review(j, cfg)["trades"] == len(res.trades)


def test_sweep_trades_use_targets(cfg):
    import copy
    cfg = copy.deepcopy(cfg)
    for m in cfg["markets"]:
        m["modules"] = ["sweep"]
    data = synthetic_universe(cfg["markets"], 3000, regimes=True)
    res = Backtester(cfg, data, 100_000).run()
    reasons = {t.exit_reason for t in res.trades}
    assert res.trades and "target" in reasons and {t.module for t in res.trades} == {"sweep"}
    for t in res.trades:
        if t.exit_reason == "target":
            assert 1.5 < t.r_multiple < 2.1        # ~2R minus costs/funding


def test_sweep_has_no_edge_on_random_walk(cfg):
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["risk"]["max_drawdown_halt_pct"] = 25
    for m in cfg["markets"]:
        m["modules"] = ["sweep"]
    rs = []
    for seed in range(6):
        data = synthetic_universe(cfg["markets"], 3000, seed=900 + seed * 10, regimes=False)
        rs += [t.r_multiple for t in Backtester(cfg, data, 1_000_000).run().trades]
    r = np.array(rs)
    assert len(r) > 150
    assert r.mean() < 0.05, f"suspicious edge on random data: avg R {r.mean():.3f}"
