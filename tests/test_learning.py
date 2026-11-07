import numpy as np
import pandas as pd

from bot.learning import _crowd, review


def _journal(spec, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for seg, n, mu, sd in spec:
        market, module = seg.split(":")
        for r in rng.normal(mu, sd, n):
            rows.append({"segment": seg, "market": market, "module": module, "direction": 1, "r": float(r),
                         "mfe_r": max(float(r), 0) + 0.3, "mae_r": 0.4, "exit_reason": "stop",
                         "features": {"regime": "TREND", "score": 6, "hour_utc": 9, "weekday": 1,
                                      "crowd_long_pct": 70}})
    return pd.DataFrame(rows)


def test_empty_journal(cfg):
    rep = review(pd.DataFrame(), cfg)
    assert rep["trades"] == 0 and rep["disable"] == []


def test_only_clear_losers_with_enough_trades_get_disabled(cfg):
    j = _journal([("A:trend", 40, -0.6, 0.8), ("B:trend", 40, 0.5, 1.0), ("C:meanrev", 10, -1.0, 0.1)])
    rep = review(j, cfg)
    assert rep["disable"] == ["A:trend"]                 # C is losing but has too few trades
    assert any("B" in f and "working" in f for f in rep["findings"])
    assert review(j, cfg, already_disabled=["A:trend"])["disable"] == []


def test_noise_is_not_called_a_finding(cfg):
    j = _journal([("A:trend", 35, 0.0, 1.0)], seed=3)
    rep = review(j, cfg)
    assert rep["disable"] == []
    assert not any("losing" in f or "working" in f for f in rep["findings"] if f.startswith("segment="))


def test_small_sample_warning(cfg):
    rep = review(_journal([("A:trend", 5, -1.0, 0.5)]), cfg)
    assert any("need 30+" in f for f in rep["findings"]) and rep["disable"] == []


def test_crowd_labels():
    assert _crowd(1, 70, 60) == "with_crowd"
    assert _crowd(1, 30, 60) == "against_crowd"
    assert _crowd(-1, 30, 60) == "with_crowd"
    assert _crowd(1, 50, 60) == "crowd_split"
    assert _crowd(1, None, 60) == "unknown"
