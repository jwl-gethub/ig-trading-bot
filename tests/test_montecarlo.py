import numpy as np
import pandas as pd

from bot.montecarlo import bootstrap, scenario, sleeve_report


def test_all_wins_never_stop_out_and_all_losses_always_do():
    assert bootstrap(np.ones(30), 10, 30, 50)["p_stopout_pct"] == 0.0
    loss = bootstrap(-np.ones(30), 10, 30, 50)
    assert loss["p_stopout_pct"] == 100.0 and loss["p_losing_pct"] == 100.0
    # 7 straight 10% losses = 0.9^7 = 52% left -> drawdown ~52%
    assert 50 <= bootstrap(-np.ones(7), 10, 7, 100)["max_drawdown_median_pct"] <= 53


def test_better_win_rate_means_lower_stopout_risk():
    p = [scenario(w, 2.0, 10, 30, 50)["p_stopout_pct"] for w in (0.3, 0.4, 0.5)]
    assert p[0] > p[1] > p[2]
    assert scenario(0.4, 2.0, 2, 30, 50)["p_stopout_pct"] < 1.0   # same edge, 2% risk: ~no stop-outs


def test_sleeve_report_uses_real_trades_when_enough(cfg):
    few = pd.DataFrame({"sleeve": ["leveraged"] * 5, "r": [1.9, -1.05, -1.05, 1.9, -1.05]})
    rep = sleeve_report(few, cfg)
    assert "scenarios_30_trades" in rep["leveraged"] and "note" in rep["trend"]
    many = pd.DataFrame({"sleeve": ["trend"] * 40, "r": np.tile([2.5, -1, -1, 0.5], 10)})
    rep = sleeve_report(many, cfg)
    assert {"30", "100"} <= set(rep["trend"]) and rep["trend"]["30"]["source"] == "40 real trades"
