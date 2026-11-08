import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from bot.data import synthetic_market
from bot.kronos_shadow import KronosError, KronosShadow, evaluate_forecasts

FAKE = [sys.executable, str(Path(__file__).parent / "fake_kronos_worker.py")]


def shadow(cfg, tmp_path, timeout=20):
    cfg["kronos"].update(enabled=True, timeout_s=timeout)
    k = KronosShadow(cfg, tmp_path / "logs", tmp_path / "state")
    k.cmd = FAKE
    return k


def test_forecast_records_and_scores(cfg, tmp_path):
    k = shadow(cfg, tmp_path)
    df = synthetic_market(1, 400, 1000.0, 0.003)
    res = k.forecast({"EURUSD": df.iloc[:300], "GOLD": df.iloc[:300]})
    assert set(res["results"]) == {"EURUSD", "GOLD"} and res["results"]["EURUSD"]["exp_ret_pct"] == 0.5
    lines = (tmp_path / "logs" / "kronos.jsonl").read_text().splitlines()
    assert len(lines) == 2
    # later bars exist -> forecasts can be scored; duplicates of the same bar count once
    k.forecast({"EURUSD": df.iloc[:300]})
    ev = evaluate_forecasts(k.log_path, {"EURUSD": df, "GOLD": df}, min_n=200)
    assert ev["scored"] == 2 and "not enough data" in ev["verdict"]


def test_worker_crash_and_timeout_raise_cleanly(cfg, tmp_path, monkeypatch):
    k = shadow(cfg, tmp_path, timeout=2)
    df = synthetic_market(2, 200, 1000.0, 0.003)
    monkeypatch.setenv("FAKE_KRONOS_MODE", "crash")
    with pytest.raises(KronosError, match="exit 3"):
        k.forecast({"EURUSD": df})
    monkeypatch.setenv("FAKE_KRONOS_MODE", "hang")
    with pytest.raises(KronosError, match="timeout"):
        k.forecast({"EURUSD": df})


def test_disabled_does_nothing(cfg, tmp_path):
    k = KronosShadow(cfg, tmp_path / "logs", tmp_path / "state")
    assert not k.enabled and k.forecast({"X": pd.DataFrame()}) == {}


def test_live_records_kronos_and_tags_trades_without_changing_them(cfg, make_broker, tmp_path, monkeypatch):
    from tests.test_live import _cfg3, _run, _runner
    cfg = _cfg3(cfg)
    base = make_broker(cfg, equity=100_000)
    r0, start = _runner(cfg, base, tmp_path / "a", dry_run=False)
    _run(r0, base, start, 300)

    cfg["kronos"]["enabled"] = True
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path / "b", dry_run=False)
    r.kronos.cmd = FAKE
    _run(r, broker, start, 300)
    assert len(broker.orders) == len(base.orders)              # shadow mode: identical trading
    ev = [json.loads(x) for x in (tmp_path / "b" / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert any(e["event"] == "KRONOS_FORECAST" for e in ev)
    tagged = [p for p in r.positions.values() if "kronos_agree" in p.features]
    journal = tmp_path / "b" / "logs" / "trades.jsonl"
    closed = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    assert tagged or any("kronos_agree" in c["features"] for c in closed)


def test_live_keeps_trading_when_kronos_breaks(cfg, make_broker, tmp_path, monkeypatch):
    from tests.test_live import _cfg3, _run, _runner
    cfg = _cfg3(cfg)
    cfg["kronos"]["enabled"] = True
    monkeypatch.setenv("FAKE_KRONOS_MODE", "crash")
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    r.kronos.cmd = FAKE
    _run(r, broker, start, 300)
    assert broker.orders
    ev = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert any(e["event"] == "KRONOS_ERROR" for e in ev)
