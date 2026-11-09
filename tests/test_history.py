import copy
import json

import pandas as pd

from bot.history import HistoryDownloader
from bot.live import LiveRunner

H = pd.Timedelta(hours=1)


def _setup(cfg, make_broker, tmp_path, n_bars=1500, **hist):
    cfg = copy.deepcopy(cfg)
    cfg["markets"] = [m for m in cfg["markets"] if m["name"] in ("EURUSD", "UK100")]
    cfg["history"].update(enabled=True, markets=["EURUSD", "UK100"], target_days=150, chunk_days=30,
                          reserve_points=500, backtest_equity=[10000], **hist)
    broker = make_broker(cfg, n_bars=n_bars, equity=10_000)
    first = next(iter(broker._bars.values()))
    now = first.index[-1] + 4 * H + pd.Timedelta(minutes=1)
    broker.now = now
    events = []
    dl = HistoryDownloader(cfg, broker, tmp_path / "state", tmp_path / "logs",
                           lambda kind, **f: events.append({"event": kind, **f}))
    return cfg, broker, dl, now, events


def test_downloads_backwards_in_chunks_until_target_then_backtests_once(cfg, make_broker, tmp_path):
    cfg, broker, dl, now, events = _setup(cfg, make_broker, tmp_path)
    while dl.step(now):
        pass
    assert dl.complete() == ["EURUSD", "UK100"]
    for name in ("EURUSD", "UK100"):
        df = dl.load(name)
        assert df.index[0] <= now - pd.Timedelta(days=150) + pd.Timedelta(hours=4)
        assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
        assert (df.index + 4 * H <= now).all()                      # closed bars only
    assert 5 <= len(broker.range_calls) <= 14                       # ~30-day chunks, not one huge call
    rep = dl.maybe_backtest()
    assert rep and "10000" in rep and "promotion" in rep["10000"]
    assert any(e["event"] == "BACKTEST_REPORT" for e in events)
    assert dl.maybe_backtest() is None                              # same data -> not re-run
    assert json.loads((tmp_path / "logs" / "backtest_report.json").read_text())


def test_keeps_allowance_reserve_for_live_trading(cfg, make_broker, tmp_path):
    cfg, broker, dl, now, events = _setup(cfg, make_broker, tmp_path)
    broker.allowance_remaining = 600            # 600 - one 30-day chunk (~180) < 500 reserve
    assert dl.step(now) is False and broker.range_calls == []
    broker.allowance_remaining = None           # unknown -> wait
    assert dl.step(now) is False
    broker.allowance_remaining = 5000
    assert dl.step(now) is True


def test_resumes_after_restart(cfg, make_broker, tmp_path):
    cfg, broker, dl, now, _ = _setup(cfg, make_broker, tmp_path)
    dl.step(now)
    dl.step(now)
    first = dl.load("EURUSD").index[0]
    dl2 = HistoryDownloader(cfg, broker, tmp_path / "state", tmp_path / "logs", lambda *a, **k: None)
    dl2.step(now)
    assert dl2.load("EURUSD").index[0] < first                      # continued further back


def _runner(cfg, broker, tmp_path, now):
    cfg = copy.deepcopy(cfg)
    cfg["safety"]["kill_file"] = str(tmp_path / "KILL")
    cfg["safety"]["pause_file"] = str(tmp_path / "PAUSE")
    r = LiveRunner(cfg, broker, tmp_path / "state", tmp_path / "logs", dry_run=True)
    r.startup(now)
    return r


def test_live_runner_downloads_only_when_idle_and_never_breaks_trading(cfg, make_broker, tmp_path):
    cfg, broker, _, _, _ = _setup(cfg, make_broker, tmp_path)
    first = next(iter(broker._bars.values()))
    now = first.index[1200] + 4 * H + pd.Timedelta(minutes=1)
    broker.now = now
    r = _runner(cfg, broker, tmp_path, now)
    r.tick(now)                                                      # processes the just-closed bar
    broker.range_calls = []
    t = now + pd.Timedelta(minutes=5)
    broker.now = t
    r.tick(t)
    assert len(broker.range_calls) == 1                              # idle: one chunk per tick
    t = now + 3 * H + pd.Timedelta(minutes=55)                       # next bar due in ~5 min
    broker.now = t
    r.tick(t)
    assert len(broker.range_calls) == 1                              # stays out of the way

    def boom(*a, **k):
        raise RuntimeError("IG said no")
    broker.bars_range = boom
    t = now + H
    broker.now = t
    assert r.tick(t) is True and r.history.disabled                  # trading loop carries on
    events = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert any(e["event"] == "HISTORY_ERROR" for e in events)
