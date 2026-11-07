import copy
import json

import pandas as pd
import pytest

from bot.ig_client import IGClient, SafetyError
from bot.live import LiveRunner, apply_live_risk_cap, assert_order_safe, resolve_environment


def test_live_requires_triple_lock(cfg):
    assert resolve_environment(cfg, False, None) == "DEMO"
    with pytest.raises(SafetyError):
        resolve_environment(cfg, True, "LIVE")             # config flag still false
    cfg["safety"]["live_trading_confirmed"] = True
    with pytest.raises(SafetyError):
        resolve_environment(cfg, False, "LIVE")            # missing --live
    with pytest.raises(SafetyError):
        resolve_environment(cfg, True, "DEMO")             # --live without IG_ENV=LIVE
    assert resolve_environment(cfg, True, "LIVE") == "LIVE"


def test_live_risk_capped_until_approved(cfg):
    apply_live_risk_cap(cfg, "LIVE")
    assert cfg["risk"]["risk_per_trade_pct"] == cfg["promotion"]["live_initial_risk_per_trade_pct"]


def test_orders_without_stop_are_refused(cfg):
    with pytest.raises(SafetyError):
        assert_order_safe({"size": 1, "stop_distance": 0})
    client = IGClient("DEMO", "k", "u", "p")
    with pytest.raises(SafetyError):
        client.open_position("X", "BUY", 1.0, 0, "DFB", "GBP")   # refused before any network call


def _run(runner, broker, start, n_steps):
    now = start
    for _ in range(n_steps):
        broker.now = now
        if not runner.tick(now):
            return False
        now += pd.Timedelta(hours=4)
    return True


LIVE_TEST_MARKETS = ("EURUSD", "UK100", "GOLD")   # one per correlation group keeps tests fast


def _cfg3(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["markets"] = [m for m in cfg["markets"] if m["name"] in LIVE_TEST_MARKETS]
    return cfg


def _runner(cfg, broker, tmp_path, dry_run):
    cfg = copy.deepcopy(cfg)
    cfg["safety"]["kill_file"] = str(tmp_path / "KILL")
    cfg["safety"]["pause_file"] = str(tmp_path / "PAUSE")
    cfg["safety"]["friday_cutoff_utc"] = None
    first = next(iter(broker._bars.values()))
    start = first.index[400] + pd.Timedelta(hours=4, minutes=1)
    broker.now = start
    r = LiveRunner(cfg, broker, tmp_path / "state", tmp_path / "logs", dry_run=dry_run)
    r.startup(start)
    return r, start


def test_dry_run_sends_no_orders(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=True)
    _run(r, broker, start, 300)
    assert broker.orders == []
    events = [json.loads(x)["event"] for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert "WARMUP" in events


def test_demo_orders_always_carry_stops_and_respect_limits(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 380)
    assert broker.orders, "expected some demo orders on regime data"
    assert all(o["stop_distance"] > 0 for o in broker.orders)
    assert len(r.positions) <= cfg["risk"]["max_open_positions"]
    # stops only ever move in the trade's favour
    assert broker.stop_updates, "expected trailing-stop updates"
    for _, old, new, direction in broker.stop_updates:
        if old is not None:
            assert (new > old) if direction == "BUY" else (new < old)
    state = json.loads((tmp_path / "state" / "state.json").read_text())
    assert "risk" in state and "positions" in state


def test_kill_file_flattens_and_halts(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 380)
    (tmp_path / "KILL").write_text("stop")
    broker.now = start + pd.Timedelta(days=200)
    assert r.tick(broker.now) is False
    assert r.risk.state.halted
    assert r.positions == {}
    assert not [d for d in broker.ig_positions if d.startswith("D")]   # no bot positions left


def test_drawdown_triggers_flatten(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 380)
    broker.set_equity(80_000)          # >15% below peak
    alive = _run(r, broker, start + pd.Timedelta(days=200), 20)
    assert alive is False and r.risk.state.halted and r.positions == {}


def test_unmanaged_positions_are_never_touched(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    broker.ig_positions["MANUAL1"] = dict(deal_id="MANUAL1", epic="X", direction="BUY", size=1, level=1,
                                          stop_level=None)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 100)
    assert "MANUAL1" in broker.ig_positions
    assert all(c["deal_id"] != "MANUAL1" for c in broker.closes)
    assert all(u[0] != "MANUAL1" for u in broker.stop_updates)


# ---------------------------------------------------------------- service-mode switches
from bot.live import InstanceLock, apply_env_overrides, main  # noqa: E402
from bot.risk import RiskManager  # noqa: E402


def test_env_kill_switch_flattens_and_halts(cfg, make_broker, tmp_path, monkeypatch):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 380)
    monkeypatch.setenv("BOT_KILL", "1")
    assert r.tick(start + pd.Timedelta(days=200)) is False
    assert r.risk.state.halted and r.risk.state.halt_code and r.positions == {}


def test_env_pause_blocks_new_entries(cfg, make_broker, tmp_path, monkeypatch):
    cfg = _cfg3(cfg)
    monkeypatch.setenv("BOT_PAUSE", "1")
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 200)
    assert broker.orders == []


def test_halt_codes_are_single_use(cfg):
    rm = RiskManager(cfg["risk"], cfg["safety"])
    rm.halt("x")
    first = rm.state.halt_code
    rm.halt("y")                      # already halted: code unchanged
    assert rm.state.halt_code == first and len(first) == 6
    rm.reset_halt()
    assert rm.state.halt_code == ""
    rm.halt("z")
    assert len(rm.state.halt_code) == 6


def test_instance_lock_blocks_second_bot(tmp_path):
    a, b = InstanceLock(tmp_path / "bot.lock"), InstanceLock(tmp_path / "bot.lock")
    assert a.try_acquire()
    assert not b.try_acquire()        # fresh heartbeat from another owner
    b.stale = 0
    assert b.try_acquire()            # stale heartbeat -> takeover allowed


def test_dry_run_env_override(cfg, monkeypatch):
    monkeypatch.setenv("BOT_DRY_RUN", "false")
    apply_env_overrides(cfg)
    assert cfg["safety"]["dry_run"] is False
    monkeypatch.setenv("BOT_DRY_RUN", "true")
    apply_env_overrides(cfg)
    assert cfg["safety"]["dry_run"] is True


def test_main_waits_for_credentials_instead_of_crashing(tmp_path, monkeypatch):
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ENV"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    from tests.conftest import ROOT
    main(["--config", str(ROOT / "config.yaml"), "--state-dir", str(tmp_path / "s"), "--once"])
    assert (tmp_path / "s" / "bot.lock").exists()


def test_startup_logs_feasibility(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=2_000)
    _runner(cfg, broker, tmp_path, dry_run=True)
    lines = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    feas = [x for x in lines if x["event"] == "FEASIBILITY"]
    assert len(feas) == 3 and all("verdict" in f for f in feas)
