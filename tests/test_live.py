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
    before = {n: sp["risk_per_trade_pct"] for n, sp in cfg["sleeves"].items()}
    apply_live_risk_cap(cfg, "LIVE")
    assert cfg["risk"]["risk_per_trade_pct"] == cfg["promotion"]["live_initial_risk_per_trade_pct"]
    for n, sp in cfg["sleeves"].items():                       # sleeves halved too (1% -> 0.5%)
        assert sp["risk_per_trade_pct"] == before[n] / 2
    assert cfg["sleeves"]["trend"]["min_size_max_risk_pct"] == 1.0   # min-stake ceiling halved too


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
    c = InstanceLock(tmp_path / "bot.lock")
    b.release()
    assert c.try_acquire()            # released on shutdown -> instant takeover


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
    assert len(feas) >= 3 and all("verdict" in f and f["sleeve"] for f in feas)



# ---------------------------------------------------------------- learning loop in live
def test_journal_records_features_and_sentiment(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 380)
    recs = [json.loads(x) for x in (tmp_path / "logs" / "trades.jsonl").read_text().splitlines()]
    assert recs, "expected closed trades in the journal"
    f = recs[0]["features"]
    assert f["crowd_long_pct"] == 65.0 and "score" in f and "hour_utc" in f
    assert recs[0]["mfe_r"] >= 0 and recs[0]["r"] is not None


def test_learned_disabled_segments_block_entries(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    r.risk.state.disabled_segments = [f"{m}:{mod}" for m in LIVE_TEST_MARKETS for mod in ("trend", "meanrev", "sweep")]
    r.risk.state.last_learning_week = "9999-W99"   # keep the weekly review out of this test
    _run(r, broker, start, 250)
    assert broker.orders == []
    events = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert any(e.get("reason") == "LEARNED_DISABLED" for e in events)


def test_weekly_review_runs_once_per_week_and_writes_report(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=True)
    now = pd.Timestamp("2026-11-09 10:00", tz="UTC")
    assert r.maybe_learn(now) is not None
    assert r.maybe_learn(now + pd.Timedelta(days=2)) is None          # same ISO week
    assert r.maybe_learn(now + pd.Timedelta(days=7)) is not None      # next week
    assert (tmp_path / "logs" / "learning_report.json").exists()


def test_sweep_orders_carry_server_side_target(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    for m in cfg["markets"]:
        m["modules"] = ["sweep"]
    broker = make_broker(cfg, n_bars=1400, equity=100_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 900)
    assert broker.orders, "expected sweep orders on regime data"
    assert all(o["limit_distance"] and o["stop_distance"] > 0 for o in broker.orders)
    assert all(abs(o["limit_distance"] / o["stop_distance"] - 2.0) < 0.05 for o in broker.orders)


# ---------------------------------------------------------------- sleeves in live
def test_live_sleeves_trade_independently_and_persist(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, n_bars=1100, equity=10_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    assert {n: st.base for n, st in r.sleeves.states.items()} == {"leveraged": 1000, "trend": 4500, "meanrev": 4500}
    _run(r, broker, start, 600)
    events = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    opened = [e for e in events if e["event"] == "OPENED"]
    assert opened, "expected trades"
    owners = {sn: set(spec["modules"]) for sn, spec in cfg["sleeves"].items()}
    for e in opened:
        assert e["module"] in owners[e["sleeve"]]
        if e["sleeve"] == "leveraged":
            assert e["exposure_x"] <= 20.0 + 1e-6 and e["risk"] <= 0.10 * e["sleeve_equity"] + 1e-6
        else:
            cap = (cfg["sleeves"][e["sleeve"]].get("min_size_max_risk_pct") or 1.0) if e["min_size_trade"] else 1.0
            assert e["exposure_x"] <= 3.0 + 1e-6 and e["risk"] <= cap / 100 * e["sleeve_equity"] + 1e-6
            if e["min_size_trade"]:
                assert e["sleeve"] == "trend" and e["risk"] > 0.01 * e["sleeve_equity"]
    # positions are keyed per sleeve, and sleeve state survives a restart
    assert all(":" in k for k in r.positions)
    r2 = LiveRunner(r.cfg, broker, tmp_path / "state", tmp_path / "logs", dry_run=False)
    assert r2.sleeves.to_dict() == r.sleeves.to_dict() and set(r2.positions) == set(r.positions)



def test_refuses_to_trade_on_wrong_account_type(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, equity=10_000)
    broker.account_type = "CFD"
    first = next(iter(broker._bars.values()))
    broker.now = first.index[400]
    r = LiveRunner(cfg, broker, tmp_path / "state", tmp_path / "logs", dry_run=False)
    with pytest.raises(SafetyError, match="ACCOUNT_TYPE_MISMATCH"):
        r.startup(broker.now)
    assert broker.orders == []


def test_spread_rule_relative_to_stop(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    for m in cfg["markets"]:
        m["max_spread_points"] = None          # only the % rule applies
    broker = make_broker(cfg, equity=100_000, spread=10_000.0)   # absurd spread
    r, start = _runner(cfg, broker, tmp_path, dry_run=False)
    _run(r, broker, start, 200)
    assert broker.orders == []
    events = [json.loads(x) for x in (tmp_path / "logs" / "decisions.jsonl").read_text().splitlines()]
    assert any(str(e.get("reason", "")).startswith("SPREAD_TOO_COSTLY") for e in events)


# ---------------------------------------------------------------- IG historical allowance
def test_restart_mid_bar_waits_for_bar_close_instead_of_polling(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, n_bars=600, equity=10_000)
    r, start = _runner(cfg, broker, tmp_path, dry_run=True)
    calls = []
    orig = broker.bars
    broker.bars = lambda epic, res, n: calls.append(epic) or orig(epic, res, n)
    t = start + pd.Timedelta(hours=1)            # restarted mid-bar; latest closed bar already handled
    broker.now = t
    r.last_bar = {n: str(df.index[-1]) for n, df in r.bars.items()}
    r.next_check = {n: t for n in r.mkts}
    assert r.refresh(t) == [] and len(calls) == len(r.mkts)
    for k in range(1, 12):                       # every 15 min until 3h45m in (bar closes at 3h59m)
        t = start + pd.Timedelta(hours=1) + k * pd.Timedelta(minutes=15)
        broker.now = t
        assert r.refresh(t) == []
    assert len(calls) == len(r.mkts)             # no fetches while the bar is still forming
    t = start + pd.Timedelta(hours=4)            # next bar has closed
    broker.now = t
    assert sorted(r.refresh(t)) == sorted(r.mkts) and len(calls) == 2 * len(r.mkts)


def test_late_bar_retries_briefly_then_closed_market_checks_once_per_bar(cfg, make_broker, tmp_path):
    cfg = _cfg3(cfg)
    broker = make_broker(cfg, n_bars=600, equity=10_000)
    r, _ = _runner(cfg, broker, tmp_path, dry_run=True)
    df = r.bars["EURUSD"]
    last = df.index[-1]
    h = pd.Timedelta(hours=1)
    assert r._next_check_without_new_bar(df, last + 5 * h) == last + 8 * h + pd.Timedelta(seconds=30)
    late = last + 8 * h + pd.Timedelta(minutes=10)
    assert r._next_check_without_new_bar(df, late) == late + pd.Timedelta(minutes=15)
    assert r._next_check_without_new_bar(df, last + 30 * h) == last + 32 * h + pd.Timedelta(seconds=30)
