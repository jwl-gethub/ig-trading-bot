"""Live / demo trading loop.

    python -m bot.live                 # DEMO, dry_run from config (default: log only)
    python -m bot.live --once          # single cycle, then exit
    python -m bot.live --reset-halt    # human-only: clear a drawdown/kill halt
    IG_ENV=LIVE python -m bot.live --live   # LIVE: also needs safety.live_trading_confirmed: true

Credentials come from environment variables (or a gitignored .env file):
    IG_API_KEY, IG_USERNAME, IG_PASSWORD, optional IG_ACCOUNT_ID, IG_ENV (DEMO|LIVE)

Human switches (env vars, e.g. Railway Variables; changing one redeploys):
    BOT_DRY_RUN=true|false   log-only vs real orders
    BOT_PAUSE=1              no new entries; open trades still managed
    BOT_KILL=1               close the bot's positions and halt
    BOT_RESET_HALT=<code>    clear a halt using the code printed in the logs
    BOT_MARKETS=EURUSD,GOLD  trade only these markets
    STATE_DIR / LOG_DIR      where state, bar cache and logs live (a persistent volume)

Every decision is written to logs/decisions.jsonl; closed trades to logs/trades.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import socket
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

from .config import load_config, market_map
from .ig_client import IGClient, SafetyError
from .feasibility import feasibility_row
from .indicators import compute_all, true_range
from .risk import RiskManager, RiskState
from .strategy import PositionState, Strategy

log = logging.getLogger("bot")
MAX_BARS_KEPT = 1200


# ---------------------------------------------------------------- safety gates
def resolve_environment(cfg: dict, cli_live: bool, env_var: str | None) -> str:
    env = (env_var or "DEMO").strip().upper()
    if env not in ("DEMO", "LIVE"):
        raise SafetyError(f"IG_ENV must be DEMO or LIVE, got {env!r}")
    if env == "LIVE":
        if not cfg["safety"].get("live_trading_confirmed"):
            raise SafetyError("LIVE refused: safety.live_trading_confirmed is false")
        if not cli_live:
            raise SafetyError("LIVE refused: pass --live on the command line")
    elif cli_live:
        raise SafetyError("--live given but IG_ENV is not LIVE")
    return env


def apply_live_risk_cap(cfg: dict, env: str) -> None:
    """First live phase trades at reduced risk unless the human approves full risk."""
    if env == "LIVE" and not cfg["safety"].get("live_full_risk_approved", False):
        cap = cfg["promotion"]["live_initial_risk_per_trade_pct"]
        cfg["risk"]["risk_per_trade_pct"] = min(cfg["risk"]["risk_per_trade_pct"], cap)


def assert_order_safe(order: dict) -> None:
    if not order.get("stop_distance") or order["stop_distance"] <= 0:
        raise SafetyError(f"order without stop: {order}")
    if order.get("size", 0) <= 0:
        raise SafetyError(f"order with non-positive size: {order}")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def apply_env_overrides(cfg: dict) -> None:
    """Human switches that can be flipped from a hosting dashboard (e.g. Railway variables).

    BOT_DRY_RUN=false  -> send real orders (to whichever account IG_ENV points at)
    BOT_DRY_RUN=true   -> log only
    """
    v = os.environ.get("BOT_DRY_RUN", "").strip().lower()
    if v:
        cfg["safety"]["dry_run"] = v in ("1", "true", "yes", "on")


class InstanceLock:
    """Single-instance guard on the state volume, so a redeploy can never run two bots at once."""

    def __init__(self, path: Path, stale_seconds: int = 180):
        self.path, self.stale = Path(path), stale_seconds
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def try_acquire(self) -> bool:
        cur = self._read()
        age = time.time() - float(cur.get("ts", 0))
        if cur and cur.get("owner") != self.owner and age < self.stale:
            return False
        self.beat()
        return True

    def acquire(self, sleep_s: int = 15) -> None:
        while not self.try_acquire():
            log.warning("another bot instance holds %s; waiting", self.path)
            time.sleep(sleep_s)

    def beat(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"owner": self.owner, "ts": time.time()}))

    def release(self) -> None:
        """Mark the lock stale so the next deployment can take over immediately."""
        if self._read().get("owner") == self.owner:
            self.path.write_text(json.dumps({"owner": self.owner, "ts": 0}))


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- runner
class LiveRunner:
    def __init__(self, cfg: dict, broker, state_dir: str | Path = "state", log_dir: str | Path = "logs",
                 dry_run: bool | None = None):
        self.cfg = cfg
        self.broker = broker
        self.mkts = market_map(cfg)
        self.strategy = Strategy(cfg["strategy"])
        self.dry_run = cfg["safety"]["dry_run"] if dry_run is None else dry_run
        self.bar_delta = pd.Timedelta(hours=cfg.get("bar_hours", 4))
        self.state_dir, self.log_dir = Path(state_dir), Path(log_dir)
        (self.state_dir / "bars").mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.positions: dict[str, PositionState] = {}
        self.last_bar: dict[str, str] = {}
        self.next_check: dict[str, pd.Timestamp] = {}
        self.bars: dict[str, pd.DataFrame] = {}
        self.unmanaged_seen: set[str] = set()
        self.risk = RiskManager(cfg["risk"], cfg["safety"])
        self._load_state()

    # ---------------- persistence & logging
    @property
    def _state_file(self) -> Path:
        return self.state_dir / "state.json"

    def _load_state(self) -> None:
        if self._state_file.exists():
            d = json.loads(self._state_file.read_text())
            self.risk.state = RiskState(**d.get("risk", {}))
            self.positions = {k: PositionState(**v) for k, v in d.get("positions", {}).items()}
            self.last_bar = d.get("last_bar", {})

    def save_state(self) -> None:
        d = {"risk": self.risk.state.to_dict(),
             "positions": {k: v.to_dict() for k, v in self.positions.items()},
             "last_bar": self.last_bar}
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self._state_file)

    def event(self, kind: str, **fields) -> None:
        rec = {"ts": pd.Timestamp.now(tz="UTC").isoformat(), "event": kind, **fields}
        with open(self.log_dir / "decisions.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        log.info("%s %s", kind, {k: v for k, v in fields.items() if k != "snapshot"})

    def _journal(self, pos: PositionState, exit_px: float, reason: str, now) -> float:
        pnl = (exit_px - pos.entry_price) * pos.direction * pos.size * pos.point_value
        f = self.log_dir / "trades.csv"
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["market", "module", "direction", "entry_ts", "entry", "exit_ts", "exit",
                            "size", "pnl_est", "reason", "deal_id"])
            w.writerow([pos.market, pos.module, pos.direction, pos.entry_ts, pos.entry_price, str(now),
                        exit_px, pos.size, round(pnl, 2), reason, pos.deal_id])
        self.risk.on_trade_closed(pnl, now)
        return pnl

    # ---------------- bars
    def _cache_path(self, name: str) -> Path:
        return self.state_dir / "bars" / f"{name}.csv"

    def _closed_only(self, df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return df[df.index + self.bar_delta <= now] if len(df) else df

    def warm_up(self, now: pd.Timestamp) -> None:
        need = int(self.cfg["warmup_bars"])
        for name, mkt in self.mkts.items():
            cached = pd.DataFrame()
            if self._cache_path(name).exists():
                cached = pd.read_csv(self._cache_path(name), index_col=0, parse_dates=True)
                cached.index = pd.to_datetime(cached.index, utc=True)
            if len(cached) >= need:
                gap = int((now - cached.index[-1]) / self.bar_delta) + 3
                n = min(need, max(gap, 3))
            else:
                n = need + 5
            fresh = self.broker.bars(mkt["epic"], self.cfg["timeframe"], n)
            df = pd.concat([cached, fresh]) if len(cached) else fresh
            df = df[~df.index.duplicated(keep="last")].sort_index()
            self.bars[name] = self._closed_only(df, now).tail(MAX_BARS_KEPT)
            self.bars[name].to_csv(self._cache_path(name))
            self.next_check[name] = now
            self.event("WARMUP", market=name, bars=len(self.bars[name]), fetched=n)

    def refresh(self, now: pd.Timestamp) -> list[str]:
        """Fetch only when a new bar should have closed. Returns markets with a new closed bar."""
        updated = []
        for name, mkt in self.mkts.items():
            if now < self.next_check.get(name, now):
                continue
            fresh = self._closed_only(self.broker.bars(mkt["epic"], self.cfg["timeframe"], 3), now)
            df = pd.concat([self.bars.get(name, pd.DataFrame()), fresh])
            df = df[~df.index.duplicated(keep="last")].sort_index().tail(MAX_BARS_KEPT)
            self.bars[name] = df
            last = str(df.index[-1]) if len(df) else ""
            if last and last != self.last_bar.get(name):
                self.last_bar[name] = last
                updated.append(name)
                df.to_csv(self._cache_path(name))
                self.next_check[name] = df.index[-1] + 2 * self.bar_delta + pd.Timedelta(seconds=30)
            else:
                # bar late -> retry soon; market closed (weekend/holiday) -> back off to save allowance
                last_close = (df.index[-1] + self.bar_delta) if len(df) else now
                closed = (now - last_close) > 2 * self.bar_delta
                self.next_check[name] = now + pd.Timedelta(minutes=60 if closed else 15)
        return updated

    # ---------------- reconciliation
    def reconcile(self, now) -> None:
        if self.dry_run:
            return
        ig = {p["deal_id"]: p for p in self.broker.open_positions()}
        for name, pos in list(self.positions.items()):
            if pos.deal_id not in ig:
                pnl = self._journal(pos, pos.stop_level, "closed_server_side", now)
                self.event("POSITION_GONE", market=name, deal_id=pos.deal_id, pnl_est=round(pnl, 2),
                           note="assumed stopped out; verify in IG history")
                del self.positions[name]
            elif ig[pos.deal_id]["stop_level"] is None:
                self.event("MISSING_STOP", market=name, deal_id=pos.deal_id)
                res = self.broker.update_stop(pos.deal_id, pos.stop_level)
                if res.get("status") != "ACCEPTED":
                    self._close(name, "emergency_no_stop", now)
        ours = {p.deal_id for p in self.positions.values()}
        for deal_id, p in ig.items():
            if deal_id not in ours and deal_id not in self.unmanaged_seen:
                self.unmanaged_seen.add(deal_id)
                self.event("UNMANAGED_POSITION_IGNORED", deal_id=deal_id, epic=p["epic"])

    # ---------------- actions
    def _close(self, name: str, reason: str, now) -> None:
        pos = self.positions[name]
        if self.dry_run:
            self.event("DRY_RUN_CLOSE", market=name, reason=reason)
            del self.positions[name]
            return
        res = self.broker.close_position(pos.deal_id, "BUY" if pos.direction == 1 else "SELL", pos.size)
        if res.get("status") == "ACCEPTED":
            px = float(res.get("level") or self.bars[name]["close"].iloc[-1])
            pnl = self._journal(pos, px, reason, now)
            self.event("CLOSED", market=name, reason=reason, level=px, pnl_est=round(pnl, 2))
            del self.positions[name]
        else:
            self.event("CLOSE_REJECTED", market=name, reason=res.get("reason"))

    def kill_all(self, reason: str, now) -> None:
        self.event("KILL_ALL", reason=reason, positions=list(self.positions))
        for name in list(self.positions):
            self._close(name, f"kill:{reason}", now)
        self.risk.halt(reason)
        self.save_state()

    def _manage(self, name: str, row: dict, now) -> None:
        pos = self.positions[name]
        dec = self.strategy.manage(pos, row)
        if dec.close:
            self._close(name, dec.reason, now)
        elif dec.new_stop is not None:
            if self.dry_run:
                pos.stop_level = dec.new_stop
                return
            res = self.broker.update_stop(pos.deal_id, round(dec.new_stop, 2))
            if res.get("status") == "ACCEPTED":
                self.event("STOP_MOVED", market=name, old=pos.stop_level, new=dec.new_stop)
                pos.stop_level = dec.new_stop
            else:
                self.event("STOP_UPDATE_REJECTED", market=name, reason=res.get("reason"))

    def _try_entry(self, name: str, row: dict, equity: float, now) -> None:
        mkt = self.mkts[name]
        sig = self.strategy.entry_signal(mkt, row)
        if not sig:
            return
        self.event("SIGNAL", market=name, direction=sig.direction, module=sig.module, score=sig.score,
                   conditions=sig.conditions, snapshot=sig.snapshot)

        def reject(why: str) -> None:
            self.event("ENTRY_REJECTED", market=name, reason=why)

        if self.pause_requested():
            return reject("PAUSE_SWITCH")
        info = self.broker.market_info(mkt["epic"])
        if not info.tradeable:
            return reject("MARKET_NOT_TRADEABLE")
        if info.spread > mkt["max_spread_points"]:
            return reject(f"SPREAD_WIDE {info.spread:.2f} > {mkt['max_spread_points']}")
        stop_dist = max(sig.stop_distance, info.min_stop_distance * 1.05)
        sz = self.risk.size(equity, stop_dist, mkt["point_value"], info.min_size, mkt["size_step"])
        if not sz.ok:
            return reject(sz.reason)
        margin = sz.size * info.mid * mkt["point_value"] * mkt["margin_pct"] / 100
        prov = PositionState(market=name, direction=sig.direction, module=sig.module, entry_price=info.mid,
                             stop_level=info.mid - sig.direction * stop_dist, size=sz.size,
                             point_value=mkt["point_value"], initial_risk_points=stop_dist,
                             entry_ts=str(now), extreme=info.mid, margin=margin, group=mkt["group"])
        ok, why = self.risk.check_entry(now, mkt, list(self.positions.values()), equity, sz.risk_amount, margin)
        if not ok:
            return reject(why)

        order = {"epic": mkt["epic"], "direction": "BUY" if sig.direction == 1 else "SELL",
                 "size": sz.size, "stop_distance": round(stop_dist, 1), "expiry": info.expiry,
                 "currency": info.currency}
        assert_order_safe(order)
        if self.dry_run:
            self.event("DRY_RUN_ORDER", market=name, risk=round(sz.risk_amount, 2), **order)
            return
        self.risk.on_order_sent()
        res = self.broker.open_position(**order)
        if res.get("status") != "ACCEPTED":
            return self.event("ORDER_REJECTED", market=name, reason=res.get("reason"), order=order)
        level = float(res.get("level") or info.mid)
        prov.entry_price = level
        prov.extreme = level
        prov.stop_level = float(res.get("stop_level") or level - sig.direction * stop_dist)
        prov.deal_id = res["deal_id"]
        self.positions[name] = prov
        self.event("OPENED", market=name, deal_id=prov.deal_id, level=level, stop=prov.stop_level,
                   size=sz.size, risk=round(sz.risk_amount, 2))

    # ---------------- switches
    def kill_requested(self) -> bool:
        return Path(self.cfg["safety"]["kill_file"]).exists() or env_flag("BOT_KILL")

    def pause_requested(self) -> bool:
        return Path(self.cfg["safety"]["pause_file"]).exists() or env_flag("BOT_PAUSE")

    # ---------------- main cycle
    def startup(self, now: pd.Timestamp | None = None) -> None:
        now = now or pd.Timestamp.now(tz="UTC")
        self.broker.login()
        self.warm_up(now)
        self.reconcile(now)
        self.log_feasibility()
        self.save_state()

    def log_feasibility(self) -> None:
        """Log, per market, whether current equity can size a trade within the risk cap."""
        try:
            equity = self.broker.equity()
            t = self.cfg["strategy"]["trend"]
            for name, mkt in self.mkts.items():
                df = self.bars.get(name)
                if df is None or len(df) < 30:
                    continue
                info = self.broker.market_info(mkt["epic"])
                a = float(compute_all(df, self.cfg["strategy"])["atr"].iloc[-1])
                stop = max(t["stop_atr"] * a, info.min_stop_distance * 1.05)
                spread = info.spread if info.spread == info.spread else None   # NaN when market closed
                row = feasibility_row(name, equity, self.risk.current_risk_pct(), info.min_size, stop,
                                      mkt["point_value"], spread)
                tail = df.tail(60)
                tr = true_range(tail).iloc[1:]
                self.event("FEASIBILITY", equity=round(equity, 2), atr=round(a, 2),
                           median_tr_60=round(float(tr.median()), 2), max_tr_60=round(float(tr.max()), 2),
                           max_tr_bar=str(tr.idxmax()), last_close=round(float(tail["close"].iloc[-1]), 2),
                           bars=len(df), first_bar=str(df.index[0]), **row)
        except Exception as exc:  # informational only; never block trading startup
            self.event("FEASIBILITY_ERROR", error=repr(exc))

    def tick(self, now: pd.Timestamp | None = None) -> bool:
        """One decision cycle. Returns False when the bot must stop."""
        now = now or pd.Timestamp.now(tz="UTC")
        if self.kill_requested():
            if not self.risk.state.halted or self.positions:
                self.kill_all("KILL_SWITCH", now)
            self.event("HALTED", reason=self.risk.state.halt_reason, reset_code=self.risk.state.halt_code)
            return False
        if self.risk.state.halted:
            if self.positions:
                self.kill_all(self.risk.state.halt_reason or "HALTED", now)
            self.event("HALTED", reason=self.risk.state.halt_reason, reset_code=self.risk.state.halt_code)
            return False

        updated = self.refresh(now)
        if not updated:
            return True
        equity = self.broker.equity()
        self.risk.on_bar(now, equity)
        if self.risk.state.halted:
            self.kill_all(self.risk.state.halt_reason, now)
            return False
        self.reconcile(now)

        for name in updated:
            df = self.bars[name]
            if len(df) < self.cfg["warmup_bars"] * 0.9:
                self.event("NOT_ENOUGH_BARS", market=name, bars=len(df))
                continue
            row = compute_all(df, self.cfg["strategy"]).iloc[-1].to_dict()
            if name in self.positions:
                self._manage(name, row, now)
            else:
                self._try_entry(name, row, equity, now)
        self.save_state()
        return True


def _idle(runner: "LiveRunner", why: str) -> None:
    """Stay up (so a hosting platform doesn't restart-loop) but do nothing except report status."""
    while True:
        st = runner.risk.state
        log.error("%s | halted=%s reason=%s | reset code: %s", why, st.halted, st.halt_reason, st.halt_code)
        time.sleep(3600)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="IG bot live loop")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset-halt", action="store_true")
    ap.add_argument("--state-dir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()
    cfg = load_config(args.config)
    apply_env_overrides(cfg)
    only = [m.strip().upper() for m in os.environ.get("BOT_MARKETS", "").split(",") if m.strip()]
    if only:
        cfg["markets"] = [m for m in cfg["markets"] if m["name"] in only]
    env = resolve_environment(cfg, args.live or env_flag("BOT_LIVE_FLAG"), os.environ.get("IG_ENV"))
    apply_live_risk_cap(cfg, env)
    state_dir = Path(args.state_dir or os.environ.get("STATE_DIR", "state"))
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))

    runner = LiveRunner(cfg, None, state_dir, log_dir)
    lock = InstanceLock(state_dir / "bot.lock")
    lock.acquire()

    def _sigterm(*_):
        runner.save_state()
        lock.release()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    # Halt reset: CLI flag, or BOT_RESET_HALT=<code printed in the logs>. A stale value
    # can't clear a later halt because every halt gets a new code.
    code = os.environ.get("BOT_RESET_HALT", "").strip().upper()
    st = runner.risk.state
    if args.reset_halt or (st.halted and code and code == st.halt_code):
        runner.risk.reset_halt()
        runner.save_state()
        runner.event("HALT_RESET", via="cli" if args.reset_halt else "env")
        if args.reset_halt:
            print("halt cleared. Remove the KILL file / BOT_KILL variable too if set.")
            return

    missing = [k for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD") if not os.environ.get(k)]
    if missing:
        log.error("waiting for credentials: set %s (Railway: service -> Variables)", ", ".join(missing))
        if args.once:
            return
        _idle(runner, "NO_CREDENTIALS")

    runner.broker = IGClient(env, os.environ["IG_API_KEY"], os.environ["IG_USERNAME"],
                             os.environ["IG_PASSWORD"], os.environ.get("IG_ACCOUNT_ID") or None)
    log.info("starting env=%s dry_run=%s risk_per_trade=%s%% markets=%s", env, runner.dry_run,
             cfg["risk"]["risk_per_trade_pct"], ",".join(runner.mkts))
    runner.event("STARTED", env=env, dry_run=runner.dry_run, risk_pct=cfg["risk"]["risk_per_trade_pct"])
    runner.startup()
    errors = 0
    while True:
        lock.beat()
        try:
            alive = runner.tick()
            errors = 0
        except Exception as exc:  # fail closed: stop opening trades, server-side stops stay in place
            errors += 1
            runner.event("ERROR", error=repr(exc), consecutive=errors)
            if errors >= 5:
                runner.event("STOPPING_AFTER_REPEATED_ERRORS")
                runner.save_state()
                raise
            alive = True
        if args.once:
            break
        if not alive:
            _idle(runner, "BOT_STOPPED")
        time.sleep(cfg["safety"]["poll_seconds"])


if __name__ == "__main__":
    main()
