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
    BOT_ENABLE_SEGMENTS=EURUSD:meanrev   re-enable a segment the learning loop switched off
    BOT_RESET_SLEEVES=leveraged          resume a sleeve that hit its own drawdown limit
    STATE_DIR / LOG_DIR      where state, bar cache and logs live (a persistent volume)

Every decision is written to logs/decisions.jsonl; closed trades to logs/trades.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
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
from .sleeves import Sleeves, sleeve_key
from .dip_live import DipEngine, is_dip
from .history import HistoryDownloader
from .kronos_shadow import KronosShadow
from .learning import append_journal, journal_record, load_journal, review
from .learning import settings as learning_settings
from .strategy import PositionState, Strategy, entry_features

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
        base = cfg["risk"]["risk_per_trade_pct"]
        factor = min(1.0, cap / base) if base > 0 else 1.0
        cfg["risk"]["risk_per_trade_pct"] = min(base, cap)
        for sleeve in (cfg.get("sleeves") or {}).values():   # sleeves size from their own %, scale them too
            sleeve["risk_per_trade_pct"] = sleeve["risk_per_trade_pct"] * factor
            if sleeve.get("min_size_max_risk_pct"):
                sleeve["min_size_max_risk_pct"] = sleeve["min_size_max_risk_pct"] * factor


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
BAR_MODULES = {"trend", "meanrev", "sweep"}


class LiveRunner:
    def __init__(self, cfg: dict, broker, state_dir: str | Path = "state", log_dir: str | Path = "logs",
                 dry_run: bool | None = None):
        self.cfg = cfg
        self.broker = broker
        self.mkts = market_map(cfg)
        self.bar_markets = {n: m for n, m in self.mkts.items() if set(m["modules"]) & BAR_MODULES}
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
        self.sleeves = Sleeves(cfg)
        self.kronos = KronosShadow(cfg, self.log_dir, self.state_dir)
        self.history = HistoryDownloader(cfg, broker, self.state_dir, self.log_dir, self.event)
        self._info_cache: dict[str, tuple] = {}
        self.dip_prices: dict[str, float] = {}
        self.dip_done: dict[str, str] = {}
        self._load_state()
        self.dip = DipEngine(self)

    # ---------------- persistence & logging
    @property
    def _state_file(self) -> Path:
        return self.state_dir / "state.json"

    def _load_state(self) -> None:
        if self._state_file.exists():
            d = json.loads(self._state_file.read_text())
            self.risk.state = RiskState(**d.get("risk", {}))
            self.sleeves = Sleeves(self.cfg, d.get("sleeves"))
            self.positions = {}
            for k, v in d.get("positions", {}).items():
                pos = PositionState(**v)
                if not pos.sleeve:   # state written before sleeves existed
                    pos.sleeve = self.sleeves.by_module.get(pos.module, self.sleeves.names()[0])
                self.positions[sleeve_key(pos.sleeve, pos.market)] = pos
            self.last_bar = d.get("last_bar", {})
            self.dip_done = d.get("dip_done", {})

    def save_state(self) -> None:
        d = {"risk": self.risk.state.to_dict(),
             "sleeves": self.sleeves.to_dict(),
             "positions": {k: v.to_dict() for k, v in self.positions.items()},
             "last_bar": self.last_bar,
             "dip_done": self.dip_done}
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
        rec = journal_record(pos, exit_px, now, reason, pnl)
        append_journal(self.log_dir / "trades.jsonl", rec)
        self.event("TRADE_JOURNAL", **{k: v for k, v in rec.items() if k != "features"})
        self.sleeves.on_trade_closed(pos.sleeve, pnl, now)
        return pnl

    def prices(self) -> dict[str, float]:
        bar = {m: float(df["close"].iloc[-1]) for m, df in self.bars.items() if len(df)}
        return {**self.dip_prices, **bar}

    def sleeve_summary(self) -> dict:
        return self.sleeves.summary(self.positions.values(), self.prices())

    # ---------------- learning loop
    def maybe_learn(self, now, force: bool = False) -> dict | None:
        s = learning_settings(self.cfg)
        if not s["enabled"]:
            return None
        iso = pd.Timestamp(now).isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        if not force and week == self.risk.state.last_learning_week:
            return None
        rep = review(load_journal(self.log_dir / "trades.jsonl"), self.cfg, self.risk.state.disabled_segments)
        if self.kronos.enabled or self.kronos.log_path.exists():
            try:
                rep["kronos"] = self.kronos.evaluate(self.bars)
            except Exception as exc:
                rep["kronos"] = {"error": repr(exc)}
        for seg in rep["disable"]:
            if seg not in self.risk.state.disabled_segments:
                self.risk.state.disabled_segments.append(seg)
                self.event("SEGMENT_DISABLED", segment=seg,
                           note="statistically losing; re-enable only via BOT_ENABLE_SEGMENTS after review")
        self.risk.state.last_learning_week = week
        (self.log_dir / "learning_report.json").write_text(json.dumps(rep, indent=2, default=str))
        self.event("LEARNING_REPORT", week=week, trades=rep["trades"], overall=rep.get("overall"),
                   findings=rep["findings"], suggestions=rep["suggestions"], disabled_now=rep["disable"],
                   disabled_total=list(self.risk.state.disabled_segments), sleeves=self.sleeve_summary(),
                   replay=rep.get("montecarlo"), kronos=rep.get("kronos"))
        self.save_state()
        return rep

    # ---------------- bars
    def _cache_path(self, name: str) -> Path:
        return self.state_dir / "bars" / f"{name}.csv"

    def _closed_only(self, df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
        return df[df.index + self.bar_delta <= now] if len(df) else df

    def warm_up(self, now: pd.Timestamp) -> None:
        need = int(self.cfg["warmup_bars"])
        for name, mkt in self.bar_markets.items():
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
        for name, mkt in self.bar_markets.items():
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
                self.next_check[name] = self._next_check_without_new_bar(df, now)
        return updated

    def _next_check_without_new_bar(self, df: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
        """No new closed bar. Wait for the next bar close (e.g. after a restart mid-bar). Retry every
        15 min for up to an hour only if a bar is overdue (IG lag). If it's still missing (weekend,
        holiday), check once per bar. Keeps history use to about 1 fetch per market per bar."""
        if not len(df):
            return now + pd.Timedelta(minutes=15)
        due = df.index[-1] + 2 * self.bar_delta + pd.Timedelta(seconds=30)   # close of the next bar
        if now < due:
            return due
        if now - due < pd.Timedelta(hours=1):
            return now + pd.Timedelta(minutes=15)
        return due + (int((now - due) / self.bar_delta) + 1) * self.bar_delta

    # ---------------- reconciliation
    def reconcile(self, now) -> None:
        if self.dry_run:
            return
        ig = {p["deal_id"]: p for p in self.broker.open_positions()}
        for key, pos in list(self.positions.items()):
            if pos.deal_id not in ig:
                exit_px, how = self._estimate_server_exit(pos)
                pnl = self._journal(pos, exit_px, how, now)
                self.event("POSITION_GONE", market=pos.market, sleeve=pos.sleeve, deal_id=pos.deal_id,
                           pnl_est=round(pnl, 2), assumed=how,
                           note="estimated from stop/target levels; verify in IG history")
                del self.positions[key]
            elif ig[pos.deal_id]["stop_level"] is None:
                self.event("MISSING_STOP", market=pos.market, sleeve=pos.sleeve, deal_id=pos.deal_id)
                res = self.broker.update_stop(pos.deal_id, pos.stop_level)
                if res.get("status") != "ACCEPTED":
                    self._close(key, "emergency_no_stop", now)
        ours = {p.deal_id for p in self.positions.values()}
        for deal_id, p in ig.items():
            if deal_id not in ours and deal_id not in self.unmanaged_seen:
                self.unmanaged_seen.add(deal_id)
                self.event("UNMANAGED_POSITION_IGNORED", deal_id=deal_id, epic=p["epic"])

    def _estimate_server_exit(self, pos: PositionState) -> tuple[float, str]:
        """IG closed it (stop or limit hit). Use the latest bar to judge which one."""
        df = self.bars.get(pos.market)
        if pos.target_level is not None and df is not None and len(df):
            bar = df.iloc[-1]
            reached = bar["high"] >= pos.target_level if pos.direction == 1 else bar["low"] <= pos.target_level
            if reached:
                return pos.target_level, "target_server_side"
        return pos.stop_level, "stop_server_side"

    # ---------------- actions
    def _close(self, key: str, reason: str, now) -> None:
        pos = self.positions[key]
        if self.dry_run:
            self.event("DRY_RUN_CLOSE", market=pos.market, sleeve=pos.sleeve, reason=reason)
            del self.positions[key]
            return
        res = self.broker.close_position(pos.deal_id, "BUY" if pos.direction == 1 else "SELL", pos.size)
        if res.get("status") == "ACCEPTED":
            px = float(res.get("level") or self.prices().get(pos.market, pos.entry_price))
            pnl = self._journal(pos, px, reason, now)
            self.event("CLOSED", market=pos.market, sleeve=pos.sleeve, reason=reason, level=px,
                       pnl_est=round(pnl, 2))
            del self.positions[key]
        else:
            self.event("CLOSE_REJECTED", market=pos.market, sleeve=pos.sleeve, reason=res.get("reason"))

    def kill_all(self, reason: str, now) -> None:
        self.event("KILL_ALL", reason=reason, positions=list(self.positions))
        for key in list(self.positions):
            self._close(key, f"kill:{reason}", now)
        self.risk.halt(reason)
        self.save_state()

    def _manage(self, key: str, row: dict, now) -> None:
        pos = self.positions[key]
        name = pos.market
        dec = self.strategy.manage(pos, row)
        if dec.close:
            self._close(key, dec.reason, now)
        elif dec.new_stop is not None:
            if self.dry_run:
                pos.stop_level = dec.new_stop
                return
            res = self.broker.update_stop(pos.deal_id, round(dec.new_stop, 2))
            if res.get("status") == "ACCEPTED":
                self.event("STOP_MOVED", market=name, sleeve=pos.sleeve, old=pos.stop_level, new=dec.new_stop)
                pos.stop_level = dec.new_stop
            else:
                self.event("STOP_UPDATE_REJECTED", market=name, reason=res.get("reason"))

    def _sentiment(self, info) -> float | None:
        if not info.market_id or not hasattr(self.broker, "client_sentiment"):
            return None
        try:
            return float(self.broker.client_sentiment(info.market_id)["long_pct"])
        except Exception as exc:  # informational only
            self.event("SENTIMENT_ERROR", market_id=info.market_id, error=repr(exc))
            return None

    def _market_context(self, epic: str):
        """market_info + sentiment, fetched at most once per market per cycle."""
        if epic not in self._info_cache:
            info = self.broker.market_info(epic)
            self._info_cache[epic] = (info, self._sentiment(info))
        return self._info_cache[epic]

    def _try_entry(self, name: str, sleeve: str, row: dict, equity: float, now) -> None:
        mkt = self.mkts[name]
        mods = self.sleeves.modules_for(sleeve, mkt["modules"])
        if not mods:
            return
        sig = self.strategy.entry_signal({**mkt, "modules": mods}, row)
        if not sig:
            return
        self._place_entry(name, sleeve, sig, row, equity, now)

    def _place_entry(self, name: str, sleeve: str, sig, row: dict, equity: float, now) -> None:
        """Every entry, from any strategy, goes through here: sizing, risk checks, stop, order."""
        mkt = self.mkts[name]
        info, crowd_long = self._market_context(mkt["epic"])
        spread = info.spread if info.spread == info.spread else None
        features = entry_features(sig, row, now, crowd_long_pct=crowd_long, spread=spread, sleeve=sleeve)
        self.event("SIGNAL", market=name, sleeve=sleeve, direction=sig.direction, module=sig.module,
                   score=sig.score, conditions=sig.conditions, crowd_long_pct=crowd_long, snapshot=sig.snapshot)

        def reject(why: str) -> None:
            self.event("ENTRY_REJECTED", market=name, sleeve=sleeve, module=sig.module, reason=why)

        if f"{name}:{sig.module}" in self.risk.state.disabled_segments:
            return reject("LEARNED_DISABLED")
        if self.pause_requested():
            return reject("PAUSE_SWITCH")
        if not info.tradeable:
            return reject("MARKET_NOT_TRADEABLE")
        if mkt.get("max_spread_points") is not None and info.spread > mkt["max_spread_points"]:
            return reject(f"SPREAD_WIDE {info.spread:.2f} > {mkt['max_spread_points']}")
        positions = list(self.positions.values())
        s_eq = self.sleeves.equity(sleeve, positions, self.prices())
        stop_dist = max(sig.stop_distance, info.min_stop_distance * 1.05)
        pct_cap = self.cfg["safety"].get("max_spread_pct_of_stop")
        if pct_cap and info.spread > stop_dist * pct_cap / 100:
            return reject(f"SPREAD_TOO_COSTLY {info.spread:.2f} > {pct_cap}% of stop {stop_dist:.1f}")
        room = self.sleeves.max_size_by_leverage(sleeve, s_eq, positions, info.mid, mkt["point_value"])
        sz = self.risk.size(s_eq, stop_dist, mkt["point_value"], info.min_size, mkt["size_step"],
                            risk_pct=self.sleeves.risk_pct(sleeve), max_size=room,
                            min_size_max_risk_pct=self.sleeves.min_size_max_risk_pct(sleeve))
        if not sz.ok:
            return reject(sz.reason)
        at_min = sz.reason.startswith("MIN_SIZE_TOLERANCE")
        ok, why = self.sleeves.check(sleeve, now, name, mkt["group"], positions)
        if not ok:
            return reject(why)
        notional = sz.size * info.mid * mkt["point_value"]
        margin = notional * mkt["margin_pct"] / 100
        prov = PositionState(market=name, direction=sig.direction, module=sig.module, entry_price=info.mid,
                             stop_level=info.mid - sig.direction * stop_dist, size=sz.size,
                             point_value=mkt["point_value"], initial_risk_points=stop_dist,
                             entry_ts=str(now), extreme=info.mid, margin=margin, group=mkt["group"],
                             sleeve=sleeve, notional=notional, features=features)
        ok, why = self.risk.check_entry(now, mkt, positions, equity, sz.risk_amount, margin, notional,
                                        sleeve=sleeve)
        if not ok:
            return reject(why)

        order = {"epic": mkt["epic"], "direction": "BUY" if sig.direction == 1 else "SELL",
                 "size": sz.size, "stop_distance": round(stop_dist, 1), "expiry": info.expiry,
                 "currency": info.currency}
        target_dist = sig.target_distance * stop_dist / sig.stop_distance if sig.target_distance else None
        if target_dist:
            order["limit_distance"] = round(target_dist, 1)
        assert_order_safe(order)
        if self.dry_run:
            self.event("DRY_RUN_ORDER", market=name, sleeve=sleeve, risk=round(sz.risk_amount, 2),
                       sleeve_equity=round(s_eq, 2), min_size_trade=at_min, **order)
            return
        self.risk.on_order_sent()
        res = self.broker.open_position(**order)
        if res.get("status") != "ACCEPTED":
            return self.event("ORDER_REJECTED", market=name, sleeve=sleeve, reason=res.get("reason"), order=order)
        level = float(res.get("level") or info.mid)
        prov.entry_price = level
        prov.extreme = level
        prov.notional = sz.size * level * mkt["point_value"]
        prov.stop_level = float(res.get("stop_level") or level - sig.direction * stop_dist)
        if target_dist:
            prov.target_level = float(res.get("limit_level") or level + sig.direction * target_dist)
        prov.deal_id = res["deal_id"]
        self.positions[sleeve_key(sleeve, name)] = prov
        self.event("OPENED", market=name, sleeve=sleeve, module=sig.module, deal_id=prov.deal_id, level=level,
                   stop=prov.stop_level, target=prov.target_level, size=sz.size, risk=round(sz.risk_amount, 2),
                   sleeve_equity=round(s_eq, 2), exposure_x=round(notional / s_eq, 2) if s_eq else None,
                   min_size_trade=at_min)

    # ---------------- switches
    def kill_requested(self) -> bool:
        return Path(self.cfg["safety"]["kill_file"]).exists() or env_flag("BOT_KILL")

    def pause_requested(self) -> bool:
        return Path(self.cfg["safety"]["pause_file"]).exists() or env_flag("BOT_PAUSE")

    # ---------------- main cycle
    def startup(self, now: pd.Timestamp | None = None) -> None:
        now = now or pd.Timestamp.now(tz="UTC")
        self.broker.login()
        self.history.attach(self.broker)          # broker is assigned after construction in main()
        self.check_account_type()
        changes = self.sleeves.ensure_bases(self.broker.equity())
        if changes:
            self.event("SLEEVES_REBASED", changes=changes,
                       note="sleeve weights changed in config; slices re-cut from the same account reference")
        self.warm_up(now)
        self.reconcile(now)
        self.event("SLEEVES", **self.sleeve_summary())
        if self.dip.enabled:
            self.dip.check_markets()
            threading.Thread(target=self.dip.prefetch, args=(now,), daemon=True, name="dip-prefetch").start()
        self.log_feasibility()
        self.log_market_search()
        self.kronos_selftest()
        self.save_state()

    # ---------------- idle-time work: history download + backtest (no trading effect)
    def _background(self, now: pd.Timestamp) -> None:
        if self.history.disabled:
            return
        if any(t <= now + pd.Timedelta(minutes=10) for t in self.next_check.values()):
            return                                   # a bar is due soon: stay out of the way
        try:
            if not self.history.step(now):
                self.history.maybe_backtest()
        except Exception as exc:                     # fail closed for this feature only
            self.history.disabled = True
            self.event("HISTORY_ERROR", error=repr(exc),
                       note="history download/backtest stopped until restart; trading unaffected")

    # ---------------- Kronos (shadow only: recorded and scored, never traded on)
    def kronos_selftest(self) -> None:
        if not self.kronos.enabled:
            return
        name = next((m for m, df in self.bars.items() if len(df) >= 100), None)
        if not name:
            return
        try:
            res = self.kronos.forecast({name: self.bars[name]})
            r = res["results"].get(name, {})
            self.event("KRONOS_SELFTEST", ok="error" not in r, market=name, seconds=res.get("seconds"),
                       load_seconds=res.get("load_seconds"), result=r)
        except Exception as exc:
            self.event("KRONOS_SELFTEST", ok=False, error=repr(exc))

    def _run_kronos(self, markets: list[str], now) -> None:
        try:
            res = self.kronos.forecast({m: self.bars[m] for m in markets if m in self.bars})
        except Exception as exc:          # never let the shadow model disturb trading
            self.event("KRONOS_ERROR", error=repr(exc))
            return
        for m, r in res.get("results", {}).items():
            if "error" in r:
                self.event("KRONOS_ERROR", market=m, error=r["error"])
                continue
            self.event("KRONOS_FORECAST", market=m, bar=r["bar"], last_close=r["last_close"],
                       pred_close=r["pred_close"], exp_ret_pct=r["exp_ret_pct"], seconds=res.get("seconds"))
            for pos in self.positions.values():
                if pos.market == m and pos.entry_ts == str(now):     # opened on this bar
                    pos.features["kronos_exp_ret_pct"] = r["exp_ret_pct"]
                    pos.features["kronos_agree"] = bool((r["exp_ret_pct"] > 0) == (pos.direction == 1))

    def check_account_type(self) -> None:
        """Refuse to trade on the wrong kind of account: sizing is GBP per point (spread bet)."""
        if not hasattr(self.broker, "accounts"):
            return
        want = str(self.cfg.get("account", {}).get("type", "SPREADBET")).upper()
        accts = self.broker.accounts()
        current = next((a for a in accts if a["account_id"] == self.broker.account_id), None)
        self.event("ACCOUNTS", using=self.broker.account_id, expected_type=want, accounts=accts)
        if current is None or str(current["type"]).upper() != want:
            got = current["type"] if current else "unknown"
            options = [a["account_id"] for a in accts if str(a["type"]).upper() == want]
            raise SafetyError(f"ACCOUNT_TYPE_MISMATCH: logged into {self.broker.account_id} ({got}), "
                              f"bot is configured for {want}. Set IG_ACCOUNT_ID to one of {options}.")

    def log_market_search(self) -> None:
        """BOT_SEARCH_MARKETS=silver,copper logs IG's matching instruments (spread-bet DFB first). No trading effect."""
        terms = [t.strip() for t in os.environ.get("BOT_SEARCH_MARKETS", "").split(",") if t.strip()]
        if not terms or not hasattr(self.broker, "search_markets"):
            return
        for term in terms:
            try:
                found = self.broker.search_markets(term)
                found.sort(key=lambda m: (m.get("expiry") != "DFB", m.get("name") or ""))
                self.event("MARKET_SEARCH", term=term, results=found[:15])
            except Exception as exc:
                self.event("MARKET_SEARCH_ERROR", term=term, error=repr(exc))

    def log_feasibility(self) -> None:
        """Log, per market, whether current equity can size a trade within the risk cap."""
        try:
            t = self.cfg["strategy"]["trend"]
            positions = list(self.positions.values())
            for name, mkt in self.bar_markets.items():
                df = self.bars.get(name)
                if df is None or len(df) < 30:
                    continue
                info = self.broker.market_info(mkt["epic"])
                a = float(compute_all(df, self.cfg["strategy"])["atr"].iloc[-1])
                stop = max(t["stop_atr"] * a, info.min_stop_distance * 1.05)
                spread = info.spread if info.spread == info.spread else None   # NaN when market closed
                tail = df.tail(60)
                tr = true_range(tail).iloc[1:]
                for sn in self.sleeves.names():
                    if not self.sleeves.modules_for(sn, mkt["modules"]):
                        continue
                    s_eq = self.sleeves.equity(sn, positions, self.prices())
                    row = feasibility_row(name, s_eq, self.sleeves.risk_pct(sn), info.min_size, stop,
                                          mkt["point_value"], spread,
                                          min_size_max_risk_pct=self.sleeves.min_size_max_risk_pct(sn))
                    self.event("FEASIBILITY", sleeve=sn, sleeve_equity=round(s_eq, 2), atr=round(a, 2),
                               median_tr_60=round(float(tr.median()), 2), max_tr_60=round(float(tr.max()), 2),
                               max_tr_bar=str(tr.idxmax()), bars=len(df), **row)
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

        self.maybe_learn(now)
        try:
            self.dip.tick(now)
        except Exception as exc:          # the daily strategy must never take down the 4H bot
            self.event("DIP_ERROR", error=repr(exc))
        updated = self.refresh(now)
        if not updated:
            self._background(now)
            return True
        self._info_cache = {}
        equity = self.broker.equity()
        self.risk.on_bar(now, equity)
        if self.risk.state.halted:
            self.kill_all(self.risk.state.halt_reason, now)
            return False
        self.reconcile(now)
        prices = self.prices()
        for sn in self.sleeves.names():
            why = self.sleeves.update(sn, self.sleeves.equity(sn, self.positions.values(), prices), now)
            if why:
                self.event("SLEEVE_HALTED", sleeve=sn, reason=why,
                           note="no new entries for this sleeve; open trades still managed; reset via BOT_RESET_SLEEVES")

        for name in updated:
            df = self.bars[name]
            if len(df) < self.cfg["warmup_bars"] * 0.9:
                self.event("NOT_ENOUGH_BARS", market=name, bars=len(df))
                continue
            row = compute_all(df, self.cfg["strategy"]).iloc[-1].to_dict()
            for key in [k for k, p in self.positions.items() if p.market == name and not is_dip(p)]:
                self._manage(key, row, now)
            for sn in self.sleeves.names():
                if sleeve_key(sn, name) not in self.positions:
                    self._try_entry(name, sn, row, equity, now)
        if self.kronos.enabled:           # after all trading decisions: adds no delay to orders
            self._run_kronos(updated, now)
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

    for sn in [x.strip() for x in os.environ.get("BOT_RESET_SLEEVES", "").split(",") if x.strip()]:
        if sn in runner.sleeves.states and runner.sleeves.states[sn].halted:
            runner.sleeves.reset(sn)
            runner.event("SLEEVE_RESET", sleeve=sn, via="env")
    for seg in [x.strip() for x in os.environ.get("BOT_ENABLE_SEGMENTS", "").split(",") if x.strip()]:
        if seg in runner.risk.state.disabled_segments:
            runner.risk.state.disabled_segments.remove(seg)
            runner.event("SEGMENT_REENABLED", segment=seg, via="env")
    runner.save_state()

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
    try:
        runner.startup()
    except SafetyError as exc:          # wrong account type etc.: stay up, trade nothing, say why
        runner.event("STARTUP_REFUSED", reason=str(exc))
        _idle(runner, f"STARTUP_REFUSED: {exc}")
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
