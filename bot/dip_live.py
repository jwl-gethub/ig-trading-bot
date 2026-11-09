"""Daily RSI(2) dip strategy inside the live bot.

Signals come from Dukascopy hourly history (bot/dukascopy.py: free, uses no IG allowance) and the
same rules the research screen tested (bot/dipbuy.py). Orders go through the runner's normal path:
sleeve sizing, RiskManager checks, server-side stop, dry-run and kill/pause switches.

Once per market per trading day, at the market's fill time (a liquid hour the morning after the
close), the engine reads the latest completed daily close and acts: exit an open dip position whose
close has crossed back over its 5-day average, or open a new one. If the close is not published yet
it retries every `retry_minutes`; after `grace_hours` it skips that day (DIP_MISSED). Nothing else
in the bot changes: 4H positions and dip positions never manage each other.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .dipbuy import DipRules, DipSchedule, daily_bars, decide, fill_time, indicators
from .dukascopy import HourlySource
from .sleeves import sleeve_key
from .strategy import Signal

SCHEDULES = {
    "us_index": DipSchedule("America/New_York", 16, "America/New_York", 10),
    "eu_index": DipSchedule("Europe/Berlin", 17, "Europe/Berlin", 10),
    "uk_index": DipSchedule("Europe/London", 16, "Europe/London", 9),
    "fx_day": DipSchedule("America/New_York", 17, "Europe/London", 8),
}
DEFAULTS = {"enabled": True, "sleeve": "dip", "history_days": 420, "retry_minutes": 15, "grace_hours": 3,
            "rules": {}}
RULE_KEYS = {"rsi_n", "rsi_long", "rsi_short", "trend_sma", "exit_sma", "atr_n", "stop_atr"}


def settings(cfg: dict) -> dict:
    s = {**DEFAULTS, **(cfg.get("dip") or {})}
    s["rules"] = {k: v for k, v in (s.get("rules") or {}).items() if k in RULE_KEYS}
    return s


def last_close_before(now: pd.Timestamp, s: DipSchedule) -> pd.Timestamp:
    """Most recent weekday close (close_hour local) at or before now, in UTC."""
    local = now.tz_convert(s.close_tz)
    day = local.normalize().tz_localize(None)
    for _ in range(10):
        if day.dayofweek < 5:
            ts = (day + pd.Timedelta(hours=s.close_hour)).tz_localize(s.close_tz, nonexistent="shift_forward",
                                                                     ambiguous=False)
            if ts <= local:
                return ts.tz_convert("UTC")
        day -= pd.Timedelta(days=1)
    raise RuntimeError("no close found")


class DipEngine:
    def __init__(self, runner, source=None):
        self.r = runner
        self.s = settings(runner.cfg)
        self.sleeve = self.s["sleeve"]
        self.markets = {n: m for n, m in runner.mkts.items() if "dip" in m["modules"]}
        self.enabled = bool(self.s["enabled"]) and bool(self.markets) and self.sleeve in runner.sleeves.names()
        self.source = source or HourlySource(Path(runner.state_dir) / "dukascopy", log=None)
        self.retry: dict[str, pd.Timestamp] = {}
        self.unavailable: set[str] = set()

    def rules(self, mkt: dict) -> DipRules:
        return DipRules(long_only=bool(mkt["dip"].get("long_only", True)), **self.s["rules"])

    def schedule(self, mkt: dict) -> DipSchedule:
        return SCHEDULES[mkt["dip"]["schedule"]]

    # ------------------------------------------------------------ startup
    def check_markets(self) -> None:
        """Confirm every dip epic exists on IG; drop (and log) the ones that don't."""
        for name, mkt in self.markets.items():
            try:
                info = self.r.broker.market_info(mkt["epic"])
                self.r.event("DIP_MARKET_OK", market=name, epic=mkt["epic"], min_size=info.min_size,
                             tradeable=info.tradeable)
            except Exception as exc:
                self.unavailable.add(name)
                hint = None
                if hasattr(self.r.broker, "search_markets"):
                    try:
                        hint = [r for r in self.r.broker.search_markets(mkt["dip"].get("search", name))
                                if r.get("expiry") == "DFB"][:5]
                    except Exception:
                        hint = None
                self.r.event("DIP_MARKET_UNAVAILABLE", market=name, epic=mkt["epic"], error=repr(exc),
                             dfb_candidates=hint)

    def prefetch(self, now: pd.Timestamp) -> None:
        """Warm the Dukascopy cache (run in a background thread at startup) so the daily decision
        only needs the latest day's file and never blocks the main loop for long."""
        for name, mkt in self.markets.items():
            if name in self.unavailable:
                continue
            try:
                h = self._history(mkt, now)
                self.r.event("DIP_PREFETCH", market=name, hourly_bars=int(len(h)),
                             last=str(h.index[-1]) if len(h) else None)
            except Exception as exc:
                self.r.event("DIP_PREFETCH_ERROR", market=name, error=repr(exc))

    # ------------------------------------------------------------ daily cycle
    def due(self, name: str, now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        s = self.schedule(self.markets[name])
        close = last_close_before(now, s)
        fill = fill_time(close, s)
        if now < fill or self.r.dip_done.get(name) == str(close):
            return None
        if now > fill + pd.Timedelta(hours=self.s["grace_hours"]):
            self.r.dip_done[name] = str(close)
            self.r.event("DIP_MISSED", market=name, close=str(close), note="past the grace window; skipped")
            return None
        if now < self.retry.get(name, now - pd.Timedelta(seconds=1)):
            return None
        return close, fill

    def tick(self, now: pd.Timestamp) -> None:
        if not self.enabled:
            return
        due = {n: d for n in self.markets if n not in self.unavailable and (d := self.due(n, now))}
        if not due:
            return
        self.r.reconcile(now)                 # catch server-side stops before deciding
        equity = self.r.broker.equity()
        for name, (close, _fill) in due.items():
            try:
                self._decide(name, close, equity, now)
            except Exception as exc:          # fail closed for this market today; others carry on
                self.r.dip_done[name] = str(close)
                self.r.event("DIP_ERROR", market=name, close=str(close), error=repr(exc))
        self.r.save_state()

    def _history(self, mkt: dict, now: pd.Timestamp) -> pd.DataFrame:
        d = mkt["dip"]
        today = now.tz_convert("UTC").date()
        hourly, failed = self.source.hourly(d["source"], today - timedelta(days=int(self.s["history_days"])),
                                            today, tuple(d["price_range"]))
        if failed:
            self.r.event("DIP_DATA_GAPS", market=mkt["name"], failed=failed[:10])
        return hourly

    def _decide(self, name: str, close: pd.Timestamp, equity: float, now: pd.Timestamp) -> None:
        mkt = self.markets[name]
        rules, sched = self.rules(mkt), self.schedule(mkt)
        daily = indicators(daily_bars(self._history(mkt, now), sched), rules)
        if not len(daily) or daily.index[-1] != close:
            self.retry[name] = now + pd.Timedelta(minutes=int(self.s["retry_minutes"]))
            self.r.event("DIP_DATA_NOT_READY", market=name, want=str(close),
                         have=str(daily.index[-1]) if len(daily) else None)
            return
        row = daily.iloc[-1]
        key = sleeve_key(self.sleeve, name)
        pos = self.r.positions.get(key)
        act = decide(row, pos.direction if pos else 0, rules)
        snap = {"close": float(row.close), "sma200": float(row.sma_trend), "sma5": float(row.sma_exit),
                "rsi2": round(float(row.rsi), 2), "atr_pct": round(float(row.atr_pct) * 100, 3)}
        self.r.dip_done[name] = str(close)
        self.r.event("DIP_DECISION", market=name, close_ts=str(close), action=act or "none",
                     position=pos.direction if pos else 0, **snap)
        if act == "exit" and pos:
            self.r._close(key, "dip_exit", now)
        elif act in ("long", "short") and not pos:
            direction = 1 if act == "long" else -1
            info, _ = self.r._market_context(mkt["epic"])
            if info.mid != info.mid:
                return self.r.event("ENTRY_REJECTED", market=name, sleeve=self.sleeve, module="dip",
                                    reason="NO_PRICE")
            self.r.dip_prices[name] = float(info.mid)
            stop_pts = float(rules.stop_atr * row.atr_pct * info.mid)
            sig = Signal(market=name, direction=direction, module="dip", stop_distance=stop_pts, score=1,
                         max_score=1, conditions={"above_200" if direction == 1 else "below_200": True,
                                                  "rsi2_extreme": True},
                         snapshot={"close": float(row.close), "ema_slow": float(row.sma_trend),
                                   "atr": float(row.atr), "rsi": float(row.rsi)})
            self.r._place_entry(name, self.sleeve, sig, snap, equity, now)


def is_dip(pos) -> bool:
    return getattr(pos, "module", "") == "dip"

