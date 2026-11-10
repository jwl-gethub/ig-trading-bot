"""15-minute scalper inside the live bot (rules in bot/scalp.py, tested in research/scalp_test.py).

Every runner tick (about once a minute) it reads each scalp market's live price from IG's market
snapshot (not the historical-data allowance) and builds its own 15-minute bars. History is seeded
once from IG (MINUTE_15) when the allowance permits and kept on disk, so restarts only fetch the gap.
On each closed bar it applies the rules and sends entries through the runner's normal path (sleeve
sizing, RiskManager checks, spread guard, server-side stop and take-profit, dry-run, kill/pause).
Open scalp trades are closed at market after `max_bars` bars. It never touches other modules' trades.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scalp import ScalpRules, decide, in_session, indicators
from .sleeves import sleeve_key
from .strategy import Signal

BAR = pd.Timedelta(minutes=15)
DEFAULTS = {"enabled": True, "sleeve": "scalp", "seed_bars": 260, "allowance_reserve": 1000, "keep_bars": 600,
            "rules": {}}
RULE_KEYS = set(ScalpRules.__dataclass_fields__) - {"long_only"}


def settings(cfg: dict) -> dict:
    s = {**DEFAULTS, **(cfg.get("scalp") or {})}
    s["rules"] = {k: v for k, v in (s.get("rules") or {}).items() if k in RULE_KEYS}
    return s


class ScalpEngine:
    def __init__(self, runner):
        self.r = runner
        self.s = settings(runner.cfg)
        self.sleeve = self.s["sleeve"]
        self.markets = {n: m for n, m in runner.mkts.items() if "scalp" in m["modules"]}
        self.enabled = bool(self.s["enabled"]) and bool(self.markets) and self.sleeve in runner.sleeves.names()
        self.dir = Path(runner.state_dir) / "scalp"
        self.bars: dict[str, pd.DataFrame] = {}
        self.forming: dict[str, dict] = {}
        self.decided: dict[str, pd.Timestamp] = {}
        self.off: set[str] = set()

    def rules(self, mkt: dict) -> ScalpRules:
        return ScalpRules(long_only=bool(mkt["scalp"].get("long_only", False)), **self.s["rules"])

    # ------------------------------------------------------------ history
    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.csv"

    def _save(self, name: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bars[name].tail(int(self.s["keep_bars"])).to_csv(self._path(name))

    def startup(self, now: pd.Timestamp) -> None:
        if not self.enabled:
            return
        for name, mkt in self.markets.items():
            try:
                df = pd.DataFrame(columns=["open", "high", "low", "close"])
                if self._path(name).exists():
                    df = pd.read_csv(self._path(name), index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
                need = int(self.s["seed_bars"])
                gap = int((now - (df.index[-1] + BAR)) / BAR) + 2 if len(df) else need
                n = need if (gap >= need or len(df) < 210) else gap
                left = getattr(self.r.broker, "allowance_remaining", None)
                fetched = 0
                if n > 2 and (left is None or left - n >= int(self.s["allowance_reserve"])):
                    fresh = self.r.broker.bars(mkt["epic"], "MINUTE_15", n)[["open", "high", "low", "close"]]
                    fresh = fresh[fresh.index + BAR <= now]
                    fetched = len(fresh)
                    if gap >= need:
                        df = fresh
                    else:
                        df = pd.concat([df, fresh])
                        df = df[~df.index.duplicated(keep="last")].sort_index()
                elif n > 2:
                    df = df.iloc[0:0] if gap > 8 else df      # stale history is worse than none
                self.bars[name] = df.astype(float).tail(int(self.s["keep_bars"]))
                self._save(name)
                self.r.event("SCALP_READY", market=name, bars=int(len(df)), fetched=int(fetched),
                             warm=bool(len(df) >= 210), allowance_left=getattr(self.r.broker, "allowance_remaining", None))
            except Exception as exc:
                self.off.add(name)
                self.r.event("SCALP_MARKET_OFF", market=name, error=repr(exc))

    # ------------------------------------------------------------ every tick
    def tick(self, now: pd.Timestamp) -> None:
        if not self.enabled:
            return
        mine = {k: p for k, p in self.r.positions.items() if p.module == "scalp"}
        if mine:
            self.r.reconcile(now)                       # notice server-side stop / take-profit fills fast
        closed_now = []
        for name, mkt in self.markets.items():
            if name in self.off:
                continue
            try:
                info = self.r.broker.market_info(mkt["epic"])
            except Exception as exc:
                self.r.event("SCALP_PRICE_ERROR", market=name, error=repr(exc))
                continue
            mid = info.mid
            if not info.tradeable or mid != mid:
                continue
            self.r.dip_prices[name] = float(mid)
            key = sleeve_key(self.sleeve, name)
            if key in self.r.positions:
                hi, lo = self.r.exit_hints.get(key, (mid, mid))
                self.r.exit_hints[key] = (max(hi, mid), min(lo, mid))
            start = now.floor("15min")
            f = self.forming.get(name)
            if f and f["start"] != start:               # the bar we were building has closed
                df = self.bars.get(name, pd.DataFrame(columns=["open", "high", "low", "close"]))
                contiguous = len(df) == 0 or f["start"] - df.index[-1] <= 2 * BAR
                if not contiguous:
                    df = df.iloc[0:0]                   # a hole in the data: start again rather than guess
                df.loc[f["start"]] = [f["open"], f["high"], f["low"], f["close"]]
                self.bars[name] = df.tail(int(self.s["keep_bars"]))
                self._save(name)
                if start - f["start"] == BAR:
                    closed_now.append(name)
                f = None
            if not f:
                self.forming[name] = {"start": start, "open": mid, "high": mid, "low": mid, "close": mid}
            else:
                f.update(high=max(f["high"], mid), low=min(f["low"], mid), close=mid)
        if closed_now:
            equity = self.r.broker.equity()
            for name in closed_now:
                try:
                    self._on_bar(name, equity, now)
                except Exception as exc:                # fail closed for this bar
                    self.r.event("SCALP_ERROR", market=name, error=repr(exc))
            self.r.save_state()

    def _on_bar(self, name: str, equity: float, now: pd.Timestamp) -> None:
        mkt = self.markets[name]
        rules = self.rules(mkt)
        key = sleeve_key(self.sleeve, name)
        pos = self.r.positions.get(key)
        if pos:
            held = int((now - pd.Timestamp(pos.entry_ts)) / BAR)
            if held >= rules.max_bars:
                self.r._close(key, "scalp_time", now)
                self.r.exit_hints.pop(key, None)
            return
        df = self.bars[name]
        if len(df) < rules.ema_trend + 10:
            return
        row = indicators(df, rules).iloc[-1]
        hours = tuple(mkt["scalp"]["hours_utc"])
        act = decide(row, rules) if in_session(now, hours) else None
        if not act:
            return
        side = 1 if act == "long" else -1
        if side == -1 and not mkt.get("allow_short", True):
            return
        snap = {"close": float(row.close), "ema_fast": float(row.ema_fast), "ema_slow": float(row.ema_slow),
                "ema_trend": float(row.ema_trend), "rsi": round(float(row.rsi), 2), "atr": float(row.atr)}
        sig = Signal(market=name, direction=side, module="scalp", stop_distance=float(rules.stop_atr * row.atr),
                     score=1, max_score=1, conditions={"trend": True, "rsi2_pullback": True}, snapshot=snap,
                     target_distance=float(rules.target_atr * row.atr))
        self.r._info_cache.pop(mkt["epic"], None)       # size and spread-check on a fresh quote
        self.r._place_entry(name, self.sleeve, sig, snap, equity, now)
        if key in self.r.positions:
            self.r.exit_hints[key] = (self.r.dip_prices[name], self.r.dip_prices[name])
