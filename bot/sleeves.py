"""Strategy sleeves: one IG account split into independent virtual pots.

Each sleeve owns a slice of the starting balance (weight), trades only its own
modules, sizes from ITS OWN equity (slice + its realised P&L + its open P&L), and has
its own limits: risk per trade, exposure (leverage) cap, open positions, drawdown halt
and loss-streak pause. Account-wide limits in RiskManager still sit on top of all of
them (daily/weekly loss, 15% drawdown halt, total risk, total exposure, per-trade
ceiling as % of the WHOLE account).

With no `sleeves:` section in config, a single sleeve "main" with all modules and the
account-level settings reproduces the original single-strategy behaviour.

CODEX: sleeve limits are ceilings like `risk:`. Never raise SLEEVE_HARD_LIMITS.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

SLEEVE_HARD_LIMITS = {
    "risk_per_trade_pct": 10.0,   # of the sleeve; validate() also caps it as % of the account
    "max_leverage": 30.0,         # IG's retail maximum on major FX
    "max_drawdown_pct": 80.0,
    "max_open_positions": 5,
}
SLEEVE_KEYS = {"modules", "weight", "risk_per_trade_pct", "max_leverage", "max_open_positions",
               "max_positions_per_group", "max_drawdown_pct"}


@dataclass
class SleeveState:
    base: float = 0.0          # starting slice in account currency (set on first run)
    realized: float = 0.0      # closed-trade P&L
    peak: float = 0.0
    trades: int = 0
    wins: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    loss_streak: int = 0
    pause_until: str = ""
    halted: bool = False
    halt_reason: str = ""


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def default_sleeves(cfg: dict) -> dict:
    r = cfg["risk"]
    mods = sorted({m for mk in cfg["markets"] for m in mk["modules"]})
    return {"main": {"modules": mods, "weight": 1.0, "risk_per_trade_pct": r["risk_per_trade_pct"],
                     "max_leverage": SLEEVE_HARD_LIMITS["max_leverage"],
                     "max_open_positions": r["max_open_positions"],
                     "max_positions_per_group": r.get("max_positions_per_group", 1),
                     "max_drawdown_pct": 100.0}}


def sleeve_key(sleeve: str, market: str) -> str:
    return f"{sleeve}:{market}"


class Sleeves:
    def __init__(self, cfg: dict, states: dict | None = None):
        self.spec: dict[str, dict] = cfg.get("sleeves") or default_sleeves(cfg)
        self.r = cfg["risk"]
        states = states or {}
        self.states = {n: SleeveState(**states.get(n, {})) for n in self.spec}
        self.by_module = {m: n for n, s in self.spec.items() for m in s["modules"]}

    # ------------------------------------------------------------ basics
    def names(self) -> list[str]:
        return list(self.spec)

    def modules_for(self, name: str, market_modules: Iterable[str]) -> list[str]:
        mine = set(self.spec[name]["modules"])
        return [m for m in market_modules if m in mine]

    def ensure_bases(self, account_equity: float) -> None:
        for n, st in self.states.items():
            if st.base <= 0:
                st.base = round(self.spec[n]["weight"] * account_equity, 2)
                st.peak = st.base

    def to_dict(self) -> dict:
        return {n: asdict(st) for n, st in self.states.items()}

    # ------------------------------------------------------------ equity
    def equity(self, name: str, positions: Iterable, prices: dict[str, float],
               cost: dict[str, float] | None = None) -> float:
        st = self.states[name]
        unreal = 0.0
        for p in positions:
            if p.sleeve != name:
                continue
            px = prices.get(p.market, p.entry_price) - p.direction * (cost or {}).get(p.market, 0.0)
            unreal += (px - p.entry_price) * p.direction * p.size * p.point_value - p.funding
        return st.base + st.realized + unreal

    def update(self, name: str, equity: float, ts) -> str | None:
        """Track the sleeve's peak; halt NEW entries on a sleeve drawdown breach. Returns reason if halted now."""
        st = self.states[name]
        st.peak = max(st.peak, equity)
        dd = float(self.spec[name].get("max_drawdown_pct", 100.0))
        if not st.halted and dd < 100 and st.peak > 0 and equity <= st.peak * (1 - dd / 100):
            st.halted = True
            st.halt_reason = f"SLEEVE_DRAWDOWN: {name} equity {equity:.2f} <= peak {st.peak:.2f} - {dd}%"
            return st.halt_reason
        return None

    def reset(self, name: str) -> None:
        """Human only (BOT_RESET_SLEEVES). Peak restarts from current equity at the next update."""
        st = self.states[name]
        st.halted, st.halt_reason, st.loss_streak, st.pause_until, st.peak = False, "", 0, "", 0.0

    # ------------------------------------------------------------ sizing inputs
    def risk_pct(self, name: str) -> float:
        pct = float(self.spec[name]["risk_per_trade_pct"])
        if self.states[name].loss_streak >= self.r["loss_streak_halve_risk"]:
            pct /= 2
        return pct

    def min_size_max_risk_pct(self, name: str) -> float | None:
        """Optional human-set ceiling for taking the broker's minimum stake (halved on a loss streak)."""
        v = self.spec[name].get("min_size_max_risk_pct")
        if not v:
            return None
        pct = float(v)
        if self.states[name].loss_streak >= self.r["loss_streak_halve_risk"]:
            pct /= 2
        return pct

    def max_size_by_leverage(self, name: str, equity: float, positions: Iterable,
                             price: float, point_value: float) -> float:
        cap = float(self.spec[name].get("max_leverage", math.inf)) * max(equity, 0.0)
        used = sum(p.notional for p in positions if p.sleeve == name)
        if price <= 0 or point_value <= 0:
            return 0.0
        return max(0.0, (cap - used) / (price * point_value))

    # ------------------------------------------------------------ gate + bookkeeping
    def check(self, name: str, ts, market: str, group: str, positions: Iterable) -> tuple[bool, str]:
        st, s = self.states[name], self.spec[name]
        if st.halted:
            return False, f"SLEEVE_HALTED:{name}"
        if st.pause_until and _ts(ts) < _ts(st.pause_until):
            return False, f"SLEEVE_PAUSED:{name}"
        mine = [p for p in positions if p.sleeve == name]
        if any(p.market == market for p in mine):
            return False, "ALREADY_IN_MARKET"
        if len(mine) >= s["max_open_positions"]:
            return False, f"SLEEVE_MAX_POSITIONS:{name}"
        if sum(1 for p in mine if p.group == group) >= s["max_positions_per_group"]:
            return False, f"SLEEVE_GROUP_LIMIT:{name}"
        return True, "OK"

    def on_trade_closed(self, name: str, pnl: float, ts) -> None:
        st = self.states[name]
        st.realized += pnl
        st.trades += 1
        if pnl > 0:
            st.wins += 1
            st.gross_win += pnl
            st.loss_streak = 0
        else:
            st.gross_loss += -pnl
            st.loss_streak += 1
            if st.loss_streak % self.r["loss_streak_pause"] == 0:
                st.pause_until = (_ts(ts) + pd.Timedelta(hours=self.r["pause_hours"])).isoformat()

    def summary(self, positions: Iterable, prices: dict[str, float]) -> dict:
        positions = list(positions)
        out = {}
        for n, st in self.states.items():
            eq = self.equity(n, positions, prices)
            out[n] = {
                "modules": self.spec[n]["modules"], "start": round(st.base, 2), "equity": round(eq, 2),
                "return_pct": round(100 * (eq / st.base - 1), 2) if st.base else None,
                "realized": round(st.realized, 2), "trades": st.trades,
                "win_rate_pct": round(100 * st.wins / st.trades, 1) if st.trades else None,
                "profit_factor": round(st.gross_win / st.gross_loss, 2) if st.gross_loss else None,
                "open_positions": sum(1 for p in positions if p.sleeve == n),
                "halted": st.halted, "paused_until": st.pause_until or None,
            }
        return out


def validate_sleeves(cfg: dict) -> None:
    from .config import ConfigError   # local import avoids a cycle
    sl = cfg.get("sleeves")
    if not sl:
        return
    seen: dict[str, str] = {}
    total_w = 0.0
    cap = float(cfg["risk"]["max_trade_risk_pct"])
    for name, s in sl.items():
        missing = SLEEVE_KEYS - set(s)
        if missing:
            raise ConfigError(f"sleeve {name} missing keys: {sorted(missing)}")
        for key, ceiling in SLEEVE_HARD_LIMITS.items():
            if float(s[key]) <= 0 or float(s[key]) > ceiling:
                raise ConfigError(f"sleeve {name}: {key}={s[key]} outside (0, {ceiling}]")
        if not 0 < float(s["weight"]) <= 1:
            raise ConfigError(f"sleeve {name}: weight must be in (0, 1]")
        total_w += float(s["weight"])
        acct_risk = float(s["weight"]) * float(s["risk_per_trade_pct"])
        if acct_risk > cap + 1e-9:
            raise ConfigError(f"sleeve {name}: one trade risks {acct_risk:.2f}% of the account > {cap}% ceiling")
        tol = s.get("min_size_max_risk_pct")
        if tol is not None:
            tol = float(tol)
            if tol < float(s["risk_per_trade_pct"]) or tol > SLEEVE_HARD_LIMITS["risk_per_trade_pct"]:
                raise ConfigError(f"sleeve {name}: min_size_max_risk_pct={tol} must be between "
                                  f"risk_per_trade_pct and {SLEEVE_HARD_LIMITS['risk_per_trade_pct']}")
            if float(s["weight"]) * tol > cap + 1e-9:
                raise ConfigError(f"sleeve {name}: a min-size trade could risk {s['weight'] * tol:.2f}% "
                                  f"of the account > {cap}% ceiling")
        for m in s["modules"]:
            if m in seen:
                raise ConfigError(f"module {m} is in two sleeves ({seen[m]}, {name})")
            seen[m] = name
    if total_w > 1 + 1e-9:
        raise ConfigError(f"sleeve weights sum to {total_w:.2f} > 1.0")
