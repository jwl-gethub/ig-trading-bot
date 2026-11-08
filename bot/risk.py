"""Risk engine. The ONLY place position size is decided and the ONLY gate for new entries.

Used identically by backtest and live. CODEX: never add a code path that opens a
position without RiskManager.size() and RiskManager.check_entry() both passing.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import asdict, dataclass, field
from typing import Iterable

import pandas as pd

from .strategy import PositionState


@dataclass
class RiskState:
    equity_peak: float = 0.0
    day_key: str = ""
    day_start_equity: float = 0.0
    week_key: str = ""
    week_start_equity: float = 0.0
    loss_streak: int = 0
    pause_until: str = ""
    halted: bool = False
    halt_reason: str = ""
    halt_code: str = ""
    orders_today: int = 0
    disabled_segments: list = field(default_factory=list)   # "MARKET:module", set by the learning loop
    last_learning_week: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SizeResult:
    ok: bool
    size: float = 0.0
    risk_amount: float = 0.0
    reason: str = "OK"


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class RiskManager:
    def __init__(self, risk_cfg: dict, safety_cfg: dict, state: RiskState | None = None):
        self.r = risk_cfg
        self.s = safety_cfg
        self.state = state or RiskState()

    # ------------------------------------------------------------ bookkeeping
    def on_bar(self, ts, equity: float) -> None:
        """Call once per decision cycle with marked-to-market equity."""
        ts = _ts(ts)
        st = self.state
        day = ts.strftime("%Y-%m-%d")
        iso = ts.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        if day != st.day_key:
            st.day_key, st.day_start_equity, st.orders_today = day, equity, 0
        if week != st.week_key:
            st.week_key, st.week_start_equity = week, equity
        st.equity_peak = max(st.equity_peak, equity)
        if not st.halted and equity <= st.equity_peak * (1 - self.r["max_drawdown_halt_pct"] / 100):
            self.halt(
                f"MAX_DRAWDOWN: equity {equity:.2f} <= peak {st.equity_peak:.2f} "
                f"- {self.r['max_drawdown_halt_pct']}%"
            )

    def on_order_sent(self) -> None:
        self.state.orders_today += 1

    def on_trade_closed(self, pnl: float, ts) -> None:
        st = self.state
        if pnl < 0:
            st.loss_streak += 1
            n = self.r["loss_streak_pause"]
            if st.loss_streak % n == 0:
                st.pause_until = (_ts(ts) + pd.Timedelta(hours=self.r["pause_hours"])).isoformat()
        else:
            st.loss_streak = 0

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            self.state.halt_code = secrets.token_hex(3).upper()
        self.state.halted = True
        self.state.halt_reason = reason

    def reset_halt(self) -> None:
        """Manual only — called by a human via `python -m bot.live --reset-halt`."""
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halt_code = ""
        self.state.loss_streak = 0
        self.state.pause_until = ""

    # ------------------------------------------------------------ sizing
    def current_risk_pct(self) -> float:
        pct = float(self.r["risk_per_trade_pct"])
        if self.state.loss_streak >= self.r["loss_streak_halve_risk"]:
            pct /= 2
        return pct

    def size(self, equity: float, stop_points: float, point_value: float,
             min_size: float, size_step: float, risk_pct: float | None = None,
             max_size: float | None = None) -> SizeResult:
        """Stake = equity x risk% / stop, rounded DOWN. `max_size` (e.g. a leverage cap) can only shrink it."""
        if equity <= 0:
            return SizeResult(False, reason="NO_EQUITY")
        if not stop_points or stop_points <= 0 or math.isnan(stop_points):
            return SizeResult(False, reason="NO_STOP")
        pct = self.current_risk_pct() if risk_pct is None else float(risk_pct)
        budget = equity * pct / 100
        raw = budget / (stop_points * point_value)
        capped = max_size is not None and max_size < raw
        if capped:
            raw = max_size
        size = math.floor(round(raw / size_step, 9)) * size_step
        size = round(size, 8)
        if size < min_size and capped:
            return SizeResult(False, reason=f"LEVERAGE_CAP: room for {max(raw, 0):.2f} < min size {min_size}")
        if size < min_size:
            at_min = min_size * stop_points * point_value
            return SizeResult(
                False,
                reason=(
                    f"UNSIZEABLE: min size {min_size} risks {at_min:.2f} "
                    f"({100 * at_min / equity:.2f}% of equity) > {pct:.2f}% cap"
                ),
            )
        return SizeResult(True, size=size, risk_amount=size * stop_points * point_value)

    # ------------------------------------------------------------ entry gate
    def check_entry(self, ts, mkt: dict, exposures: Iterable[PositionState], equity: float,
                    new_risk: float, new_margin: float, new_notional: float = 0.0,
                    sleeve: str = "") -> tuple[bool, str]:
        ts = _ts(ts)
        st = self.state
        exposures = list(exposures)

        if st.halted:
            return False, f"HALTED: {st.halt_reason}"
        if st.pause_until and ts < _ts(st.pause_until):
            return False, f"PAUSED_LOSS_STREAK until {st.pause_until}"
        if st.day_start_equity and equity <= st.day_start_equity * (1 - self.r["daily_loss_limit_pct"] / 100):
            return False, "DAILY_LOSS_LIMIT"
        if st.week_start_equity and equity <= st.week_start_equity * (1 - self.r["weekly_loss_limit_pct"] / 100):
            return False, "WEEKLY_LOSS_LIMIT"
        if st.orders_today >= self.s["max_orders_per_day"]:
            return False, "MAX_ORDERS_PER_DAY"

        cutoff = self.s.get("friday_cutoff_utc")
        if cutoff and ts.weekday() == 4:
            hh, mm = (int(x) for x in cutoff.split(":"))
            if (ts.hour, ts.minute) >= (hh, mm):
                return False, "FRIDAY_CUTOFF"
        if ts.weekday() >= 5:
            return False, "WEEKEND"

        if any(p.market == mkt["name"] and p.sleeve == sleeve for p in exposures):
            return False, "ALREADY_IN_MARKET"
        ceiling = self.r.get("max_trade_risk_pct")
        if ceiling and new_risk > equity * ceiling / 100 + 1e-9:
            return False, "TRADE_RISK_CEILING"
        if len(exposures) >= self.r["max_open_positions"]:
            return False, "MAX_OPEN_POSITIONS"
        group_n = sum(1 for p in exposures if p.group == mkt["group"])
        if group_n >= self.r["max_positions_per_group"]:
            return False, f"GROUP_LIMIT:{mkt['group']}"

        open_risk = sum(p.open_risk_amount() for p in exposures)
        if open_risk + new_risk > equity * self.r["max_total_open_risk_pct"] / 100 + 1e-9:
            return False, "TOTAL_OPEN_RISK_CAP"
        open_margin = sum(p.margin for p in exposures)
        if open_margin + new_margin > equity * self.r["max_margin_utilisation_pct"] / 100:
            return False, "MARGIN_CAP"
        lev = self.r.get("max_total_leverage")
        if lev and sum(p.notional for p in exposures) + new_notional > equity * lev + 1e-9:
            return False, "TOTAL_LEVERAGE_CAP"
        return True, "OK"
