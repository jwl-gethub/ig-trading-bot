"""Regime-switching confluence strategy. Spec: STRATEGY.md.

Pure logic: no I/O, no broker calls. The same code runs in backtest and live.
Rows are dicts produced by indicators.compute_all() for a CLOSED bar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

SNAPSHOT_KEYS = (
    "close", "ema_fast", "ema_mid", "ema_slow", "adx", "plus_di", "minus_di", "macd_hist",
    "rsi", "rsi_fast", "bb_lower", "bb_mid", "bb_upper", "dc_high", "dc_low", "atr",
    "atr_pctile", "regime",
)


@dataclass
class Signal:
    market: str
    direction: int          # +1 long, -1 short
    module: str             # "trend" | "meanrev"
    stop_distance: float    # price points
    score: int
    max_score: int
    conditions: dict
    snapshot: dict


@dataclass
class PositionState:
    market: str
    direction: int
    module: str
    entry_price: float
    stop_level: float
    size: float
    point_value: float
    initial_risk_points: float
    entry_ts: str
    bars_held: int = 0
    extreme: float = 0.0     # highest high (long) / lowest low (short) since entry
    funding: float = 0.0
    deal_id: str | None = None
    margin: float = 0.0
    group: str = ""

    def open_risk_amount(self) -> float:
        """Money still at risk to the stop, relative to entry (0 once stop is past breakeven)."""
        pts = (self.entry_price - self.stop_level) * self.direction
        return max(0.0, pts) * self.size * self.point_value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExitDecision:
    close: bool = False
    new_stop: float | None = None
    reason: str = ""
    detail: dict = field(default_factory=dict)


def _snap(row: Mapping) -> dict:
    out = {}
    for k in SNAPSHOT_KEYS:
        v = row.get(k)
        out[k] = round(float(v), 5) if isinstance(v, (int, float)) and k != "regime" else v
    return out


class Strategy:
    def __init__(self, params: dict):
        self.p = params

    # ------------------------------------------------------------ entries
    def entry_signal(self, mkt: Mapping, row: Mapping) -> Signal | None:
        if not row.get("ready") or not row.get("vol_ok"):
            return None
        modules = mkt.get("modules", ["trend", "meanrev"])
        directions = (1, -1) if mkt.get("allow_short", False) else (1,)
        regime = row.get("regime")

        if regime == "TREND" and "trend" in modules:
            for d in directions:
                sig = self._trend(mkt, row, d)
                if sig:
                    return sig
        if regime == "RANGE" and "meanrev" in modules:
            for d in directions:
                sig = self._meanrev(mkt, row, d)
                if sig:
                    return sig
        return None

    def _trend(self, mkt: Mapping, r: Mapping, d: int) -> Signal | None:
        t = self.p["trend"]
        if d == 1:
            cond = {
                "bias": r["close"] > r["ema_slow"],
                "ema_stack": r["ema_fast"] > r["ema_mid"],
                "di": r["plus_di"] > r["minus_di"],
                "macd": r["macd_hist"] > 0,
                "rsi": t["rsi_long_min"] <= r["rsi"] <= t["rsi_long_max"],
            }
            breakout = r["close"] > r["dc_high"]
            pullback = bool(r["touched_fast_long"]) and r["close"] > r["ema_fast"] and r["close"] > r["prev_high"]
        else:
            cond = {
                "bias": r["close"] < r["ema_slow"],
                "ema_stack": r["ema_fast"] < r["ema_mid"],
                "di": r["minus_di"] > r["plus_di"],
                "macd": r["macd_hist"] < 0,
                "rsi": 100 - t["rsi_long_max"] <= r["rsi"] <= 100 - t["rsi_long_min"],
            }
            breakout = r["close"] < r["dc_low"]
            pullback = bool(r["touched_fast_short"]) and r["close"] < r["ema_fast"] and r["close"] < r["prev_low"]
        cond["trigger"] = bool(breakout or pullback)
        cond = {k: bool(v) for k, v in cond.items()}
        score = sum(cond.values())
        if cond["bias"] and cond["trigger"] and score >= t["min_score"]:
            return Signal(
                market=mkt["name"], direction=d, module="trend",
                stop_distance=float(t["stop_atr"] * r["atr"]), score=score, max_score=len(cond),
                conditions={**cond, "breakout": bool(breakout), "pullback": bool(pullback)},
                snapshot=_snap(r),
            )
        return None

    def _meanrev(self, mkt: Mapping, r: Mapping, d: int) -> Signal | None:
        m = self.p["meanrev"]
        if d == 1:
            cond = {
                "below_bb": r["close"] < r["bb_lower"],
                "rsi_fast_extreme": r["rsi_fast"] < m["rsi_fast_entry"],
                "rsi_slow_weak": r["rsi"] < m["rsi_slow_max"],
                "trend_filter": r["close"] > r["ema_slow"],
            }
        else:
            cond = {
                "below_bb": r["close"] > r["bb_upper"],
                "rsi_fast_extreme": r["rsi_fast"] > 100 - m["rsi_fast_entry"],
                "rsi_slow_weak": r["rsi"] > 100 - m["rsi_slow_max"],
                "trend_filter": r["close"] < r["ema_slow"],
            }
        cond = {k: bool(v) for k, v in cond.items()}
        if all(cond.values()):
            return Signal(
                market=mkt["name"], direction=d, module="meanrev",
                stop_distance=float(m["stop_atr"] * r["atr"]), score=len(cond), max_score=len(cond),
                conditions=cond, snapshot=_snap(r),
            )
        return None

    # ------------------------------------------------------------ exits
    def manage(self, pos: PositionState, r: Mapping) -> ExitDecision:
        """Call once per CLOSED bar while a position is open. Updates bars_held/extreme."""
        d = pos.direction
        pos.bars_held += 1
        pos.extreme = max(pos.extreme, r["high"]) if d == 1 else min(pos.extreme, r["low"])

        if pos.module == "trend":
            t = self.p["trend"]
            failed = r["ema_fast"] < r["ema_mid"] if d == 1 else r["ema_fast"] > r["ema_mid"]
            if failed:
                return ExitDecision(close=True, reason="trend_failure")
            trail = pos.extreme - d * t["trail_atr"] * r["atr"]
            tighter = trail > pos.stop_level if d == 1 else trail < pos.stop_level
            right_side = trail < r["close"] if d == 1 else trail > r["close"]
            if tighter and right_side:
                return ExitDecision(new_stop=float(trail), reason="trail")
            return ExitDecision()

        m = self.p["meanrev"]
        if d == 1 and (r["close"] >= r["bb_mid"] or r["rsi_fast"] > m["rsi_fast_exit"]):
            return ExitDecision(close=True, reason="mr_target")
        if d == -1 and (r["close"] <= r["bb_mid"] or r["rsi_fast"] < 100 - m["rsi_fast_exit"]):
            return ExitDecision(close=True, reason="mr_target")
        if pos.bars_held >= m["max_bars"]:
            return ExitDecision(close=True, reason="time_stop")
        return ExitDecision()
