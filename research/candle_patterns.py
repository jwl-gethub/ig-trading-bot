"""Pure signal logic for two candle-pattern strategies (RESEARCH ONLY, not wired into the bot).

A) "manipulation engulfing": two same-direction candles, then a candle that sweeps the previous
   candle's extreme and closes back through the other side of it. Entry at the next bar's open.
B) "retracement candle break": 2+ candles in one direction, then one counter-direction candle C.
   Stop order beyond C's far side, valid for the next 2 bars; cancelled if the other side of C trades first.

Everything at bar i uses bars <= i only (closed bars). Longs are the exact mirror of shorts.
Entries are described by Setup; `resolve_entry` walks forward bars to find the fill (simulation step).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.structure import atr_series

N_CONTEXT = 2
ATR_N = 14
STOP_BUFFER_ATR = 0.1     # same buffer as bot.structure.StructureRules.stop_buffer_atr
PENDING_BARS = 2          # strategy B: order valid for the next 2 bars only


@dataclass(frozen=True)
class Setup:
    kind: str              # "engulf" or "retrace"
    idx: int               # signal bar (closed)
    side: int              # +1 long, -1 short
    stop_level: float
    atr: float
    trigger: float | None = None   # retrace: stop-order level (C's low for shorts, C's high for longs)
    cancel: float | None = None    # retrace: cancel level (C's high for shorts, C's low for longs)


def _arrays(df: pd.DataFrame):
    return tuple(df[k].to_numpy(float) for k in ("open", "high", "low", "close"))


def engulf_setups(df: pd.DataFrame, rule: str = "high", n_ctx: int = N_CONTEXT, atr_n: int = ATR_N,
                  buffer_atr: float = STOP_BUFFER_ATR) -> list[Setup]:
    """rule 'high': long closes above the previous high (short: below the previous low).
    rule 'close': long closes above the previous close (short: below the previous close)."""
    if rule not in ("high", "close"):
        raise ValueError(rule)
    o, h, lo, c = _arrays(df)
    atr = atr_series(df, atr_n)
    out = []
    for i in range(max(n_ctx, 1) + 1, len(df)):
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        p = i - 1
        ctx = range(i - n_ctx, i)
        if all(c[k] > o[k] for k in ctx) and lo[i] < lo[p] and c[i] > (h[p] if rule == "high" else c[p]):
            out.append(Setup("engulf", i, 1, lo[i] - buffer_atr * a, a))
        elif all(c[k] < o[k] for k in ctx) and h[i] > h[p] and c[i] < (lo[p] if rule == "high" else c[p]):
            out.append(Setup("engulf", i, -1, h[i] + buffer_atr * a, a))
    return out


def retrace_setups(df: pd.DataFrame, n_ctx: int = N_CONTEXT, atr_n: int = ATR_N,
                   buffer_atr: float = STOP_BUFFER_ATR) -> list[Setup]:
    """Short: >= n_ctx bearish candles then a bullish candle C -> sell stop at C's low, stop above C's high.
    Long is the mirror. Only the n_ctx candles immediately before C are tested (so '>= n_ctx')."""
    o, h, lo, c = _arrays(df)
    atr = atr_series(df, atr_n)
    out = []
    for i in range(n_ctx, len(df)):
        a = atr[i]
        if not (np.isfinite(a) and a > 0):
            continue
        ctx = range(i - n_ctx, i)
        if all(c[k] < o[k] for k in ctx) and c[i] > o[i]:
            out.append(Setup("retrace", i, -1, h[i] + buffer_atr * a, a, trigger=lo[i], cancel=h[i]))
        elif all(c[k] > o[k] for k in ctx) and c[i] < o[i]:
            out.append(Setup("retrace", i, 1, lo[i] - buffer_atr * a, a, trigger=h[i], cancel=lo[i]))
    return out


def resolve_entry(s: Setup, o: np.ndarray, h: np.ndarray, lo: np.ndarray, n: int) -> tuple[int, float] | None:
    """-> (fill bar index, fill price) or None if the setup never fills / is cancelled."""
    if s.kind == "engulf":
        return (s.idx + 1, float(o[s.idx + 1])) if s.idx + 1 < n else None
    for k in range(s.idx + 1, min(s.idx + 1 + PENDING_BARS, n)):
        if s.side == -1:
            cancelled, hit = h[k] > s.cancel, lo[k] <= s.trigger
        else:
            cancelled, hit = lo[k] < s.cancel, h[k] >= s.trigger
        if cancelled:                      # includes a bar that touches both levels (conservative)
            return None
        if hit:                            # gap through the level fills at the open
            return k, float(min(s.trigger, o[k]) if s.side == -1 else max(s.trigger, o[k]))
    return None
