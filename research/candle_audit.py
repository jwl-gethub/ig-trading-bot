"""Audit of strategy B (retracement candle break) with the whole trade resolved on 1-MINUTE data (RESEARCH ONLY).

Why: research/candle_test.py resolves entry, cancel, stop and target at signal-bar level. Intra-bar ordering is
therefore assumed (see AUDIT NOTES in the job output). Here:

* Setup detection is unchanged (research.candle_patterns.retrace_setups, closed signal bars only).
* From the close of C, 1-minute bars are walked. Whichever of entry level / cancel level is touched first decides
  fill vs cancel. If both are inside the same 1-minute bar the order is unknowable, so it counts as FILLED and then
  (if the stop level is reached in that or any later minute) STOPPED: the worst case. Exception: a bar that OPENS
  beyond the cancel level traded the cancel level first (the open is the first print) -> cancelled.
  A bar that opens through the entry level fills at the open (gap). Expiry after PENDING_BARS signal bars.
* After a fill the 1-minute bars (including the fill minute) are walked for stop / target / time stop (the close of
  signal bar idx + HOLD_BARS, same as the bar-level test). Stop and target in the same minute -> stop first.
  A bar opening through the stop fills the stop at the open.
* Same costs, spread guard, stop buffer, splits, markets and pass criteria as candle_test. One position per market:
  a new fill must come after the previous exit MINUTE (the bar-level test blocks the whole exit signal bar).
* Like-for-like baseline: same number of entries per market/split as the real 1-minute trades, random signal bar,
  random side, entry at the open of a random minute inside the same 2-bar window B can fill in, stop distance (in
  ATR) drawn from the real trades' distribution, same target multiple, same costs and the same 1-minute exit engine.

No parameters are tuned; variants are fixed in advance (B_4H_2R only completes the 2x2 grid and is post-hoc).
"""
from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bot.structure import atr_series

from .candle_patterns import PENDING_BARS, Setup, retrace_setups
from .candle_test import AGG, COLS, OOS_MIN_TRADES, PF_MIN, SEEDS, TF, fmt, log, simulate, split, stat
from .dukascopy import load_range
from .scalp_test import END, MARKETS
from .structure_test import HOLD_BARS, PERIODS_FULL, PERIODS_SHORT, SPREAD_GUARD

# name: (timeframe, target R, status)
AUDIT_VARIANTS = {
    "B_1D_3R": ("1D", 3.0, "PRIMARY"),
    "B_1D_2R": ("1D", 2.0, "secondary"),
    "B_4H_3R_secondary": ("4H", 3.0, "secondary"),
    "B_4H_2R": ("4H", 2.0, "POST-HOC (completes the 2x2 grid only)"),
}
AUDIT_SPLITS = ["development", "validation", "out_of_sample", "all"]
LABELS = ["expired", "cancelled", "skip_risk", "skip_spread", "skip_overlap", "stop", "target", "time"]

NOTES = [
    "N1 resolve_entry: per bar, `cancelled` is tested before `hit`; a bar touching both the cancel level and the "
    "trigger is CANCELLED whatever the order. If the trigger came first this is a filled trade that is not "
    "recorded; it removes trades that would be losers (price then ran to the cancel level and often the stop).",
    "N2 resolve_entry: a bar that gaps/opens through the trigger but whose high exceeds the cancel level is also "
    "cancelled, although the open filled first (and a gap fill is followed by an adverse move to the stop).",
    "N3 resolve_entry: a fill on bar k assumes the trigger trades during bar k; the exit loop then starts at bar k "
    "itself and uses that bar's full high/low (fine for the target, which lies beyond the trigger, and the stop "
    "cannot be hit on the fill bar because high <= cancel < stop by construction of N1).",
    "N4 simulate exit loop: stop is tested before target in the same bar (stop-first, conservative for winners); "
    "a gap through the stop fills at the open; a target hit fills at the target price even if the bar gapped "
    "through it (conservative).",
    "N5 simulate time stop: exit at the CLOSE of signal bar idx+48 (known only at that bar's end; no lookahead), "
    "after stop/target were tested on that bar.",
    "N6 simulate overlap: a new fill is rejected if its signal bar index <= the previous exit bar index, i.e. "
    "the whole exit bar is blocked (bar-level approximation of 'one position').",
    "N7 random_setups (old baseline): fills at the NEXT OPEN (kind 'engulf'), and the stop multiple is measured "
    "from C's close to the stop (|close-stop|/ATR) whereas the real B stop distance is measured from the "
    "trigger (C's low) to the stop, so the old baseline has different R-denominators, a different fill type and "
    "randomly dropped overlapping entries: not like-for-like.",
    "N8 split assignment is by the FILL bar's start timestamp (not exit), consistent; 'pooled' (all) also holds "
    "Nov-Dec 2018 trades that belong to no split.",
    "N9 thin/partial signal bars: resampled 4H/1D bars include partial bars (Sunday evening, holidays, data "
    "holes); these can act as context bars or C. Not lookahead; counted as thin_bars in CANDLE_TEST.",
    "LOOKAHEAD CHECK: ATR(14) at the setup bar uses bars <= idx (Wilder, prev close shift); setup levels use bar "
    "idx only; entry search starts at bar idx+1; no daily/HTF data is mixed in; the 00:00 UTC 1D bar is complete "
    "when C is evaluated. No genuine lookahead bug found in detection; the biases are the intra-bar ordering "
    "assumptions N1/N2 (cancel-before-fill) and the non-like-for-like baseline N7.",
]


def aud(variant: str, section: str, text: str) -> None:
    print(f"REPORT AUDIT|{variant}|{section}| {text}", flush=True)


# ---------------------------------------------------------------- 1-minute engine
@dataclass
class Tape:
    t: np.ndarray      # int64 ns UTC
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray

    @classmethod
    def from_frame(cls, m1: pd.DataFrame) -> "Tape":
        return cls(m1.index.as_unit("ns").asi8.copy(), *(m1[k].to_numpy(float).copy()
                                                          for k in ("open", "high", "low", "close")))


def _first(mask: np.ndarray) -> int:
    if not len(mask):
        return -1
    i = int(mask.argmax())
    return i if mask[i] else -1


def entry_window(tape: Tape, starts: np.ndarray, tf_ns: int, i: int) -> tuple[int, int]:
    """1-minute index range [a, b) from the close of signal bar i to the end of signal bar i + PENDING_BARS."""
    n = len(starts)
    a = int(np.searchsorted(tape.t, starts[i] + tf_ns, "left"))
    k = min(i + PENDING_BARS, n - 1)
    b = int(np.searchsorted(tape.t, starts[k] + tf_ns, "left"))
    return a, max(a, b)


def resolve_entry_1m(s: Setup, tape: Tape, a: int, b: int) -> tuple[str, int, float, bool]:
    """-> (kind, minute index, fill price, both_in_same_minute); kind in expired / cancelled / fill."""
    if b <= a:
        return "expired", -1, 0.0, False
    o, h, lo = tape.o[a:b], tape.h[a:b], tape.l[a:b]
    if s.side == -1:
        cm, hm = h > s.cancel, lo <= s.trigger
    else:
        cm, hm = lo < s.cancel, h >= s.trigger
    ci, hi = _first(cm), _first(hm)
    if ci < 0 and hi < 0:
        return "expired", -1, 0.0, False
    if hi < 0:
        return "cancelled", -1, 0.0, False
    both = False
    if ci >= 0:
        if ci < hi:
            return "cancelled", -1, 0.0, False
        if ci == hi:
            beyond = o[ci] > s.cancel if s.side == -1 else o[ci] < s.cancel
            if beyond:                      # opened beyond the cancel level: it traded first
                return "cancelled", -1, 0.0, False
            both = True                     # unknowable order inside the minute -> worst case: filled
    m = a + hi
    px = float(min(s.trigger, tape.o[m]) if s.side == -1 else max(s.trigger, tape.o[m]))
    return "fill", m, px, both


def exit_walk(tape: Tape, fm: int, he: int | None, side: int, stop: float, target: float):
    """Walk minutes fm..he (inclusive; he None = to the end of data) -> (minute, price, why) or None."""
    last = len(tape.h) - 1 if he is None else he
    pos, chunk = fm, 4096
    while pos <= last:
        end = min(pos + chunk, last + 1)
        hh, ll = tape.h[pos:end], tape.l[pos:end]
        sm = (ll <= stop) if side == 1 else (hh >= stop)
        tm = (hh >= target) if side == 1 else (ll <= target)
        si, ti = _first(sm), _first(tm)
        if si >= 0 and (ti < 0 or si <= ti):          # stop first on a tie
            j = pos + si
            o = tape.o[j]
            return j, float(min(o, stop) if side == 1 else max(o, stop)), "stop"
        if ti >= 0:
            return pos + ti, float(target), "target"
        pos, chunk = end, chunk * 2
    if he is None:
        return None
    return he, float(tape.c[he]), "time"


def hold_end(tape: Tape, starts: np.ndarray, tf_ns: int, idx: int) -> int | None:
    e = idx + HOLD_BARS
    if e >= len(starts):
        return None
    return int(np.searchsorted(tape.t, starts[e] + tf_ns, "left")) - 1


def _trade_row(tape, fm, xm, side, px, xpx, risk, cost, why, idx, atr):
    return {"ts": pd.Timestamp(int(tape.t[fm]), unit="ns", tz="UTC"),
            "xts": pd.Timestamp(int(tape.t[xm]), unit="ns", tz="UTC"), "side": side,
            "r": (side * (xpx - px) - 2 * cost) / risk, "why": why, "cost_r": 2 * cost / risk,
            "idx": idx, "risk_atr": risk / atr}


def simulate_1m(tape: Tape, starts: np.ndarray, tf_ns: int, setups: list[Setup], target_r: float,
                cost: float) -> tuple[pd.DataFrame, dict[int, str], dict]:
    """-> (trades, {setup idx: label}, counters). Entry, cancel, stop, target and time stop on 1-minute bars."""
    trades, labels, free = [], {}, -1
    cnt = Counter()
    for s in setups:
        a, b = entry_window(tape, starts, tf_ns, s.idx)
        kind, fm, px, both = resolve_entry_1m(s, tape, a, b)
        if kind != "fill":
            labels[s.idx] = kind
            continue
        if fm <= free:
            labels[s.idx] = "skip_overlap"
            continue
        side = s.side
        risk = side * (px - s.stop_level)
        if risk <= 0:
            labels[s.idx] = "skip_risk"
            continue
        if 2 * cost > SPREAD_GUARD * risk:
            labels[s.idx] = "skip_spread"
            continue
        stop, target = s.stop_level, px + side * target_r * risk
        ex = exit_walk(tape, fm, hold_end(tape, starts, tf_ns, s.idx), side, stop, target)
        if ex is None:
            break
        xm, xpx, why = ex
        labels[s.idx] = why
        trades.append(_trade_row(tape, fm, xm, side, px, xpx, risk, cost, why, s.idx, s.atr))
        cnt["filled"] += 1
        if both:
            cnt["fill_both_levels_same_minute"] += 1
            cnt["fill_both_then_" + why] += 1
        free = xm
    return pd.DataFrame(trades, columns=COLS + ["idx", "risk_atr"]), labels, dict(cnt)


# ---------------------------------------------------------------- bar-level labelling (replica of candle_test.simulate)
def original_labels(bars: pd.DataFrame, setups: list[Setup], target_r: float, cost: float) -> dict[int, str]:
    """Per-setup outcome label of the ORIGINAL bar-level engine (same logic as candle_test.simulate)."""
    from .candle_patterns import resolve_entry
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(bars)
    out, free_at = {}, -1
    for s in setups:
        f = resolve_entry(s, o, h, lo, n)
        if f is None:
            cancelled = False
            for k in range(s.idx + 1, min(s.idx + 1 + PENDING_BARS, n)):
                if (h[k] > s.cancel) if s.side == -1 else (lo[k] < s.cancel):
                    cancelled = True
                    break
                if (lo[k] <= s.trigger) if s.side == -1 else (h[k] >= s.trigger):
                    break
            out[s.idx] = "cancelled" if cancelled else "expired"
            continue
        fi, px = f
        if fi <= free_at:
            out[s.idx] = "skip_overlap"
            continue
        side = s.side
        risk = side * (px - s.stop_level)
        if risk <= 0:
            out[s.idx] = "skip_risk"
            continue
        if 2 * cost > SPREAD_GUARD * risk:
            out[s.idx] = "skip_spread"
            continue
        stop, target = s.stop_level, px + side * target_r * risk
        why, j = None, fi
        while j < n:
            if (lo[j] <= stop) if side == 1 else (h[j] >= stop):
                why = "stop"
            elif (h[j] >= target) if side == 1 else (lo[j] <= target):
                why = "target"
            elif j - s.idx >= HOLD_BARS:
                why = "time"
            if why:
                break
            j += 1
        if why is None:
            break
        out[s.idx] = why
        free_at = j
    return out


# ---------------------------------------------------------------- like-for-like baseline
def baseline_1m(tape: Tape, starts: np.ndarray, tf_ns: int, atr: np.ndarray, cands: list, target_r: float,
                cost: float) -> pd.DataFrame:
    """cands: [(signal idx, side, entry minute, stop multiple in ATR)] sorted by entry minute."""
    trades, free = [], -1
    for idx, side, fm, mult in cands:
        if fm <= free:
            continue
        px = float(tape.o[fm])
        risk = mult * atr[idx]
        if risk <= 0 or 2 * cost > SPREAD_GUARD * risk:
            continue
        stop, target = px - side * risk, px + side * target_r * risk
        ex = exit_walk(tape, fm, hold_end(tape, starts, tf_ns, idx), side, stop, target)
        if ex is None:
            continue
        xm, xpx, why = ex
        trades.append(_trade_row(tape, fm, xm, side, px, xpx, risk, cost, why, idx, atr[idx]))
        free = xm
    return pd.DataFrame(trades, columns=COLS + ["idx", "risk_atr"])


def baseline_candidates(rng: np.random.Generator, tape: Tape, starts: np.ndarray, tf_ns: int, atr: np.ndarray,
                        periods: dict, counts: dict[str, int], mults: np.ndarray) -> list:
    n = len(starts)
    ok = np.flatnonzero(np.isfinite(atr) & (atr > 0))
    ok = ok[ok < n - 1]
    out = []
    for sp, k in counts.items():
        if not k or sp not in periods:
            continue
        a, b = periods[sp]
        lo_ns = pd.Timestamp(a, tz="UTC").value
        hi_ns = pd.Timestamp(b + " 23:59:59", tz="UTC").value
        pool = ok[(starts[ok] >= lo_ns) & (starts[ok] <= hi_ns)]
        if not len(pool):
            continue
        for i in rng.choice(pool, size=k, replace=k > len(pool)):
            w0, w1 = entry_window(tape, starts, tf_ns, int(i))
            if w1 <= w0:
                continue
            out.append((int(i), int(rng.choice([-1, 1])), int(rng.integers(w0, w1)), float(rng.choice(mults))))
    out.sort(key=lambda x: x[2])
    return out


# ---------------------------------------------------------------- job
def verdict(row: dict) -> bool:
    oos, val = row["out_of_sample"], row["validation"]
    return bool((oos.get("profit_factor") or 0) >= PF_MIN and (val.get("profit_factor") or 0) >= PF_MIN
                and oos.get("trades", 0) >= OOS_MIN_TRADES)


def cat(frames: list) -> pd.DataFrame:
    fr = [f for f in frames if len(f)]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=COLS)


def run(cache: Path) -> None:
    aud("ALL", "header", "CANDLE AUDIT: strategy B resolved on 1-minute data (research only, nothing wired to trading)")
    aud("ALL", "criteria", "PASS criteria unchanged: OOS PF >= 1.2 AND validation PF >= 1.2 AND OOS trades >= 100 "
                           "(pooled over markets, after costs, R units)")
    for k in NOTES:
        aud("ALL", "audit_note", k)
    pool = {v: {"orig": {s: [] for s in AUDIT_SPLITS}, "m1": {s: [] for s in AUDIT_SPLITS}} for v in AUDIT_VARIANTS}
    base = {v: [{s: [] for s in AUDIT_SPLITS} for _ in range(SEEDS)] for v in AUDIT_VARIANTS}
    permkt = {v: {"orig": {}, "m1": {}} for v in AUDIT_VARIANTS}
    trans = {v: Counter() for v in AUDIT_VARIANTS}
    nsetups = Counter()
    counters = {v: Counter() for v in AUDIT_VARIANTS}
    cf = {v: {"all": [], "out_of_sample": []} for v in AUDIT_VARIANTS}      # orig cancelled -> 1-min filled
    for name, (inst, pr, _hours, cost, start) in MARKETS.items():
        try:
            m1 = load_range(inst, start, END, cache, pr, log=lambda *_: None)
            frames = {tf: m1.resample(rule).agg({**AGG, "volume": "count"}).dropna(subset=["open"])
                      for tf, rule in TF.items() if tf in ("4H", "1D")}
            tape = Tape.from_frame(m1)
            del m1
            periods = PERIODS_FULL if start.year < 2024 else PERIODS_SHORT
            setups_by_tf = {}
            for vname, (tf, tgt, _status) in AUDIT_VARIANTS.items():
                bars = frames[tf]
                if tf not in setups_by_tf:
                    setups_by_tf[tf] = retrace_setups(bars)
                setups = setups_by_tf[tf]
                starts = bars.index.as_unit("ns").asi8.copy()
                tf_ns = pd.Timedelta(TF[tf]).value
                t0, sk0 = simulate(bars, setups, tgt, cost)
                lab0 = original_labels(bars, setups, tgt, cost)
                assert sum(v in ("stop", "target", "time") for v in lab0.values()) == len(t0), "label replica mismatch"
                t1, lab1, cnt = simulate_1m(tape, starts, tf_ns, setups, tgt, cost)
                nsetups[vname] += len(setups)
                counters[vname].update(cnt)
                for s in setups:
                    trans[vname][(lab0.get(s.idx, "none"), lab1.get(s.idx, "none"))] += 1
                if len(t1):
                    chg = t1[t1.idx.map(lambda i: lab0.get(i) == "cancelled")]
                    if len(chg):
                        cf[vname]["all"].append(chg)
                        cf[vname]["out_of_sample"].append(split(chg, periods, "out_of_sample"))
                for s_ in AUDIT_SPLITS:
                    pool[vname]["orig"][s_].append(split(t0, periods, s_))
                    pool[vname]["m1"][s_].append(split(t1, periods, s_))
                permkt[vname]["orig"][name] = stat(split(t0, periods, "out_of_sample"))
                permkt[vname]["m1"][name] = stat(split(t1, periods, "out_of_sample"))
                log({"event": "CANDLE_AUDIT_MARKET", "market": name, "variant": vname, "setups": len(setups),
                     "orig_trades": len(t0), "m1_trades": len(t1), **{"m1_" + k: v for k, v in cnt.items()}})
                # like-for-like baseline
                if len(t1):
                    atr = atr_series(bars, 14)
                    mults = t1.risk_atr.to_numpy(float)
                    counts = {s_: len(split(t1, periods, s_)) for s_ in periods}
                    for k in range(SEEDS):
                        rng = np.random.default_rng([k, zlib.crc32((name + vname + "audit").encode())])
                        cands = baseline_candidates(rng, tape, starts, tf_ns, atr, periods, counts, mults)
                        bt = baseline_1m(tape, starts, tf_ns, atr, cands, tgt, cost)
                        for s_ in AUDIT_SPLITS:
                            base[vname][k][s_].append(split(bt, periods, s_))
            del tape, frames
        except Exception as exc:
            log({"event": "CANDLE_AUDIT_ERROR", "market": name, "error": repr(exc)})
            import traceback
            traceback.print_exc()

    summary = {}
    for vname, (tf, tgt, status) in AUDIT_VARIANTS.items():
        rows = {m: {s_: stat(cat(pool[vname][m][s_])) for s_ in AUDIT_SPLITS} for m in ("orig", "m1")}
        ok = {m: verdict(rows[m]) for m in rows}
        summary[vname] = ok
        aud(vname, "title", f"{vname} [{status}] timeframe {tf} target {tgt}R")
        for s_ in AUDIT_SPLITS:
            sn = s_ if s_ != "all" else "pooled"
            aud(vname, f"side_by_side:{sn}", "ORIG-BAR-LEVEL " + fmt(rows["orig"][s_]))
            aud(vname, f"side_by_side:{sn}", "1-MINUTE       " + fmt(rows["m1"][s_]))
        aud(vname, "verdict", f"ORIG-BAR-LEVEL -> {'PASS' if ok['orig'] else 'FAIL'} | "
                              f"1-MINUTE -> {'PASS' if ok['m1'] else 'FAIL'}")
        # outcome changes
        tr = trans[vname]
        total = sum(tr.values())
        same = sum(v for (a, b), v in tr.items() if a == b)
        aud(vname, "outcome_changes", f"setups {total}; same label in both engines {same}; changed {total - same}")
        for (a, b), v in sorted(tr.items(), key=lambda kv: -kv[1]):
            if a != b:
                aud(vname, "outcome_changes", f"orig={a} -> 1min={b}: {v}")
        c = counters[vname]
        aud(vname, "outcome_changes", f"1min fills {c['filled']}; fills where entry and cancel level were both in "
                                      f"the same minute (counted filled, worst case) {c['fill_both_levels_same_minute']}; "
                                      f"of which stop {c['fill_both_then_stop']} target {c['fill_both_then_target']} "
                                      f"time {c['fill_both_then_time']}")
        for part in ("all", "out_of_sample"):
            g = cat(cf[vname][part])
            aud(vname, f"orig_cancelled_now_filled:{part}",
                fmt(stat(g)) if len(g) else "trades 0")
        # baseline
        brow = []
        for s_ in ("validation", "out_of_sample"):
            pfs, avs, ns = [], [], []
            for k in range(SEEDS):
                m = stat(cat(base[vname][k][s_]))
                if m.get("profit_factor") is not None:
                    pfs.append(m["profit_factor"])
                    avs.append(m["avg_r"])
                    ns.append(m["trades"])
            real = rows["m1"][s_].get("profit_factor")
            if pfs:
                beat = sum(p >= (real or 0) for p in pfs)
                aud(vname, f"baseline_1min:{s_}",
                    f"random-entry 1-min-engine PF mean {np.mean(pfs):.2f} p95 {np.percentile(pfs, 95):.2f} "
                    f"avgR mean {np.mean(avs):.3f} mean trades {np.mean(ns):.0f} (real 1-min trades "
                    f"{rows['m1'][s_].get('trades')}, real PF {real}); random >= real PF in {beat}/{len(pfs)} seeds")
            else:
                aud(vname, f"baseline_1min:{s_}", "no baseline trades")
        log({"event": "CANDLE_AUDIT_SUMMARY", "variant": vname, "orig": rows["orig"], "m1": rows["m1"], "pass": ok})
    aud("ALL", "per_market_oos_header", "PER-MARKET OUT-OF-SAMPLE")
    for vname in AUDIT_VARIANTS:
        for mk in permkt[vname]["m1"]:
            aud(vname, f"per_market_oos:{mk}", "ORIG-BAR-LEVEL " + fmt(permkt[vname]["orig"][mk]))
            aud(vname, f"per_market_oos:{mk}", "1-MINUTE       " + fmt(permkt[vname]["m1"][mk]))
    for vname, ok in summary.items():
        aud(vname, "final", f"ORIG {'PASS' if ok['orig'] else 'FAIL'} | 1-MINUTE {'PASS' if ok['m1'] else 'FAIL'}")
    aud("ALL", "final", "1-MINUTE PASSED: " + (", ".join(v for v, o in summary.items() if o["m1"]) or "none"))
    log({"event": "CANDLE_AUDIT_DONE", "passed_1min": [v for v, o in summary.items() if o["m1"]]})
