"""Strategy 2 (turn of the month, indices) and strategy 3 (RSI(2) short side, "rip-sell"). Research only.

Both work on trade tables with columns: ts (entry, UTC), xts (exit, UTC), r (R multiple, net), ret (net return on
entry price, fraction), why.  R = the disaster-stop distance from the fill (strategy 2: 3 x ATR(20); strategy 3:
2.5 x ATR(14), as in the dip module).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from bot.dipbuy import DipRules, daily_bars, fill_time, indicators

from .cand_common import (SPLIT_NAMES, UNIVERSE, Mkt, f2, funding_frac, max_dd, pf, say, split_mask, STRESS)
from .dip_markets import EU_INDEX, FX_DAY, NY_INDEX, UK_INDEX

SEEDS = 20
SCHEDS = {"NY_INDEX": NY_INDEX, "EU_INDEX": EU_INDEX, "UK_INDEX": UK_INDEX, "FX_DAY": FX_DAY}
COLS = ["ts", "xts", "r", "ret", "why"]
NS = 10 ** 9
EMPTY = pd.DataFrame(columns=COLS)


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


class Hr:
    """Hourly arrays for stop scanning."""

    def __init__(self, h: pd.DataFrame):
        self.ts = h.index.as_unit("ns").asi8
        self.o, self.hi, self.lo = (h[k].to_numpy(float) for k in ("open", "high", "low"))
        self.n = len(h)


def atr_simple(d: pd.DataFrame, n: int = 20) -> np.ndarray:
    pc = d.close.shift()
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    return tr.rolling(n).mean().to_numpy()


# ------------------------------------------------------------------------------------------ statistics
def tstats(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    t = t.sort_values("xts")
    r, ret = t.r.to_numpy(float), t.ret.to_numpy(float)
    return {"trades": int(len(t)), "win": 100 * float((ret > 0).mean()), "pf_r": pf(r), "pf_pct": pf(ret),
            "avg_r": float(r.mean()), "tot_r": float(r.sum()), "dd_r": max_dd(r),
            "avg_pct": 100 * float(ret.mean()), "tot_pct": 100 * float(ret.sum()), "dd_pct": 100 * max_dd(ret)}


def fmt(s: dict) -> str:
    if not s.get("trades"):
        return "trades=0"
    return (f"trades={s['trades']} win%={f2(s['win'], 1)} PF={f2(s['pf_r'])} PF_pct={f2(s['pf_pct'])} "
            f"avgR={f2(s['avg_r'], 3)} totR={f2(s['tot_r'], 1)} maxDD_R={f2(s['dd_r'], 1)} "
            f"avg_ret%={f2(s['avg_pct'], 3)} total_ret%={f2(s['tot_pct'], 1)} maxDD%={f2(s['dd_pct'], 1)}")


def pooled(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fr = [t for t in tables.values() if len(t)]
    return pd.concat(fr, ignore_index=True) if fr else empty()


def in_split(t: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "pooled" or not len(t):
        return t
    return t[split_mask(pd.DatetimeIndex(t.ts), name)]


def report_tables(strat: str, variant: str, section: str, label: str, tables: dict[str, pd.DataFrame]) -> dict:
    """Print one line per split (+pooled); return {split: stats}."""
    p = pooled(tables)
    out = {}
    for s in SPLIT_NAMES + ["pooled"]:
        out[s] = tstats(in_split(p, s))
        say(strat, variant, section, f"set={label} split={s} {fmt(out[s])}")
    return out


def per_market_oos(strat: str, variant: str, tables: dict[str, pd.DataFrame]) -> None:
    for mk, t in tables.items():
        say(strat, variant, "per_market_oos", f"market={mk} split=out_of_sample {fmt(tstats(in_split(t, 'out_of_sample')))}")


def baseline_lines(strat: str, variant: str, real: dict[str, pd.DataFrame], allhold: dict[str, pd.DataFrame],
                   what: str, key: str = "pf_r") -> dict:
    """Random-entry baseline: for each market and split draw as many rows from the all-days table as the real run has
    trades there. -> {split: (pf_p95, avg_pct_p95)}. Also prints the unconditional (all days) average."""
    out = {}
    for s in SPLIT_NAMES + ["pooled"]:
        pools, counts = {}, {}
        for mk, t in real.items():
            n = len(in_split(t, s))
            a = in_split(allhold.get(mk, empty()), s)
            if n and len(a):
                pools[mk], counts[mk] = a, n
        if not counts:
            continue
        pfs, avgs, avgr = [], [], []
        for k in range(SEEDS):
            pick = []
            for mk, a in pools.items():
                rng = np.random.default_rng([k, zlib.crc32((mk + s + what).encode())])
                pick.append(a.iloc[rng.choice(len(a), size=counts[mk], replace=counts[mk] > len(a))])
            st = tstats(pd.concat(pick, ignore_index=True))
            if st.get(key) is not None:
                pfs.append(st[key])
            avgs.append(st["avg_pct"])
            avgr.append(st["avg_r"])
        allp = pd.concat([in_split(allhold[mk], s) for mk in pools], ignore_index=True)
        u = tstats(allp)
        out[s] = (float(np.percentile(pfs, 95)) if pfs else None, float(np.percentile(avgs, 95)))
        say(strat, variant, "baseline",
            f"split={s} {what}: PF mean={f2(np.mean(pfs) if pfs else None)} p95={f2(out[s][0])} | avg_ret% mean="
            f"{f2(np.mean(avgs), 3)} p95={f2(out[s][1], 3)} | avgR mean={f2(np.mean(avgr), 3)} (seeds={SEEDS}, "
            f"n per seed={sum(counts.values())}) | PF basis={key} | unconditional all-days: n={u['trades']} "
            f"avg_ret%={f2(u['avg_pct'], 3)} PF={f2(u.get(key))} avgR={f2(u['avg_r'], 3)}")
    return out


# ------------------------------------------------------------------------------------------ strategy 2
def tom_schedule(dates: pd.DatetimeIndex, entry_off: int, exit_day: int = 3, min_days: int = 10) -> list[tuple[int, int]]:
    """(entry_idx, exit_idx) per month: entry = close of trading day -entry_off of the month (-1 = last trading day),
    exit = close of trading day +exit_day of the NEXT calendar month. Trading days = days present in `dates`.
    Needs the following month to be present and consecutive; incomplete leading months are skipped."""
    ym = np.asarray(dates.year) * 12 + np.asarray(dates.month) - 1
    groups: dict[int, list[int]] = {}
    for i, k in enumerate(ym):
        groups.setdefault(int(k), []).append(i)
    out = []
    for k in sorted(groups):
        nxt = groups.get(k + 1)
        cur = groups[k]
        if not nxt or len(cur) < max(entry_off, min_days) or len(nxt) < exit_day:
            continue
        out.append((cur[-entry_off], nxt[exit_day - 1]))
    return out


def sim_tom(m: Mkt, d: pd.DataFrame, atr: np.ndarray, hr: Hr, i: int, j: int, stress: float = 1.0) -> dict | None:
    hs = m.spread * stress / 2
    c = d.close.to_numpy(float)
    a = atr[i]
    if not np.isfinite(a) or a <= 0:
        return None
    entry = c[i] + hs
    stop = entry - 3 * a
    t0, t1 = int(d.close_ts.iloc[i].value), int(d.close_ts.iloc[j].value)
    k0, k1 = np.searchsorted(hr.ts, t0, "left"), np.searchsorted(hr.ts, t1, "left")
    hit = np.flatnonzero(hr.lo[k0:k1] <= stop)
    if len(hit):
        k = k0 + int(hit[0])
        px, why, tout = min(hr.o[k], stop), "stop", int(hr.ts[k])
    else:
        px, why, tout = c[j], "time", t1
    exit_ = px - hs
    fund = funding_frac(m, 1, [t0], [tout], stress)[0]
    pnl = exit_ - entry - fund * entry
    return {"ts": pd.Timestamp(t0, unit="ns", tz="UTC"), "xts": pd.Timestamp(tout, unit="ns", tz="UTC"),
            "r": pnl / (3 * a), "ret": pnl / entry, "why": why}


def tom_market(m: Mkt, d: pd.DataFrame, hr: Hr, entry_off: int, stress: float = 1.0, want_all: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (trades for the real schedule, all-days same-length holds)."""
    atr = atr_simple(d, 20)
    sched = tom_schedule(d.index, entry_off)
    rows = [x for x in (sim_tom(m, d, atr, hr, i, j, stress) for i, j in sched) if x]
    real = pd.DataFrame(rows, columns=COLS) if rows else empty()
    hold = int(np.median([j - i for i, j in sched])) if sched else 4
    allr = [x for x in (sim_tom(m, d, atr, hr, i, i + hold, stress) for i in range(len(d) - hold)) if x] if want_all else []
    return real, (pd.DataFrame(allr, columns=COLS) if allr else empty())


def run_tom(daily: dict[str, pd.DataFrame], hours: dict[str, Hr]) -> list[str]:
    S = "TOM"
    crit = ("PASS (primary only) = validation PF >= 1.2 AND OOS PF >= 1.2 after costs AND OOS trades >= 100 pooled over "
            "the primary five markets AND OOS average return per trade > random-day p95")
    say(S, "ALL", "criteria", crit)
    say(S, "ALL", "conventions", "primary: long at close of day -2, exit close of day +3 of next month; secondary: day -5; "
        "entry/exit at the daily close (last hourly bar before 21:00 UTC) +/- half spread; disaster stop 3 x ATR(20) "
        "(simple mean of true range) below the fill, checked on hourly bars, stop-first; the PASS PF is PF_pct (on % returns); "
        "PF (R-based, R = 3xATR stop distance) is also printed; funding long: index benchmark + 2.5%/365 per night, Friday x3")
    passed = []
    for variant, off in (("primary", 2), ("secondary", 5)):
        prim = [m for m in ("US500", "USTECH", "US30", "GER40", "UK100") if m in daily]
        say(S, variant, "markets", f"primary set available: {','.join(prim)}")
        real, allh, real_s = {}, {}, {}
        for mk in prim + [x for x in ("FRA40", "JPN225", "AUS200") if x in daily]:
            real[mk], allh[mk] = tom_market(UNIVERSE[mk], daily[mk], hours[mk], off)
            real_s[mk] = tom_market(UNIVERSE[mk], daily[mk], hours[mk], off, STRESS, False)[0]
        P = {k: v for k, v in real.items() if k in prim}
        res = report_tables(S, variant, "splits", "primary5", P)
        per_market_oos(S, variant, P)
        extra = {k: v for k, v in real.items() if k not in prim}
        if extra:
            report_tables(S, variant, "extra_markets", "extra(" + ",".join(extra) + ")", extra)
            for mk, t in extra.items():
                say(S, variant, "extra_markets", f"market={mk} split=out_of_sample {fmt(tstats(in_split(t, 'out_of_sample')))}")
        report_tables(S, variant, "cost_stress", f"primary5 costs x{STRESS}", {k: v for k, v in real_s.items() if k in prim})
        base = baseline_lines(S, variant, P, {k: v for k, v in allh.items() if k in prim}, "random same-length hold days", key="pf_pct")
        if variant == "primary":
            v, o = res["validation"], res["out_of_sample"]
            b = base.get("out_of_sample", (None, None))
            c = [v.get("pf_pct") and v["pf_pct"] >= 1.2, o.get("pf_pct") and o["pf_pct"] >= 1.2, o.get("trades", 0) >= 100,
                 b[1] is not None and o.get("avg_pct", -9) > b[1]]
            ok = all(bool(x) for x in c)
            say(S, variant, "verdict", f"{'PASS' if ok else 'FAIL'} validation_PF={f2(v.get('pf_pct'))}(>=1.2:{bool(c[0])}) "
                f"OOS_PF={f2(o.get('pf_pct'))}(>=1.2:{bool(c[1])}) OOS_trades={o.get('trades', 0)}(>=100:{c[2]}) "
                f"OOS_avg_ret%={f2(o.get('avg_pct'), 3)} random_p95={f2(b[1], 3)}(beats:{c[3]})")
            if ok:
                passed.append("TOM_primary")
    return passed


# ------------------------------------------------------------------------------------------ strategy 3
class RipCtx:
    def __init__(self, m: Mkt, hourly: pd.DataFrame, rules: DipRules):
        self.m, self.sched, self.rules = m, SCHEDS[m.sched], rules
        self.d = indicators(daily_bars(hourly, self.sched), rules)
        self.hr = Hr(hourly)
        self.c = self.d.close.to_numpy(float)
        self.sma_t, self.sma_x = self.d.sma_trend.to_numpy(float), self.d.sma_exit.to_numpy(float)
        self.rsi, self.atr_pct = self.d.rsi.to_numpy(float), self.d.atr_pct.to_numpy(float)
        self.atr = self.d.atr.to_numpy(float)
        self.ok = np.isfinite(self.c) & np.isfinite(self.sma_t) & np.isfinite(self.sma_x) & np.isfinite(self.rsi) & \
            np.isfinite(self.atr) & (self.atr > 0)


def sim_short(x: RipCtx, i: int, stress: float = 1.0, time_stop: int = 10) -> dict | None:
    """Decision at daily close i -> short filled at the first hourly bar at/after the next fill time, closed at the first
    daily close below SMA5 (or after `time_stop` closes), disaster stop 2.5 x ATR(14) above the fill, stop-first."""
    hr, d, hs = x.hr, x.d, x.m.spread * stress / 2
    k0 = int(np.searchsorted(hr.ts, fill_time(d.index[i], x.sched).value, "left"))
    if k0 >= hr.n:
        return None
    px = hr.o[k0]
    pct = x.rules.stop_atr * x.atr_pct[i]
    stop = px * (1 + pct)
    j = i + 1
    while j < len(d) and not (j - i >= time_stop or x.c[j] < x.sma_x[j]):
        j += 1
    if j >= len(d):
        return None
    kx = int(np.searchsorted(hr.ts, fill_time(d.index[j], x.sched).value, "left"))
    if kx >= hr.n or kx <= k0:
        return None
    hit = np.flatnonzero(hr.hi[k0:kx] >= stop)
    if len(hit):
        k = k0 + int(hit[0])
        xp, why, kk = max(hr.o[k], stop), "stop", k
    else:
        xp, why, kk = hr.o[kx], ("time" if j - i >= time_stop else "exit"), kx
    entry, exit_ = px - hs, xp + hs
    fund = funding_frac(x.m, -1, [hr.ts[k0]], [hr.ts[kk]], stress)[0]
    ret = (entry - exit_) / entry - fund
    return {"ts": pd.Timestamp(int(hr.ts[k0]), unit="ns", tz="UTC"), "xts": pd.Timestamp(int(hr.ts[kk]), unit="ns", tz="UTC"),
            "r": ret / pct, "ret": ret, "why": why}


def rip_market(x: RipCtx, thr: float, stress: float = 1.0) -> pd.DataFrame:
    cand = np.flatnonzero(x.ok & (x.c < x.sma_t) & (x.rsi > thr))
    rows, free = [], -1
    for i in cand:
        if x.d.index[i].value < free:
            continue
        t = sim_short(x, int(i), stress)
        if t:
            rows.append(t)
            free = t["xts"].value
    return pd.DataFrame(rows, columns=COLS) if rows else empty()


def rip_allhold(x: RipCtx) -> pd.DataFrame:
    """Every day with close < SMA200 as an independent entry (random-entry baseline pool)."""
    rows = [t for t in (sim_short(x, int(i)) for i in np.flatnonzero(x.ok & (x.c < x.sma_t))) if t]
    return pd.DataFrame(rows, columns=COLS) if rows else empty()


def run_rip(hourly: dict[str, pd.DataFrame]) -> list[str]:
    S = "RIPSELL"
    say(S, "ALL", "criteria", "PASS (primary only) = validation PF >= 1.2 AND OOS PF >= 1.2 AND OOS trades >= 100 pooled "
        "AND OOS PF > random-entry p95 (fewer than 100 OOS trades fails by rule)")
    say(S, "ALL", "conventions", "short when close < SMA200 and RSI(2) > threshold; exit at first daily close < SMA5 or "
        "after 10 daily closes; disaster stop 2.5 x ATR(14) above the fill (the dip module's multiple, mirrored); decision "
        "schedules per market type as bot dip module (US index close 16:00 NY fill 10:00 NY; GER40 close 17:00 Berlin fill "
        "10:00; UK100 close 16:00 London fill 09:00; FX/GOLD close 17:00 NY fill 08:00 London); stop checked on hourly "
        "bars; R = stop distance; PF on R")
    mk = [m for m in ("US500", "USTECH", "GER40", "UK100", "GBPUSD", "EURUSD", "NZDUSD", "EURJPY", "GOLD") if m in hourly]
    say(S, "ALL", "markets", f"available: {','.join(mk)}")
    passed = []
    for variant, thr in (("primary", 90.0), ("secondary", 95.0)):
        rules = DipRules(long_only=False, rsi_short=thr)
        real, real_s, allh = {}, {}, {}
        for name in mk:
            x = RipCtx(UNIVERSE[name], hourly[name], rules)
            real[name], real_s[name] = rip_market(x, thr), rip_market(x, thr, STRESS)
            if variant == "primary":
                allh[name] = rip_allhold(x)
        res = report_tables(S, variant, "splits", f"rsi>{thr:g}", real)
        per_market_oos(S, variant, real)
        report_tables(S, variant, "cost_stress", f"rsi>{thr:g} costs x{STRESS}", real_s)
        if variant == "primary":
            base = baseline_lines(S, variant, real, allh, "random days with close<SMA200, same exits")
            v, o = res["validation"], res["out_of_sample"]
            b = base.get("out_of_sample", (None, None))
            c = [v.get("pf_r") and v["pf_r"] >= 1.2, o.get("pf_r") and o["pf_r"] >= 1.2, o.get("trades", 0) >= 100,
                 b[0] is not None and (o.get("pf_r") or 0) > b[0]]
            ok = all(bool(z) for z in c)
            say(S, variant, "verdict", f"{'PASS' if ok else 'FAIL'} validation_PF={f2(v.get('pf_r'))}(>=1.2:{bool(c[0])}) "
                f"OOS_PF={f2(o.get('pf_r'))}(>=1.2:{bool(c[1])}) OOS_trades={o.get('trades', 0)}(>=100:{c[2]}) "
                f"random_PF_p95={f2(b[0])}(beaten:{c[3]})")
            if ok:
                passed.append("RIPSELL_primary")
    return passed
