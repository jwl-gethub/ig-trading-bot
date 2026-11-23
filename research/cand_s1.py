"""Strategy 1: diversified time-series momentum (TSMOM), weekly decisions on daily bars (21:00 UTC close). Research only.

Positions are tracked in units per market; P&L is units x price change (price currency treated as GBP: no FX translation,
a stated limitation). Decisions are made at the last daily close of each ISO week and executed at the next daily bar's
open (that market's next bar), half spread paid on every change in units. Funding per night held (Friday x3).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .cand_common import (CAPITAL, SPLIT_NAMES, STRESS, UNIVERSE, annual_rate, f2, say, split_mask, stress_rate)

SEEDS = 20
BAND = 0.25
SIGMA_TARGET = 0.10
MIN_SIGMA = 0.02
CAL0 = np.datetime64("2009-01-01")


def night_cum(n_days: int) -> np.ndarray:
    """cum[k] = sum of night weights of calendar days (CAL0 + j) for j < k  (Mon-Thu 1, Fri 3, Sat/Sun 0)."""
    days = CAL0 + np.arange(n_days)
    wd = (days.astype("datetime64[D]").astype(int) + 3) % 7        # 1970-01-01 was a Thursday -> Monday = 0
    w = np.where(wd <= 3, 1.0, np.where(wd == 4, 3.0, 0.0))
    return np.r_[0.0, np.cumsum(w)]


class Panel:
    def __init__(self, daily: dict[str, pd.DataFrame]):
        self.names = list(daily)
        self.dates = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in daily.values()])))
        T, M = len(self.dates), len(self.names)
        self.O, self.H, self.L, self.C = (np.full((T, M), np.nan) for _ in range(4))
        self.nights = np.zeros((T, M))
        self.Rl, self.Rs = np.zeros((T, M)), np.zeros((T, M))
        self.hs = np.array([UNIVERSE[n].spread / 2 for n in self.names])
        self.feat = {}
        cum = night_cum(int((self.dates.values[-1].astype('datetime64[D]') - CAL0).astype(int)) + 30)
        years = self.dates.year.to_numpy()
        for j, n in enumerate(self.names):
            d = daily[n]
            ix = self.dates.get_indexer(d.index)
            self.O[ix, j], self.H[ix, j], self.L[ix, j], self.C[ix, j] = d.open, d.high, d.low, d.close
            dd = (d.index.to_numpy().astype("datetime64[D]") - CAL0).astype(int)
            nxt = np.r_[dd[1:], dd[-1] + 1]
            self.nights[ix, j] = cum[nxt] - cum[dd]
            self.Rl[:, j] = stress_rate(annual_rate(UNIVERSE[n], 1, years), 1.0)
            self.Rs[:, j] = annual_rate(UNIVERSE[n], -1, years)
        self.has = np.isfinite(self.C)
        self.Cf = pd.DataFrame(self.C).ffill().to_numpy()
        self.T, self.M = T, M
        iso = self.dates.isocalendar()
        self.week = (iso.year.to_numpy() * 100 + iso.week.to_numpy()).astype(int)

    def features(self, daily: dict[str, pd.DataFrame]) -> None:
        T, M = self.T, self.M
        self.sgn = np.full((T, M, 3), np.nan)
        self.sig = np.full((T, M), np.nan)
        self.dec = np.zeros((T, M), bool)
        for j, n in enumerate(self.names):
            d = daily[n]
            c = d.close
            ix = self.dates.get_indexer(d.index)
            for k, lb in enumerate((21, 63, 252)):
                self.sgn[ix, j, k] = np.sign(c / c.shift(lb) - 1).to_numpy()
            self.sig[ix, j] = (c.pct_change().rolling(60).std() * np.sqrt(252)).to_numpy()
            wk = self.week[ix]
            last = np.r_[wk[1:] != wk[:-1], False]          # last bar of its ISO week (and not the final bar)
            self.dec[ix, j] = last
        # data-sanity guard: a market with annualised 60-day vol under 2% (stale/flat feed; lowest genuine value in the universe
        # is ~2.3%) is treated as having no usable history that day, otherwise 10%/sigma explodes
        self.valid = np.isfinite(self.sgn).all(axis=2) & np.isfinite(self.sig) & (self.sig >= MIN_SIGMA)
        self.dec &= self.valid
        wmax = self.week.max() + 1
        cnt = np.bincount(self.week, weights=self.dec.sum(axis=1), minlength=wmax)
        self.nactive = cnt[self.week]


def target_weights(p: Panel, s: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        tw = s * (SIGMA_TARGET / np.sqrt(np.maximum(p.nactive, 1)))[:, None] / p.sig
    return np.where(p.dec, tw, np.nan)


def units_from_events(T: int, ev: list[tuple[int, float]]) -> np.ndarray:
    if not ev:
        return np.zeros(T)
    e = np.array([x[0] for x in ev])
    u = np.array([x[1] for x in ev])
    pos = np.searchsorted(e, np.arange(T), "right") - 1
    return np.where(pos >= 0, u[np.maximum(pos, 0)], 0.0)


def band_trade(tgt: float, curw: float) -> bool:
    """No-trade band: trade only if |target - current| > 25% of |target|, the sign changes, or the target is zero
    while a position exists."""
    if tgt == 0.0:
        return curw != 0.0
    if curw != 0.0 and np.sign(curw) != np.sign(tgt):
        return True
    return abs(tgt - curw) > BAND * abs(tgt)


def build_units(p: Panel, tw: np.ndarray, cap: float = CAPITAL) -> np.ndarray:
    U = np.zeros((p.T, p.M))
    for j in range(p.M):
        has_idx = np.flatnonzero(p.has[:, j])
        cur, ev = 0.0, []
        for td in np.flatnonzero(p.dec[:, j]):
            tgt = float(tw[td, j])
            if not np.isfinite(tgt) or not band_trade(tgt, cur * p.Cf[td, j] / cap):
                continue
            pos = np.searchsorted(has_idx, td, "right")
            if pos >= len(has_idx):
                continue
            e = int(has_idx[pos])
            cur = tgt * cap / p.O[e, j]
            ev.append((e, cur))
        U[:, j] = units_from_events(p.T, ev)
    return U


def pnl_cells(p: Panel, U: np.ndarray, stress: float = 1.0, cap: float = CAPITAL) -> dict[str, np.ndarray]:
    Cp = np.vstack([np.full((1, p.M), np.nan), p.Cf[:-1]])
    Uo = np.vstack([np.zeros((1, p.M)), U[:-1]])
    oo = np.where(p.has, p.O, Cp)
    price = np.nan_to_num(Uo * (oo - Cp) + U * (p.Cf - oo))
    dU = np.abs(U - Uo)
    spread = dU * p.hs * stress
    rate = np.where(U > 0, np.where(p.Rl > 0, p.Rl * stress, p.Rl), np.where(p.Rs > 0, p.Rs * stress, p.Rs))
    fund = np.nan_to_num(np.abs(U) * p.Cf * rate * p.nights / 365.0)
    return {"price": price / cap, "spread": spread / cap, "fund": fund / cap, "net": (price - spread - fund) / cap,
            "traded": np.nan_to_num(dU * oo) / cap, "changes": (dU > 0), "gross": np.nan_to_num(np.abs(U) * p.Cf) / cap}


def perf(x: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    n = len(x)
    if n < 60 or np.std(x) == 0:
        return {"days": n}
    mu, sd = x.mean(), x.std(ddof=1)
    eq = np.cumsum(x)
    dd = float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max())
    mon = pd.Series(x, index=dates).groupby([dates.year, dates.month]).sum()
    return {"days": n, "ann": 100 * mu * 252, "vol": 100 * sd * np.sqrt(252), "sharpe": mu / sd * np.sqrt(252),
            "dd": 100 * dd, "worst_month": 100 * float(mon.min()), "pos_months": 100 * float((mon > 0).mean())}


def split_slices(p: Panel, start: int) -> dict[str, np.ndarray]:
    out = {s: split_mask(p.dates, s) for s in SPLIT_NAMES}
    out["pooled"] = np.ones(p.T, bool)
    live = np.arange(p.T) >= start
    return {k: v & live for k, v in out.items()}


def report_run(strat: str, variant: str, section: str, p: Panel, U: np.ndarray, stress: float = 1.0,
               quiet: bool = False) -> tuple[dict, dict]:
    cells = pnl_cells(p, U, stress)
    port = cells["net"].sum(axis=1)
    nz = np.flatnonzero(np.abs(U).sum(axis=1) > 0)
    start = int(nz[0]) if len(nz) else p.T
    res = {}
    for s, mask in split_slices(p, start).items():
        yrs = max(mask.sum() / 252, 1e-9)
        m = perf(port[mask], p.dates[mask])
        if m.get("ann") is not None:
            m.update({"turn": cells["traded"][mask].sum() / yrs, "spread": 100 * cells["spread"][mask].sum() / yrs,
                      "fund": 100 * cells["fund"][mask].sum() / yrs, "changes": int(cells["changes"][mask].sum()),
                      "gross": float(cells["gross"][mask].sum(axis=1).mean())})
        res[s] = m
        if not quiet:
            if m.get("ann") is None:
                say(strat, variant, section, f"split={s} days={m['days']} (insufficient data)")
            else:
                say(strat, variant, section,
                    f"split={s} days={m['days']} ann_ret%={f2(m['ann'])} ann_vol%={f2(m['vol'])} sharpe={f2(m['sharpe'])} "
                    f"maxDD%={f2(m['dd'])} worst_month%={f2(m['worst_month'])} pos_months%={f2(m['pos_months'], 1)} "
                    f"turnover_x_capital/yr={f2(m['turn'], 1)} cost_spread%/yr={f2(m['spread'])} cost_funding%/yr="
                    f"{f2(m['fund'])} position_changes={m['changes']} avg_gross_notional_x={f2(m['gross'])}")
    return res, cells


def per_market_oos(strat: str, variant: str, p: Panel, cells: dict) -> None:
    mask = split_mask(p.dates, "out_of_sample")
    yrs = max(mask.sum() / 252, 1e-9)
    for j, n in enumerate(p.names):
        c = cells["net"][mask, j]
        if np.abs(cells["gross"][mask, j]).sum() == 0:
            say(strat, variant, "per_market_oos", f"market={n} split=out_of_sample no positions")
            continue
        say(strat, variant, "per_market_oos",
            f"market={n} split=out_of_sample net_contribution_ann%={f2(100 * c.sum() / yrs)} "
            f"gross_ann%={f2(100 * cells['price'][mask, j].sum() / yrs)} cost_spread%/yr="
            f"{f2(100 * cells['spread'][mask, j].sum() / yrs)} cost_funding%/yr={f2(100 * cells['fund'][mask, j].sum() / yrs)}")


# ------------------------------------------------------------------------------------------ secondary: Donchian
def donchian_units(p: Panel, daily: dict[str, pd.DataFrame], cap: float = CAPITAL) -> np.ndarray:
    U = np.zeros((p.T, p.M))
    for j, n in enumerate(p.names):
        d = daily[n]
        c = d.close.to_numpy(float)
        hi55, lo55 = d.high.shift(1).rolling(55).max().to_numpy(), d.low.shift(1).rolling(55).min().to_numpy()
        hi20, lo20 = d.high.shift(1).rolling(20).max().to_numpy(), d.low.shift(1).rolling(20).min().to_numpy()
        pc = d.close.shift()
        tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
        atr = tr.rolling(20).mean().to_numpy()
        ix = p.dates.get_indexer(d.index)
        pos, ev = 0, []
        for t in range(55, len(d) - 1):
            if not (np.isfinite(hi55[t]) and np.isfinite(atr[t]) and atr[t] > 0):
                continue
            want = pos
            if pos == 0:
                want = 1 if c[t] > hi55[t] else (-1 if c[t] < lo55[t] else 0)
            elif pos > 0:
                if c[t] < lo55[t]:
                    want = -1
                elif c[t] < lo20[t]:
                    want = 0
            else:
                if c[t] > hi55[t]:
                    want = 1
                elif c[t] > hi20[t]:
                    want = 0
            if want != pos:
                ev.append((int(ix[t + 1]), want * 0.005 * cap / (2 * atr[t]) if want else 0.0))
                pos = want
        U[:, j] = units_from_events(p.T, ev)
    return U


# ------------------------------------------------------------------------------------------ runner
def run_tsmom(daily: dict[str, pd.DataFrame]) -> list[str]:
    S = "TSMOM"
    crit = ("PASS (primary only) = validation Sharpe >= 0.5 AND OOS Sharpe >= 0.5 after costs AND OOS annualised "
            "return >= 4% AND OOS Sharpe > random-sign p95 (OOS)")
    say(S, "ALL", "criteria", crit)
    say(S, "ALL", "conventions", f"capital fixed GBP {CAPITAL:.0f}; s = mean(sign r21, r63, r252); target weight = s x "
        f"(10%/sqrt(N_active))/sigma60 (annualised); decision at the last daily close of each ISO week, execution at the "
        f"market's next daily open; no-trade band {BAND:.0%}; no leverage cap; Sharpe on daily returns of the whole "
        "portfolio (rf ignored), splits by date; price P&L not FX-translated")
    p = Panel(daily)
    p.features(daily)
    say(S, "ALL", "markets", "universe used: " + ",".join(f"{n}(from {daily[n].index[0].date()})" for n in p.names))
    det = np.nanmean(p.sgn, axis=2)
    tw = target_weights(p, det)
    U = build_units(p, tw)
    diagnose(S, p, tw, U)
    res, cells = report_run(S, "primary", "splits", p, U)
    per_market_oos(S, "primary", p, cells)
    report_run(S, "primary", "cost_stress", p, U, STRESS)
    # baselines
    rs = {s: [] for s in res}
    for k in range(SEEDS):
        rng = np.random.default_rng([k, zlib.crc32(b"tsmom")])
        sr = rng.choice([-1.0, 1.0], size=(p.T, p.M, 3)).mean(axis=2)
        r, _ = report_run(S, "random", "x", p, build_units(p, target_weights(p, sr)), quiet=True)
        for s in res:
            if r[s].get("sharpe") is not None:
                rs[s].append(r[s]["sharpe"])
    for s in res:
        if rs[s]:
            say(S, "primary", "baseline", f"split={s} random-sign Sharpe mean={f2(np.mean(rs[s]))} "
                f"p95={f2(np.percentile(rs[s], 95))} max={f2(np.max(rs[s]))} (seeds={len(rs[s])})")
    lo_res, lo_cells = report_run(S, "long_only_equal_risk", "baseline", p, build_units(p, target_weights(p, np.ones((p.T, p.M)))))
    # secondary
    report_run(S, "secondary_donchian", "splits", p, donchian_units(p, daily))
    # feasibility
    feas_lines(S, p, tw)
    v, o = res["validation"], res["out_of_sample"]
    p95 = np.percentile(rs["out_of_sample"], 95) if rs["out_of_sample"] else None
    c = [(v.get("sharpe") or -9) >= 0.5, (o.get("sharpe") or -9) >= 0.5, (o.get("ann") or -99) >= 4.0,
         p95 is not None and (o.get("sharpe") or -9) > p95]
    ok = all(c)
    say(S, "primary", "verdict", f"{'PASS' if ok else 'FAIL'} validation_Sharpe={f2(v.get('sharpe'))}(>=0.5:{c[0]}) "
        f"OOS_Sharpe={f2(o.get('sharpe'))}(>=0.5:{c[1]}) OOS_ann_ret%={f2(o.get('ann'))}(>=4:{c[2]}) "
        f"random_sign_p95_OOS={f2(p95)}(beaten:{c[3]})")
    return ["TSMOM_primary"] if ok else []


def diagnose(S: str, p: Panel, tw: np.ndarray, U: np.ndarray) -> None:
    """Sanity lines: per-market sigma / weight extremes and the largest gross exposures (to catch bad data)."""
    wgt = np.abs(U) * p.Cf / CAPITAL
    for j, n in enumerate(p.names):
        t = np.abs(tw[:, j])
        if not np.isfinite(t).any():
            continue
        k = int(np.nanargmax(t))
        gap = np.abs(p.O[:, j] / np.r_[np.nan, p.Cf[:-1, j]] - 1)
        say(S, "diag", "market", f"market={n} sigma_min={np.nanmin(p.sig[:, j]):.4f} sigma_med={np.nanmedian(p.sig[:, j]):.4f} "
            f"max_abs_target={np.nanmax(t):.2f}@{p.dates[k].date()} max_gross_weight={np.nanmax(wgt[:, j]):.2f} "
            f"open_gap_gt5pct_days={int(np.nansum(gap > 0.05))} max_open_gap={np.nanmax(gap):.3f}")
    flat = np.argsort(-wgt.max(axis=0))[:3]
    for j in flat:
        i = int(np.argmax(wgt[:, j]))
        say(S, "diag", "top_exposure", f"market={p.names[j]} date={p.dates[i].date()} weight={wgt[i, j]:.2f} open={p.O[i, j]:.5g} "
            f"close={p.C[i, j]:.5g} prev_close={p.Cf[i - 1, j]:.5g} units={U[i, j]:.4g}")


def feas_lines(S: str, p: Panel, tw: np.ndarray) -> None:
    for j, n in enumerate(p.names):
        m = UNIVERSE[n]
        sel = np.isfinite(tw[:, j]) & (tw[:, j] != 0)
        if not sel.any():
            continue
        for cap in (10_000, 20_000):
            st = np.abs(tw[sel, j]) * cap / p.Cf[sel, j] / m.pscale
            below = "unknown" if m.min_size is None else f"{100 * float((st < m.min_size).mean()):.1f}"
            say(S, "primary", "min_stake", f"market={n} capital={cap} targets={int(sel.sum())} "
                f"median_stake_gbp_per_pt={f2(np.median(st), 4)} p10_stake_gbp_per_pt={f2(np.percentile(st, 10), 4)} "
                f"ig_min_stake={m.min_size if m.min_size is not None else 'unknown'} pct_targets_below_min={below}")
