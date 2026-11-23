"""Strategy 4 (time-of-day holds on 1-minute data) and the gold-scalper side check. Research only.

4a: long US500 / USTECH from the last price before 16:00 New York to the price at 09:35 New York on the next trading
    day, Monday-Thursday nights only (DST aware via America/New_York).
4b: long GOLD from 23:00 UTC to 07:00 UTC next morning, sessions starting Sunday-Thursday.
"""
from __future__ import annotations

import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .cand_common import (SPLIT_NAMES, STRESS, UNIVERSE, Mkt, f2, funding_frac, max_dd, pf, say, split_mask)
from .cand_data import frame
from .dukascopy import _fetch

SEEDS = 20
TOL = 5 * 60 * 10 ** 9           # a price must exist within 5 minutes of the wanted time
MIN = 60 * 10 ** 9
COLS = ["ts", "xts", "bps", "gross", "ret"]
NY = "America/New_York"


def load_minutes(inst: str, scale: int, start: date, end: date, cache_dir: Path, workers: int = 3, log=None,
                 cols: tuple = ("open", "high", "low", "close")) -> pd.DataFrame:
    """Same monthly cache files as research.dukascopy.load_range, but with a fixed integer price scale."""
    cache_dir = Path(cache_dir) / inst
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research backtest)"
    frames = []
    for m in pd.period_range(start, end, freq="M"):
        f = cache_dir / f"{m}.csv.gz"
        if f.exists():
            frames.append(pd.read_csv(f, index_col=0, parse_dates=True)[list(cols)])
            continue
        last_day = min(m.end_time.date(), end)
        days = [d.date() for d in pd.date_range(max(m.start_time.date(), start), last_day, freq="D") if d.weekday() < 5]
        with ThreadPoolExecutor(workers) as ex:
            got = dict(zip(days, ex.map(lambda d: _fetch(session, inst, d), days)))
        failed = [d for d, (a, _) in got.items() if a is None]
        for d in failed:
            time.sleep(5)
            got[d] = _fetch(session, inst, d)
        still = {str(d): got[d][1] for d in failed if got[d][0] is None}
        arrs = [a for a, _ in got.values() if a is not None and len(a)]
        df = frame(np.vstack(arrs), scale) if arrs else frame(np.empty((0, 6)), scale)
        if m.end_time.date() <= end and not still:
            df.to_csv(f, compression="gzip")
        frames.append(df[list(cols)])
        if log:
            log(f"{inst} {m}: rows={len(df)} missing_days={len(still)}")
        time.sleep(0.3)
    out = pd.concat([x for x in frames if len(x)]).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out[~out.index.duplicated(keep="last")]


# ------------------------------------------------------------------------------------------ price lookups
class Minute:
    def __init__(self, m1: pd.DataFrame):
        self.ts = m1.index.as_unit("ns").asi8.copy()
        self.o = m1.open.to_numpy(float).copy()
        self.c = m1.close.to_numpy(float).copy()
        self.n = len(self.ts)

    def last_before(self, t: np.ndarray) -> np.ndarray:
        """index of the last bar starting before t and within TOL of it, else -1 (price = its close)."""
        i = np.searchsorted(self.ts, t, "left") - 1
        ok = (i >= 0) & (t - self.ts[np.maximum(i, 0)] <= TOL)
        return np.where(ok, i, -1)

    def first_at(self, t: np.ndarray) -> np.ndarray:
        """index of the first bar starting at/after t and within TOL, else -1 (price = its open)."""
        i = np.searchsorted(self.ts, t, "left")
        ok = (i < self.n) & (self.ts[np.minimum(i, self.n - 1)] - t <= TOL)
        return np.where(ok, i, -1)


def ny_instants(dates: pd.DatetimeIndex, hh: int, mm: int) -> np.ndarray:
    """UTC ns of hh:mm New York local time on each (naive) date; DST aware."""
    loc = (dates + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    return loc.tz_convert("UTC").as_unit("ns").asi8


def make_trades(m: Mkt, a: Minute, i_in: np.ndarray, i_out: np.ndarray, entry_last: bool, stress: float = 1.0) -> pd.DataFrame:
    ok = (i_in >= 0) & (i_out >= 0)
    i_in, i_out = i_in[ok], i_out[ok]
    if not len(i_in):
        return pd.DataFrame(columns=COLS)
    hs = m.spread * stress / 2
    pin = a.c[i_in] if entry_last else a.o[i_in]
    pout = a.o[i_out]
    t_in = a.ts[i_in] + (MIN if entry_last else 0)
    t_out = a.ts[i_out]
    fund = funding_frac(m, 1, t_in, t_out, stress)
    ret = (pout - hs) / (pin + hs) - 1 - fund
    return pd.DataFrame({"ts": pd.to_datetime(t_in, unit="ns", utc=True), "xts": pd.to_datetime(t_out, unit="ns", utc=True),
                         "bps": 1e4 * ret, "gross": 1e4 * (pout / pin - 1), "ret": ret})


def trades_4a(m: Mkt, a: Minute, stress: float = 1.0) -> pd.DataFrame:
    lo = pd.Timestamp(a.ts[0], unit="ns").normalize() - pd.Timedelta(days=1)
    hi = pd.Timestamp(a.ts[-1], unit="ns").normalize() + pd.Timedelta(days=1)
    days = pd.date_range(lo, hi, freq="D")
    wk = days[days.dayofweek < 5]
    t_in, t_out = ny_instants(wk, 16, 0), ny_instants(wk, 9, 35)
    ii, io = a.last_before(t_in), a.first_at(t_out)
    pos = {d: k for k, d in enumerate(wk)}
    i_in, i_out = [], []
    for k, d in enumerate(wk):
        if d.dayofweek > 3 or ii[k] < 0:                         # Monday-Thursday nights only
            continue
        kk = pos.get(d + pd.Timedelta(days=1))                    # exit on the NEXT weekday only (one night); a
        if kk is not None and io[kk] >= 0:                        # holiday without a 09:35 price skips the trade
            i_in.append(ii[k]); i_out.append(io[kk])
    return make_trades(m, a, np.array(i_in, int), np.array(i_out, int), True, stress)


def trades_4b(m: Mkt, a: Minute, stress: float = 1.0) -> pd.DataFrame:
    lo = pd.Timestamp(a.ts[0], unit="ns").normalize()
    days = pd.date_range(lo, pd.Timestamp(a.ts[-1], unit="ns").normalize(), freq="D")
    days = days[(days.dayofweek == 6) | (days.dayofweek <= 3)]   # Sunday ... Thursday
    t_in = days.as_unit("ns").asi8 + 23 * 3600 * 10 ** 9
    t_out = t_in + 8 * 3600 * 10 ** 9
    return make_trades(m, a, a.first_at(t_in), a.first_at(t_out), False, stress)


def random_windows(m: Mkt, a: Minute, n: int, lengths: np.ndarray, lo_ns: int, hi_ns: int, rng, stress: float = 1.0) -> pd.DataFrame:
    """n windows starting at random minute bars within [lo_ns, hi_ns), length drawn from `lengths` (ns)."""
    i0, i1 = np.searchsorted(a.ts, lo_ns, "left"), np.searchsorted(a.ts, hi_ns, "left")
    if i1 <= i0 or n == 0:
        return pd.DataFrame(columns=COLS)
    i_in = rng.integers(i0, i1, size=n)
    t_out = a.ts[i_in] + rng.choice(lengths, size=n)
    return make_trades(m, a, i_in, a.first_at(t_out), False, stress)


# ------------------------------------------------------------------------------------------ statistics
def bstats(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    t = t.sort_values("xts")
    b = t.bps.to_numpy(float)
    return {"trades": int(len(t)), "win": 100 * float((b > 0).mean()), "pf": pf(b), "avg": float(b.mean()),
            "gross": float(t.gross.mean()), "tot": float(b.sum() / 100), "dd": float(max_dd(b) / 100),
            "worst": float(b.min())}


def bfmt(s: dict) -> str:
    if not s.get("trades"):
        return "trades=0"
    return (f"trades={s['trades']} win%={f2(s['win'], 1)} PF={f2(s['pf'])} avg_net_bps={f2(s['avg'], 2)} "
            f"avg_gross_bps={f2(s['gross'], 2)} total_ret%={f2(s['tot'], 1)} maxDD%={f2(s['dd'], 1)} "
            f"worst_trade_bps={f2(s['worst'], 1)}")


def _split(t: pd.DataFrame, s: str) -> pd.DataFrame:
    return t if s == "pooled" or not len(t) else t[split_mask(pd.DatetimeIndex(t.ts), s)]


def bounds(s: str, a: Minute) -> tuple[int, int]:
    lo, hi = a.ts[0], a.ts[-1] + 1
    b = {"development": (lo, pd.Timestamp("2019-01-01", tz="UTC").value), "validation": (
        pd.Timestamp("2019-01-01", tz="UTC").value, pd.Timestamp("2023-01-01", tz="UTC").value),
        "out_of_sample": (pd.Timestamp("2023-01-01", tz="UTC").value, hi), "pooled": (lo, hi)}[s]
    return max(b[0], lo), min(b[1], hi)


def run_variant(S: str, variant: str, mk: dict[str, tuple[Mkt, Minute]], fn, desc: str) -> list[str]:
    real = {n: fn(m, a) for n, (m, a) in mk.items()}
    stressed = {n: fn(m, a, STRESS) for n, (m, a) in mk.items()}
    allt = pd.concat([t for t in real.values() if len(t)], ignore_index=True)
    allst = pd.concat([t for t in stressed.values() if len(t)], ignore_index=True)
    res = {}
    for s in SPLIT_NAMES + ["pooled"]:
        res[s] = bstats(_split(allt, s))
        say(S, variant, "splits", f"split={s} {bfmt(res[s])}")
    for s in SPLIT_NAMES + ["pooled"]:
        say(S, variant, "cost_stress", f"split={s} costs x{STRESS}: {bfmt(bstats(_split(allst, s)))}")
    for n, t in real.items():
        say(S, variant, "per_market_oos", f"market={n} split=out_of_sample {bfmt(bstats(_split(t, 'out_of_sample')))}")
        say(S, variant, "per_market_oos", f"market={n} first_data={pd.Timestamp(mk[n][1].ts[0], unit='ns').date()}")
    base = {}
    for s in SPLIT_NAMES + ["pooled"]:
        pfs, avgs = [], []
        for k in range(SEEDS):
            fr = []
            for n, (m, a) in mk.items():
                cnt = len(_split(real[n], s))
                if not cnt:
                    continue
                lens = (real[n].xts - real[n].ts).to_numpy("timedelta64[ns]").astype("int64") if variant.startswith("4a") \
                    else np.array([8 * 3600 * 10 ** 9])
                lo, hi = bounds(s, a)
                rng = np.random.default_rng([k, zlib.crc32((n + s + variant).encode())])
                fr.append(random_windows(m, a, cnt, lens, lo, hi, rng))
            fr = [x for x in fr if len(x)]
            if fr:
                st = bstats(pd.concat(fr, ignore_index=True))
                avgs.append(st["avg"])
                if st["pf"] is not None:
                    pfs.append(st["pf"])
        if avgs:
            base[s] = (float(np.percentile(pfs, 95)) if pfs else None, float(np.percentile(avgs, 95)))
            say(S, variant, "baseline", f"split={s} random windows (same count and length rule, same costs): PF mean="
                f"{f2(np.mean(pfs) if pfs else None)} p95={f2(base[s][0])} | avg_net_bps mean={f2(np.mean(avgs), 2)} "
                f"p95={f2(base[s][1], 2)} (seeds={SEEDS})")
    v, o = res["validation"], res["out_of_sample"]
    b = base.get("out_of_sample", (None, None))
    c = [(v.get("pf") or 0) >= 1.2, (o.get("pf") or 0) >= 1.2, o.get("trades", 0) >= 100,
         b[1] is not None and o.get("avg", -999) > b[1]]
    ok = all(c)
    say(S, variant, "verdict", f"{'PASS' if ok else 'FAIL'} validation_PF={f2(v.get('pf'))}(>=1.2:{c[0]}) "
        f"OOS_PF={f2(o.get('pf'))}(>=1.2:{c[1]}) OOS_trades={o.get('trades', 0)}(>=100:{c[2]}) "
        f"OOS_avg_bps={f2(o.get('avg'), 2)} random_p95={f2(b[1], 2)}(beaten:{c[3]})")
    return [f"TOD_{variant}"] if ok else []


# ------------------------------------------------------------------------------------------ side check: gold scalper
def scalp_stats(t: pd.DataFrame) -> str:
    if not len(t):
        return "trades=0"
    w, l = t.r[t.r > 0], t.r[t.r <= 0]
    p = float(w.sum() / -l.sum()) if l.sum() < 0 else None
    return (f"trades={len(t)} win%={100 * len(w) / len(t):.1f} PF={f2(p)} avgR={t.r.mean():.3f} totR={t.r.sum():.1f}")


def scalp_split(t: pd.DataFrame, s: str) -> pd.DataFrame:
    return t if s == "pooled" or not len(t) else t[split_mask(pd.DatetimeIndex(t.ts), s)]


def bars15(m1: pd.DataFrame) -> pd.DataFrame:
    return m1.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
