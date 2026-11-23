"""Data for the candidate strategies: hourly history (daily strategies), 1-minute arrays (time-of-day holds).

Hourly: Dukascopy monthly hour files via the same cache layout as bot.dukascopy.HourlySource (/data/hourly), but the
integer price scale is detected ONCE per instrument from a recent month and then applied to every month (a per-month
range check breaks over a 15-year history).  Daily bars close at 21:00 UTC for every market.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bot.dukascopy import BASE, REC, SCALES, HourlySource, _epoch, get, parse

from .cand_common import Mkt

MIN_BARS_PER_DAY = 4


def detect_scale(arr: np.ndarray, probe: tuple[float, float]) -> int | None:
    """Integer divisor so that the median close lands in the probe range (same column logic as to_frame)."""
    if not len(arr):
        return None
    p = arr[:, 1:5]
    lo = int(np.argmax([np.mean(p[:, j] == p.min(axis=1)) for j in range(4)]))
    hi = int(np.argmax([np.mean(p[:, j] == p.max(axis=1)) for j in range(4)]))
    oc = [j for j in range(4) if j not in (lo, hi)]
    med = float(np.median(p[:, oc[1]]))
    return next((s for s in SCALES if probe[0] <= med / s <= probe[1]), None)


def frame(arr: np.ndarray, scale: int) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    if not len(arr):
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    p = arr[:, 1:5]
    lo = int(np.argmax([np.mean(p[:, j] == p.min(axis=1)) for j in range(4)]))
    hi = int(np.argmax([np.mean(p[:, j] == p.max(axis=1)) for j in range(4)]))
    oc = [j for j in range(4) if j not in (lo, hi)]
    return pd.DataFrame({"open": p[:, oc[0]] / scale, "high": p[:, hi] / scale, "low": p[:, lo] / scale,
                         "close": p[:, oc[1]] / scale, "volume": arr[:, 5]},
                        index=pd.to_datetime(arr[:, 0], unit="s", utc=True))


class Hourly(HourlySource):
    """HourlySource with a fixed per-instrument scale. Same cache files as the parent."""

    def probe_scale(self, m: Mkt, today: date, log) -> int:
        this = pd.Period(today, freq="M")
        for back in range(1, 7):
            per = this - back
            raw = get(self.session, f"{BASE}/{m.inst}/{per.year:04d}/{per.month - 1:02d}/BID_candles_hour_1.bi5")
            time.sleep(self.pause)
            if raw:
                s = detect_scale(parse(raw, _epoch(per.start_time.date())), m.probe)
                if s:
                    return s
                log(f"{m.name}: probe range {m.probe} fits no scale in {per}; using default {m.scale}")
                return m.scale
        raise RuntimeError("no data in the last 6 months")

    def month_scaled(self, m: Mkt, per: pd.Period, scale: int) -> pd.DataFrame | None:
        f = self.cache / m.inst / "hour" / f"{per}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        raw = get(self.session, f"{BASE}/{m.inst}/{per.year:04d}/{per.month - 1:02d}/BID_candles_hour_1.bi5")
        time.sleep(self.pause)
        if raw is None:
            return None
        df = frame(parse(raw, _epoch(per.start_time.date())), scale)
        if len(df):
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, compression="gzip")
        return df

    def day_scaled(self, m: Mkt, d: date, scale: int) -> pd.DataFrame | None:
        f = self.cache / m.inst / "min" / f"{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        raw = get(self.session, f"{BASE}/{m.inst}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5")
        time.sleep(self.pause)
        if raw is None:
            return None
        mm = frame(parse(raw, _epoch(d)), scale)
        h = mm.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                   "volume": "sum"}).dropna(subset=["close"]) if len(mm) else mm
        if len(h):
            f.parent.mkdir(parents=True, exist_ok=True)
            h.to_csv(f, compression="gzip")
        return h

    def load(self, m: Mkt, start: date, today: date, log) -> tuple[pd.DataFrame, list, int]:
        scale = self.probe_scale(m, today, log)
        frames, failed = [], []
        this = pd.Period(today, freq="M")
        for per in pd.period_range(start, today, freq="M"):
            if per < this:
                df = self.month_scaled(m, per, scale)
                if df is None:
                    failed.append(str(per))
                elif len(df):
                    frames.append(df)
            else:
                for d in pd.date_range(per.start_time, pd.Timestamp(today) - pd.Timedelta(days=1), freq="D"):
                    if d.dayofweek == 5:
                        continue
                    df = self.day_scaled(m, d.date(), scale)
                    if df is None:
                        failed.append(str(d.date()))
                    elif len(df):
                        frames.append(df)
        if not frames:
            raise RuntimeError("no hourly data")
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        bad = ~((out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out.high >= out.low))
        return out[~bad], failed, scale


def daily_21utc(h: pd.DataFrame, min_bars: int = MIN_BARS_PER_DAY) -> pd.DataFrame:
    """Hourly bars (index = bar start, UTC) -> daily bars closing at 21:00 UTC, labelled by the closing date.
    Bars starting at or after 21:00 UTC belong to the NEXT day; Friday-evening and weekend bars go to Monday.
    close = close of the last bar starting before 21:00 UTC. Days with fewer than `min_bars` hourly bars are dropped.
    Extra columns: n (hourly bars), open_ts / close_ts (UTC, the first bar start and the last bar end)."""
    if h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "n", "open_ts", "close_ts"])
    lab = (h.index + pd.Timedelta(hours=3)).tz_localize(None).normalize()
    wd = lab.dayofweek
    lab = lab + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    g = h.groupby(lab)
    out = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                        "n": g.close.count()})
    starts = pd.Series(h.index, index=h.index)
    out["open_ts"] = starts.groupby(lab).first()
    out["close_ts"] = starts.groupby(lab).last() + pd.Timedelta(hours=1)
    out = out[out.n >= min_bars]
    return out[out.index.dayofweek < 5]


def data_quality(name: str, d: pd.DataFrame) -> str:
    r = d.close.pct_change()
    flat = int(((d.high - d.low) / d.close < 1e-5).sum())
    big = r.abs().idxmax() if r.notna().any() else None
    jumps = r.index[r.abs() > 0.2]
    jtxt = ",".join(f"{x.date()}:{100 * r[x]:.0f}%" for x in jumps[:6]) or "none"
    return (f"{name} jumps_over_20pct={jtxt} days={len(d)} first={d.index[0].date()} last={d.index[-1].date()} "
            f"min_px={d.low.min():.5g} max_px={d.high.max():.5g} near_zero_range_days={flat} "
            f"max_abs_daily_move={100 * r.abs().max():.1f}%@{big.date() if big is not None else 'n/a'}")


def minute_arrays(m1: pd.DataFrame) -> dict:
    return {"ts": m1.index.as_unit("ns").asi8.copy(), "open": m1.open.to_numpy(float).copy(),
            "close": m1.close.to_numpy(float).copy()}
