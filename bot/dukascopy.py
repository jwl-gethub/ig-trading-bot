"""Free hourly price history from Dukascopy's public datafeed (signals for the daily dip strategy).

- Complete months: one file per month, .../<INST>/<YYYY>/<MM-1>/BID_candles_hour_1.bi5
- The current month: one 1-minute file per day, .../<INST>/<YYYY>/<MM-1>/<DD>/BID_candles_min_1.bi5,
  aggregated to hours. Today is never used (incomplete).
Records are 24 bytes big-endian: seconds from the file's start, four integer prices, volume.
Column order (low/high) and price scale are detected from the data, not assumed.
Files are cached on disk; a failed download returns nothing for that file (the caller decides).
Uses no IG allowance. Never sends anything but GET requests for public files.
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">5if")
SCALES = (1, 10, 100, 1000, 10000, 100000)


def get(session: requests.Session, url: str, tries: int = 6) -> bytes | None:
    """Content; b'' for 404 (no data); None if every try failed."""
    for i in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return b""
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(min(30, 2 ** i))
    return None


def parse(raw: bytes, base_epoch: float) -> np.ndarray:
    if not raw:
        return np.empty((0, 6))
    data = lzma.decompress(raw)
    n = len(data) // REC.size
    arr = np.array([REC.unpack_from(data, i * REC.size) for i in range(n)], dtype=float).reshape(-1, 6)
    arr[:, 0] += base_epoch
    return arr


def to_frame(arr: np.ndarray, price_range: tuple[float, float]) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    if not len(arr):
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    p = arr[:, 1:5]
    lo = int(np.argmax([np.mean(p[:, j] == p.min(axis=1)) for j in range(4)]))
    hi = int(np.argmax([np.mean(p[:, j] == p.max(axis=1)) for j in range(4)]))
    oc = [j for j in range(4) if j not in (lo, hi)]
    med = float(np.median(p[:, oc[1]]))
    scale = next((s for s in SCALES if price_range[0] <= med / s <= price_range[1]), None)
    if scale is None:
        raise ValueError(f"dukascopy: no price scale fits median {med} in {price_range}")
    return pd.DataFrame({"open": p[:, oc[0]] / scale, "high": p[:, hi] / scale, "low": p[:, lo] / scale,
                         "close": p[:, oc[1]] / scale, "volume": arr[:, 5]},
                        index=pd.to_datetime(arr[:, 0], unit="s", utc=True))


def _epoch(d: date) -> float:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def _read(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


class HourlySource:
    def __init__(self, cache_dir: str | Path, pause: float = 0.2, log=None):
        self.cache = Path(cache_dir)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (personal research)"
        self.pause = pause
        self.log = log or (lambda *a, **k: None)

    def _month(self, inst: str, m: pd.Period, pr) -> pd.DataFrame | None:
        f = self.cache / inst / "hour" / f"{m}.csv.gz"
        if f.exists():
            return _read(f)
        raw = get(self.session, f"{BASE}/{inst}/{m.year:04d}/{m.month - 1:02d}/BID_candles_hour_1.bi5")
        time.sleep(self.pause)
        if raw is None:
            return None
        df = to_frame(parse(raw, _epoch(m.start_time.date())), pr)
        if len(df):                                 # never cache "no data" (may be a publishing delay)
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, compression="gzip")
        return df

    def _day(self, inst: str, d: date, pr) -> pd.DataFrame | None:
        f = self.cache / inst / "min" / f"{d}.csv.gz"
        if f.exists():
            return _read(f)
        raw = get(self.session, f"{BASE}/{inst}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5")
        time.sleep(self.pause)
        if raw is None:
            return None
        m = to_frame(parse(raw, _epoch(d)), pr)
        h = m.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                  "volume": "sum"}).dropna(subset=["close"]) if len(m) else m
        if len(h):
            f.parent.mkdir(parents=True, exist_ok=True)
            h.to_csv(f, compression="gzip")
        return h

    def hourly(self, inst: str, start: date, today: date, price_range: tuple[float, float]) -> tuple[pd.DataFrame, list]:
        """Hourly bars from `start` up to (not including) `today`. Returns (bars, failed_files)."""
        frames, failed = [], []
        this_month = pd.Period(today, freq="M")
        for m in pd.period_range(start, today, freq="M"):
            if m < this_month:
                df = self._month(inst, m, price_range)
                if df is None:
                    failed.append(str(m))
                    continue
                frames.append(df)
            else:
                for d in pd.date_range(m.start_time, pd.Timestamp(today) - pd.Timedelta(days=1), freq="D"):
                    if d.dayofweek == 5:
                        continue
                    df = self._day(inst, d.date(), price_range)
                    if df is None:
                        failed.append(str(d.date()))
                        continue
                    frames.append(df)
        if not frames:
            return to_frame(np.empty((0, 6)), price_range), failed
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")], failed
