"""Free 1-minute index CFD history from Dukascopy's public datafeed, cached on disk.

URL: https://datafeed.dukascopy.com/datafeed/<INSTRUMENT>/<YYYY>/<MM-1>/<DD>/BID_candles_min_1.bi5
Each file is one UTC day, LZMA-compressed, 24-byte big-endian records:
seconds-from-midnight, four integer prices (open, close, low, high in Dukascopy's order), volume (float).
Field order and price scale are checked against the data rather than assumed.
"""
from __future__ import annotations

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
REC = struct.Struct(">5if")
SCALES = (1, 10, 100, 1000, 10000, 100000)


def _url(inst: str, d: date) -> str:
    return f"{BASE}/{inst}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"


def parse_bi5(raw: bytes, d: date) -> np.ndarray:
    """-> array of (epoch_seconds, a, b, c, d, volume) with prices still as raw ints."""
    if not raw:
        return np.empty((0, 6))
    data = lzma.decompress(raw)
    n = len(data) // REC.size
    arr = np.array([REC.unpack_from(data, i * REC.size) for i in range(n)], dtype=float)
    day0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
    arr[:, 0] += day0
    return arr


def _fetch(session: requests.Session, inst: str, d: date, tries: int = 7) -> tuple[np.ndarray | None, str]:
    """-> (records, last status). records is None when every try failed (caller retries later)."""
    status = ""
    for i in range(tries):
        try:
            r = session.get(_url(inst, d), timeout=30)
            status = str(r.status_code)
            if r.status_code == 404:
                return np.empty((0, 6)), status
            if r.status_code == 200:
                return parse_bi5(r.content, d), status
        except (requests.RequestException, lzma.LZMAError) as exc:
            status = type(exc).__name__
        time.sleep(min(60, 2 ** i))
    return None, status


def to_frame(arr: np.ndarray, price_range: tuple[float, float]) -> pd.DataFrame:
    """Raw records -> OHLCV DataFrame (UTC index). Detects low/high columns and price scale."""
    if not len(arr):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    p = arr[:, 1:5]
    # the low column is the one equal to the row minimum most often; same for high
    lo_col = int(np.argmax([np.mean(p[:, j] == p.min(axis=1)) for j in range(4)]))
    hi_col = int(np.argmax([np.mean(p[:, j] == p.max(axis=1)) for j in range(4)]))
    oc = [j for j in range(4) if j not in (lo_col, hi_col)]
    med = float(np.median(p[:, oc[1]]))
    scale = next((s for s in SCALES if price_range[0] <= med / s <= price_range[1]), None)
    if scale is None:
        raise ValueError(f"cannot find price scale: median raw {med}, wanted {price_range}")
    df = pd.DataFrame({"open": p[:, oc[0]] / scale, "high": p[:, hi_col] / scale,
                       "low": p[:, lo_col] / scale, "close": p[:, oc[1]] / scale,
                       "volume": arr[:, 5]},
                      index=pd.to_datetime(arr[:, 0], unit="s", utc=True))
    return df


def load_range(inst: str, start: date, end: date, cache_dir: Path, price_range: tuple[float, float],
               workers: int = 3, log=print) -> pd.DataFrame:
    """Download (or read from cache) every weekday between start and end, one CSV.gz per month."""
    cache_dir = Path(cache_dir) / inst
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(start, end, freq="M")
    frames = []
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research backtest)"
    for m in months:
        f = cache_dir / f"{m}.csv.gz"
        last_day = min(m.end_time.date(), end)
        complete_month = m.end_time.date() <= end
        if f.exists():
            frames.append(pd.read_csv(f, index_col=0, parse_dates=True))
            continue
        days = [d.date() for d in pd.date_range(max(m.start_time.date(), start), last_day, freq="D")
                if d.weekday() < 5]
        with ThreadPoolExecutor(workers) as ex:
            got = dict(zip(days, ex.map(lambda d: _fetch(session, inst, d), days)))
        failed = [d for d, (a, _) in got.items() if a is None]
        for d in failed:                                   # slow second pass
            time.sleep(5)
            got[d] = _fetch(session, inst, d)
        still = {str(d): got[d][1] for d in failed if got[d][0] is None}
        arrs = [a for a, _ in got.values() if a is not None and len(a)]
        df = to_frame(np.vstack(arrs), price_range) if arrs else to_frame(np.empty((0, 6)), price_range)
        if complete_month and not still:
            df.to_csv(f, compression="gzip")
        frames.append(df)
        log({"event": "DOWNLOADED", "instrument": inst, "month": str(m), "rows": int(len(df)),
             "missing_days": still or None})
        time.sleep(0.5)
    out = pd.concat(frames).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out[~out.index.duplicated(keep="last")]
