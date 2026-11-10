import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.config import load_config  # noqa: E402
from bot.data import synthetic_market  # noqa: E402
from bot.ig_client import MarketInfo  # noqa: E402


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config(ROOT / "config.yaml"))
    c["kronos"]["enabled"] = False          # tests that need Kronos switch it on with a fake worker
    c["history"]["enabled"] = False         # tests that need the history downloader switch it on
    c["scalp"]["enabled"] = False           # test_scalp_live switches it on
    c["dip"]["enabled"] = False             # no network in tests; test_dip_live switches it on with a fake source
    return c


class FakeBroker:
    """In-memory stand-in for IGClient. Records every order so tests can assert guardrails."""

    def __init__(self, bars: dict[str, pd.DataFrame], equity: float = 10_000, spread: float = 0.5,
                 min_size: float = 0.5):
        self._bars = bars            # epic -> full DataFrame
        self._equity = equity
        self.spread = spread
        self.min_size = min_size
        self.now = None
        self.orders: list[dict] = []
        self.closes: list[dict] = []
        self.stop_updates: list[tuple] = []
        self.ig_positions: dict[str, dict] = {}
        self._n = 0

    account_type = "SPREADBET"
    account_id = "SB1"

    def login(self):
        pass

    def accounts(self):
        return [{"account_id": "SB1", "type": self.account_type, "name": "Spread bet", "currency": "GBP",
                 "preferred": True, "balance": self._equity},
                {"account_id": "CFD1", "type": "CFD", "name": "CFD", "currency": "GBP",
                 "preferred": False, "balance": 0}]

    def equity(self):
        return self._equity

    def set_equity(self, v):
        self._equity = v

    def _visible(self, epic):
        df = self._bars[epic]
        return df[df.index <= self.now] if self.now is not None else df

    def bars(self, epic, resolution, n):
        return self._visible(epic).tail(n)

    allowance_remaining = 9000
    range_calls: list = []

    def bars_range(self, epic, resolution, start, end):
        df = self._visible(epic)
        out = df[(df.index >= start) & (df.index <= end)]
        self.range_calls = self.range_calls + [(epic, start, end)]
        self.allowance_remaining -= len(out)
        return out

    def market_info(self, epic):
        px = float(self._visible(epic)["close"].iloc[-1])
        return MarketInfo(epic=epic, tradeable=True, bid=px - self.spread / 2, offer=px + self.spread / 2,
                          min_size=self.min_size, min_stop_distance=1.0, expiry="DFB", currency="GBP",
                          market_id=epic)

    def client_sentiment(self, market_id):
        return {"long_pct": 65.0, "short_pct": 35.0}

    def open_positions(self):
        # simulate IG's server-side stops: a position whose stop was touched by the latest bar is gone
        for deal_id, p in list(self.ig_positions.items()):
            if p.get("stop_level") is None or p["epic"] not in self._bars:
                continue
            bar = self._visible(p["epic"]).iloc[-1]
            hit = bar["low"] <= p["stop_level"] if p["direction"] == "BUY" else bar["high"] >= p["stop_level"]
            lim = p.get("limit_level")
            if lim is not None:
                hit = hit or (bar["high"] >= lim if p["direction"] == "BUY" else bar["low"] <= lim)
            if hit:
                self.ig_positions.pop(deal_id)
        return list(self.ig_positions.values())

    def open_position(self, epic, direction, size, stop_distance, expiry, currency, limit_distance=None):
        assert stop_distance and stop_distance > 0, "order without stop reached the broker"
        self._n += 1
        deal_id = f"D{self._n}"
        px = float(self._visible(epic)["close"].iloc[-1])
        sign = 1 if direction == "BUY" else -1
        self.orders.append(dict(epic=epic, direction=direction, size=size, stop_distance=stop_distance,
                                limit_distance=limit_distance))
        limit = px + sign * limit_distance if limit_distance else None
        self.ig_positions[deal_id] = dict(deal_id=deal_id, epic=epic, direction=direction, size=size,
                                          level=px, stop_level=px - sign * stop_distance, limit_level=limit)
        return {"status": "ACCEPTED", "deal_id": deal_id, "level": px,
                "stop_level": px - sign * stop_distance, "limit_level": limit}

    def update_stop(self, deal_id, stop_level):
        p = self.ig_positions.get(deal_id, {})
        self.stop_updates.append((deal_id, p.get("stop_level"), stop_level, p.get("direction")))
        if p:
            p["stop_level"] = stop_level
        return {"status": "ACCEPTED"}

    def close_position(self, deal_id, direction, size):
        self.closes.append(dict(deal_id=deal_id, direction=direction, size=size))
        self.ig_positions.pop(deal_id, None)
        return {"status": "ACCEPTED", "level": None}


@pytest.fixture
def make_broker():
    def _make(cfg, n_bars=900, regimes=True, **kw):
        bars = {}
        for i, m in enumerate(cfg["markets"]):
            bars[m["epic"]] = synthetic_market(11 + i, n_bars, 10_000.0, 0.003, regimes=regimes)
        return FakeBroker(bars, **kw)
    return _make
