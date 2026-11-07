"""IGClient request/response handling against a fake HTTP layer (no network)."""
import pytest
import requests

from bot.ig_client import IGClient, IGError


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/gateway/deal")[1]
        self.calls.append(dict(method=method, path=path, headers=headers, json=json, params=params))
        handler = self.routes[(method, path.split("?")[0])]
        return handler(headers, json, params) if callable(handler) else handler


def client(routes):
    c = IGClient("DEMO", "KEY", "user", "pw")
    c.session = FakeSession(routes)
    c._non_trading.n = c._trading.n = 10_000
    return c


LOGIN = Resp(200, {"currentAccountId": "ABC"}, {"CST": "cst", "X-SECURITY-TOKEN": "xst"})


def test_demo_url_and_login_tokens():
    c = client({("POST", "/session"): LOGIN})
    assert c.base.startswith("https://demo-api.ig.com")
    c.login()
    assert c._tokens == {"CST": "cst", "X-SECURITY-TOKEN": "xst"} and c.account_id == "ABC"


def test_open_position_body_has_stop_and_confirms():
    def open_handler(h, body, p):
        assert h["Version"] == "2"
        assert body["stopDistance"] == 42.5 and body["forceOpen"] is True
        assert body["orderType"] == "MARKET" and body["expiry"] == "DFB" and body["guaranteedStop"] is False
        return Resp(200, {"dealReference": "REF1"})
    routes = {("POST", "/session"): LOGIN, ("POST", "/positions/otc"): open_handler,
              ("GET", "/confirms/REF1"): Resp(200, {"dealStatus": "ACCEPTED", "dealId": "D1", "level": 100.0,
                                                    "stopLevel": 57.5})}
    c = client(routes)
    c.login()
    res = c.open_position("CS.D.EURUSD.TODAY.IP", "BUY", 0.5, 42.5, "DFB", "GBP")
    assert res["status"] == "ACCEPTED" and res["deal_id"] == "D1"


def test_close_uses_delete_override():
    def close_handler(h, body, p):
        assert h["_method"] == "DELETE" and h["Version"] == "1"
        assert body["direction"] == "SELL" and body["dealId"] == "D1"
        return Resp(200, {"dealReference": "REF2"})
    c = client({("POST", "/positions/otc"): close_handler,
                ("GET", "/confirms/REF2"): Resp(200, {"dealStatus": "ACCEPTED", "level": 101})})
    assert c.close_position("D1", "BUY", 0.5)["status"] == "ACCEPTED"


def test_trading_request_never_retried_on_network_error():
    calls = {"n": 0}

    def boom(h, b, p):
        calls["n"] += 1
        raise requests.ConnectionError("down")
    c = client({("POST", "/positions/otc"): boom})
    with pytest.raises(IGError):
        c.open_position("X", "BUY", 1.0, 10, "DFB", "GBP")
    assert calls["n"] == 1


def test_market_info_and_bars_parsing():
    market = {"instrument": {"expiry": "DFB", "currencies": [{"code": "GBP", "isDefault": True}]},
              "dealingRules": {"minDealSize": {"value": 0.5}, "minNormalStopOrLimitDistance": {"unit": "POINTS",
                                                                                              "value": 2}},
              "snapshot": {"bid": 10850.1, "offer": 10850.7, "marketStatus": "TRADEABLE"}}
    px = lambda o: {"bid": o, "ask": o + 1, "lastTraded": None}  # noqa: E731
    prices = {"prices": [{"snapshotTimeUTC": "2026-11-02T08:00:00", "openPrice": px(10), "highPrice": px(12),
                          "lowPrice": px(9), "closePrice": px(11)}],
              "metadata": {"allowance": {"remainingAllowance": 9990}}}
    c = client({("GET", "/markets/E"): Resp(200, market), ("GET", "/prices/E"): Resp(200, prices)})
    info = c.market_info("E")
    assert info.min_size == 0.5 and info.expiry == "DFB" and info.tradeable and round(info.spread, 1) == 0.6
    df = c.bars("E", "HOUR_4", 3)
    assert df["close"].iloc[0] == 11.5 and str(df.index[0].tz) == "UTC"


def test_market_info_when_market_closed():
    market = {"instrument": {"expiry": "DFB", "currencies": [{"code": "GBP", "isDefault": True}]},
              "dealingRules": {"minDealSize": {"value": 0.5},
                               "minNormalStopOrLimitDistance": {"unit": "PERCENTAGE", "value": 0.5}},
              "snapshot": {"bid": None, "offer": None, "marketStatus": "EDITS_ONLY"}}
    c = client({("GET", "/markets/E"): Resp(200, market)})
    info = c.market_info("E")
    assert not info.tradeable and info.min_size == 0.5 and info.min_stop_distance == 0.0
