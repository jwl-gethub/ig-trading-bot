"""Thin IG REST API client (https://labs.ig.com). DEMO by default.

Endpoints used (header `Version` in brackets):
  POST /session [2]                  login -> CST + X-SECURITY-TOKEN headers
  PUT  /session [1]                  switch account (if IG_ACCOUNT_ID given)
  GET  /accounts [1]                 balance / P&L / available
  GET  /markets/{epic} [3]           dealing rules, expiry, currency, bid/offer, status
  GET  /prices/{epic} [3]            historical bars (resolution, max, pageSize=0)
  GET  /positions [2]                open positions
  POST /positions/otc [2]            open market position WITH stop
  PUT  /positions/otc/{dealId} [2]   amend stop
  POST /positions/otc [1] + `_method: DELETE`   close position
  GET  /confirms/{dealReference} [1] deal confirmation

Published limits (LIVE; demo can be lower): 30 non-trading req/min per account,
60 per app, 100 trading req/min per account, 10,000 historical price points per week.
Never log request headers — they contain the API key and session tokens.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger("ig")

BASE_URLS = {
    "DEMO": "https://demo-api.ig.com/gateway/deal",
    "LIVE": "https://api.ig.com/gateway/deal",
}


class IGError(RuntimeError):
    def __init__(self, status: int, code: str, path: str):
        super().__init__(f"IG {status} {code} on {path}")
        self.status, self.code, self.path = status, code, path


class SafetyError(RuntimeError):
    pass


@dataclass
class MarketInfo:
    epic: str
    tradeable: bool
    bid: float
    offer: float
    min_size: float
    min_stop_distance: float
    expiry: str
    currency: str

    @property
    def spread(self) -> float:
        return self.offer - self.bid

    @property
    def mid(self) -> float:
        return (self.offer + self.bid) / 2


class _Bucket:
    """Sliding-window limiter: at most `n` calls per `window` seconds."""

    def __init__(self, n: int, window: float = 60.0):
        self.n, self.window, self.calls = n, window, deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        if len(self.calls) >= self.n:
            time.sleep(self.window - (now - self.calls[0]) + 0.05)
        self.calls.append(time.monotonic())


class IGClient:
    def __init__(self, env: str, api_key: str, username: str, password: str,
                 account_id: str | None = None, timeout: float = 15.0):
        env = env.upper()
        if env not in BASE_URLS:
            raise SafetyError(f"unknown IG environment {env}")
        self.env = env
        self.base = BASE_URLS[env]
        self._api_key, self._user, self._pw = api_key, username, password
        self.account_id = account_id
        self.timeout = timeout
        self.session = requests.Session()
        self._tokens: dict[str, str] = {}
        self._non_trading = _Bucket(25)   # headroom under the 30/min account limit
        self._trading = _Bucket(60)

    # ------------------------------------------------------------ plumbing
    def _headers(self, version: int, extra: dict | None = None) -> dict:
        h = {
            "X-IG-API-KEY": self._api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": str(version),
            **self._tokens,
        }
        if extra:
            h.update(extra)
        return h

    def _request(self, method: str, path: str, version: int, *, json=None, params=None,
                 trading: bool = False, extra_headers: dict | None = None, _retry: bool = True):
        (self._trading if trading else self._non_trading).wait()
        url = f"{self.base}{path}"
        for attempt in range(3):
            try:
                resp = self.session.request(method, url, headers=self._headers(version, extra_headers),
                                            json=json, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == 2 or trading:
                    # never blindly resend a trading request: it may have executed
                    raise IGError(0, f"network:{type(exc).__name__}", path) from exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401 and _retry and path != "/session":
                self.login()
                return self._request(method, path, version, json=json, params=params, trading=trading,
                                     extra_headers=extra_headers, _retry=False)
            if resp.status_code >= 500 and not trading and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                try:
                    code = resp.json().get("errorCode", "unknown")
                except ValueError:
                    code = resp.text[:200]
                raise IGError(resp.status_code, code, path)
            return resp
        raise IGError(0, "retries_exhausted", path)

    # ------------------------------------------------------------ session
    def login(self) -> None:
        resp = self._request("POST", "/session", 2,
                             json={"identifier": self._user, "password": self._pw}, _retry=False)
        self._tokens = {"CST": resp.headers["CST"], "X-SECURITY-TOKEN": resp.headers["X-SECURITY-TOKEN"]}
        current = resp.json().get("currentAccountId")
        if self.account_id and current != self.account_id:
            resp = self._request("PUT", "/session", 1,
                                 json={"accountId": self.account_id, "defaultAccount": False})
            if "X-SECURITY-TOKEN" in resp.headers:
                self._tokens["X-SECURITY-TOKEN"] = resp.headers["X-SECURITY-TOKEN"]
        else:
            self.account_id = current
        log.info("IG login ok env=%s account=%s", self.env, self.account_id)

    # ------------------------------------------------------------ data
    def equity(self) -> float:
        accts = self._request("GET", "/accounts", 1).json()["accounts"]
        acct = next(a for a in accts if a["accountId"] == self.account_id)
        bal = acct["balance"]
        return float(bal["balance"]) + float(bal.get("profitLoss") or 0.0)

    def market_info(self, epic: str) -> MarketInfo:
        d = self._request("GET", f"/markets/{epic}", 3).json()
        inst, rules, snap = d["instrument"], d["dealingRules"], d["snapshot"]
        currency = next((c["code"] for c in inst.get("currencies", []) if c.get("isDefault")),
                        (inst.get("currencies") or [{"code": "GBP"}])[0]["code"])
        min_stop = rules.get("minNormalStopOrLimitDistance") or {}
        stop_val = float(min_stop.get("value") or 0.0)
        if min_stop.get("unit") == "PERCENTAGE":
            stop_val = stop_val / 100 * float(snap["offer"])
        return MarketInfo(
            epic=epic,
            tradeable=snap.get("marketStatus") == "TRADEABLE",
            bid=float(snap["bid"]), offer=float(snap["offer"]),
            min_size=float(rules["minDealSize"]["value"]),
            min_stop_distance=stop_val,
            expiry=inst.get("expiry") or "-",
            currency=currency,
        )

    def bars(self, epic: str, resolution: str, n: int) -> pd.DataFrame:
        """Last n bars (mid prices). The newest bar may still be forming — caller drops it."""
        d = self._request("GET", f"/prices/{epic}", 3,
                          params={"resolution": resolution, "max": n, "pageSize": 0}).json()
        rows = []
        for p in d.get("prices", []):
            def mid(k):
                bid, ask = p[k].get("bid"), p[k].get("ask")
                if bid is None or ask is None:
                    return p[k].get("lastTraded")
                return (float(bid) + float(ask)) / 2
            raw_ts = (p.get("snapshotTimeUTC") or p.get("snapshotTime")).replace("/", "-")
            ts = pd.Timestamp(raw_ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            rows.append({"timestamp": ts,
                         "open": mid("openPrice"), "high": mid("highPrice"),
                         "low": mid("lowPrice"), "close": mid("closePrice")})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        allowance = d.get("metadata", {}).get("allowance", {})
        if allowance:
            log.info("historical allowance remaining=%s", allowance.get("remainingAllowance"))
        return df.set_index("timestamp").sort_index().dropna()

    def open_positions(self) -> list[dict]:
        d = self._request("GET", "/positions", 2).json()
        out = []
        for item in d.get("positions", []):
            p, m = item["position"], item["market"]
            out.append({
                "deal_id": p["dealId"], "epic": m["epic"], "direction": p["direction"],
                "size": float(p["size"]), "level": float(p["level"]),
                "stop_level": float(p["stopLevel"]) if p.get("stopLevel") is not None else None,
                "bid": m.get("bid"), "offer": m.get("offer"),
            })
        return out

    # ------------------------------------------------------------ trading
    def confirm(self, deal_reference: str) -> dict:
        for i in range(5):
            try:
                return self._request("GET", f"/confirms/{deal_reference}", 1).json()
            except IGError as e:
                if e.status == 404 and i < 4:
                    time.sleep(0.5 * (i + 1))
                    continue
                raise
        raise IGError(0, "confirm_timeout", deal_reference)

    def open_position(self, epic: str, direction: str, size: float, stop_distance: float,
                      expiry: str, currency: str) -> dict:
        # GUARDRAIL: refuse any order without a positive server-side stop distance.
        if stop_distance is None or not stop_distance > 0:
            raise SafetyError("refusing to open a position without a stop")
        if direction not in ("BUY", "SELL") or not size > 0:
            raise SafetyError(f"bad order direction={direction} size={size}")
        body = {
            "epic": epic, "expiry": expiry, "direction": direction, "size": round(size, 2),
            "orderType": "MARKET", "timeInForce": "FILL_OR_KILL", "guaranteedStop": False,
            "stopDistance": round(stop_distance, 1), "forceOpen": True, "currencyCode": currency,
            "trailingStop": False,
        }
        ref = self._request("POST", "/positions/otc", 2, json=body, trading=True).json()["dealReference"]
        c = self.confirm(ref)
        return {"status": c.get("dealStatus"), "reason": c.get("reason"), "deal_id": c.get("dealId"),
                "level": c.get("level"), "stop_level": c.get("stopLevel"), "deal_reference": ref}

    def update_stop(self, deal_id: str, stop_level: float) -> dict:
        body = {"stopLevel": stop_level, "limitLevel": None, "trailingStop": False}
        ref = self._request("PUT", f"/positions/otc/{deal_id}", 2, json=body, trading=True).json()["dealReference"]
        c = self.confirm(ref)
        return {"status": c.get("dealStatus"), "reason": c.get("reason")}

    def close_position(self, deal_id: str, direction: str, size: float) -> dict:
        close_dir = "SELL" if direction == "BUY" else "BUY"
        body = {"dealId": deal_id, "direction": close_dir, "size": size, "orderType": "MARKET",
                "timeInForce": "FILL_OR_KILL"}
        ref = self._request("POST", "/positions/otc", 1, json=body, trading=True,
                            extra_headers={"_method": "DELETE"}).json()["dealReference"]
        c = self.confirm(ref)
        return {"status": c.get("dealStatus"), "reason": c.get("reason"), "level": c.get("level")}
