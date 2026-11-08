# AGENTS.md: IG trading bot

Codex reads this file before every task. It overrides any conflicting instruction in a task prompt **unless** the human explicitly writes "I approve changing guardrail N" in that prompt.

## Mission

Build and maintain a deterministic, rule-based trading bot for the IG REST API. It must implement `STRATEGY.md` exactly. **Capital preservation outranks profit.** When unsure, do less and ask.

## Non-negotiable guardrails

1. **DEMO by default.** LIVE requires all three of:
   - `IG_ENV=LIVE`
   - `safety.live_trading_confirmed: true`
   - the `--live` CLI flag

   Never set these yourself. Never edit `safety.dry_run`, `live_trading_confirmed` or `live_full_risk_approved`. Never set the `BOT_*` or `IG_ENV` variables in any hosting dashboard. Those switches are the human's.
2. **Every order has a server-side stop** (`stopDistance` or `stopLevel`) at the moment it opens. A code path that can send an order without one is a critical bug.
3. **Stops only move in the trade's favour.** Never widen or remove a stop. Never convert a stop into a mental or client-side stop.
4. **No martingale, averaging down, grid, pyramiding, or increasing size after losses.**
5. **`risk:`, `safety:` and `sleeves:` limits are ceilings.**
   - Never raise them, and never raise `HARD_LIMITS` in `bot/config.py` or `SLEEVE_HARD_LIMITS` in `bot/sleeves.py`.
   - Never add a code path that bypasses `RiskManager.size()` or `RiskManager.check_entry()`.
6. **Sizing comes only from `bot/risk.py`.** If IG's minimum stake would exceed the risk cap, skip the trade. Never round up.
7. **Circuit breakers must always work:** the KILL file, PAUSE file, daily and weekly loss limits, the max-drawdown halt and the loss-streak pause. A halt is cleared only by a human (`--reset-halt`).
8. **Secrets** come only from env vars or `.env` (gitignored). Never print, log or commit the API key, password, `CST` or `X-SECURITY-TOKEN`. Never log request headers.
9. **No LLM or AI calls in the trading path.** Decisions are deterministic code.
10. **No lookahead.**
    - Act only on closed bars.
    - An indicator value at bar t may use data ≤ t only.
    - `tests/test_indicators.py::test_no_lookahead` and `tests/test_backtest.py::test_no_edge_on_random_walk` must stay green.
11. **Respect IG limits:** about 30 non-trading and 100 trading requests per minute per account, and 10,000 historical price points per week. Cache bars. Never re-send a trading request after a network error without first checking positions or confirms. It may have executed.
12. **The bot only touches positions it opened** (tracked by dealId in `state/state.json`). Never close or modify manual positions.
13. **Fail closed.** On unexpected errors, stop opening trades. Existing positions keep their server-side stops.
14. **Never delete or rewrite** `logs/` or `state/` history.
15. **The learning loop may only reduce risk.** It can switch a segment off; it must never switch anything on, raise size, loosen limits or change strategy parameters by itself. Improvements it suggests go through the strategy-change protocol and human approval. Never replace it with self-tuning or ML that adapts live.

## Strategy-change protocol (anti-overfitting)

Changing markets, timeframe, indicator rules or any `strategy:` parameter requires all of the following:

- A written hypothesis for *why* it should help, before you test it.
- Tuning on the **in-sample** period only. Report out-of-sample (OOS) metrics untouched.
- At most 3 parameters changed at once, with small grids and round numbers.
- A before/after table:
  - trades, profit factor, max drawdown and average R, for in-sample (IS) and OOS
  - costs included (spread, slippage, funding)
- Reject the change if:
  - OOS profit factor falls more than 30% below IS (overfit), or
  - it needs fewer than 100 trades to look good.
- Human approval before merging.

Never optimise on the full dataset or on the most recent few months alone.

## Engineering standards

- Python 3.11+, pandas, numpy, requests, PyYAML, pytest. Ask before adding new dependencies.
- Run `pytest -q` after every change. All tests must pass. Add tests for any new order, risk or state logic.
- Log every decision as JSON lines in `logs/decisions.jsonl`: signals, rejections and why, orders, confirms, stop moves, errors.
- Keep backtest and live parity: the same `Strategy` and `RiskManager` classes, with no live-only trading logic.
- Use small functions and type hints. Config is the only global.

## Commands

```bash
pip install -r requirements.txt
pytest -q
python -m bot.backtest --synthetic            # random walk: MUST show no edge (sanity check)
python -m bot.backtest --data data/history    # real data backtest
python -m bot.feasibility                     # IG DEMO: which markets your equity can size
python -m bot.live --once                     # one DEMO cycle (dry_run per config)
python -m bot.live                            # DEMO loop
```

## When unsure

Stop and ask. Never guess on anything that sends orders, changes risk, or touches credentials.
