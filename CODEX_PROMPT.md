# Codex kickoff prompt

Paste everything below the line into Codex as the first task. Run it from the repo root, which already contains `AGENTS.md`, `STRATEGY.md`, `config.yaml` and the `bot/` reference implementation.

---

You are the engineer on a rule-based trading bot for the IG REST API (UK spread betting account, GBP per point). The repo already contains a working reference implementation.

Your job is to connect it to IG DEMO, validate it on real data, harden it for unattended running, and prepare it for a cautious live launch. **You do not decide whether it goes live. The human does.**

Read `AGENTS.md` (guardrails: these override this prompt), `STRATEGY.md` (trading rules) and `config.yaml` before doing anything. Work through the phases in order. At the end of each phase, stop and give me a short report in the format at the bottom. Don't start the next phase until I reply "continue".

## Phase 0: Orient (no network)

1. `pip install -r requirements.txt`, then `pytest -q`. Everything must pass.
2. Run `python -m bot.backtest --synthetic --bars 6000 --equity 20000`. This uses a pure random walk, so the result **must not** show an edge. If it does, there is lookahead bias: find it and fix it.
3. Summarise the architecture in 10 lines or fewer, and list any bug or spec mismatch between `STRATEGY.md` and the code.

## Phase 1: Connect to IG DEMO

1. I will provide `.env` with `IG_API_KEY`, `IG_USERNAME`, `IG_PASSWORD`, `IG_ACCOUNT_ID` and `IG_ENV=DEMO`. Never print or log these.
2. Write `scripts/check_ig.py`. It should:
   - log in
   - print account equity
   - call `GET /markets/{epic}` for every market in `config.yaml`
   - print min deal size, min stop distance, expiry, currency, current spread and market status

   If an epic 404s, use `GET /markets?searchTerm=...` to find the correct **DFB spread-bet** epic and *propose* the fix. Don't silently change it.
3. Run `python -m bot.feasibility` and show the table.
4. Recommend which markets to keep, based only on the feasibility verdict and `spread_%_of_stop`. Wait for my approval before editing `markets:`.

## Phase 2: Real-data backtest

1. Get at least 3 years of 4H OHLC (mid prices, UTC) for the approved markets. Prefer a CSV export I provide. If you use the IG API, stay well inside the 10,000 points/week allowance and cache everything under `data/history/`. Convert to `data/history/<MARKET>.csv` in the format documented in `bot/data.py`, with `price_scale` correct for FX.
2. Run `python -m bot.backtest --data data/history`. Report the full, in-sample and out-of-sample metrics, the top rejection reasons and the promotion checks.
3. **Do not tune parameters** in this phase. If results fail the promotion criteria, report that honestly and list at most 3 hypotheses for why. I decide whether to run the strategy-change protocol.

## Phase 3: Hardening (one PR-sized change at a time, each with tests)

1. **Real P&L.** When a position closes server-side (stop hit), replace the estimated P&L in `LiveRunner.reconcile` with the actual figure from IG's transaction or activity history endpoint. Verify the endpoint against IG's API reference first.
2. **Alerts.** Send Telegram *or* email notifications on OPENED, CLOSED, STOP_MOVED, HALT, KILL_ALL, ERROR and a daily equity summary. Credentials go in env vars.
3. **Run as a service.** Provide Docker *or* systemd, with an auto-restart policy, log rotation and a health heartbeat file. Restart must be safe: state reloads, reconcile runs before any new order, and no duplicate orders.
4. **Reports.** Add `scripts/weekly_report.py`. It reads `logs/trades.csv` and `logs/decisions.jsonl` and reports trades, win rate, profit factor, average R, drawdown, rejection reasons, and average live spread and slippage vs backtest assumptions.
5. **News blackout.** Read `data/blackouts.csv` (UTC start and end, markets affected, maintained by me) and block new entries in those windows. Add this as a `RiskManager.check_entry` reason with tests.
6. Keep `pytest -q` green. Keep `test_no_lookahead`, `test_no_edge_on_random_walk` and every guardrail test exactly as strict as they are now.

## Phase 4: Demo forward test

1. Tell me which single config change switches to real **demo** orders (`safety.dry_run: false`). I make that change myself.
2. Produce the weekly report every week. Flag any divergence from the backtest: fills, spreads, signal counts, rejected orders.
3. Exit criteria: at least 6 weeks, at least 30 trades, no unexplained errors, and live behaviour in line with the backtest.

## Phase 5: Live readiness (you prepare, I flip the switch)

Give me a written checklist. Do not edit any live flag yourself. The checklist must cover:

- API key for the **live** account
- `live_trading_confirmed: true`
- `IG_ENV=LIVE` plus the `--live` flag
- risk auto-capped at 0.5% for phase 1
- KILL file tested
- alerts tested
- funding amount confirmed

## Hard rules (repeat of AGENTS.md, in case of conflict)

- Never loosen anything under `risk:` or `safety:`, or `HARD_LIMITS`.
- Never place an order without a server-side stop. Never widen a stop.
- No martingale, averaging down, grid or AI-driven trade decisions.
- If you believe a guardrail is costing money, write it up as a *suggestion with evidence*. Don't implement it.

## Report format (end of every phase)

```
PHASE n: DONE / BLOCKED
What I did:        (3-6 bullets)
Results:           (table or numbers)
Tests:             pytest -q -> N passed
Risks / concerns:  (bullets)
Decisions needed from you: (numbered)
```
