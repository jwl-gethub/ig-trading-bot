# IG Trading Bot: starter kit for Codex

A rule-based 4H trading bot for an IG UK spread betting account. It combines a regime filter, a trend module, a mean-reversion module and a sweep-and-reclaim module, wrapped in a strict risk engine and a learning loop.

| File | What it is |
|---|---|
| `CODEX_PROMPT.md` | Paste into Codex as the first task (phased plan) |
| `AGENTS.md` | Guardrails Codex reads automatically before every task |
| `STRATEGY.md` | Exact trading rules |
| `config.yaml` | Markets, parameters, risk limits, safety switches |
| `bot/` | Reference implementation: indicators, strategy, risk, backtester, IG client, live loop |
| `tests/` | 118 tests, including guardrail tests, a no-lookahead check and random-walk "no fake edge" checks |
| `bot/history.py` | Downloads real IG price history when the bot is idle (keeps 1,500 allowance points for live trading), then backtests the exact live rules on it; results in the `BACKTEST_REPORT` log event and `logs/backtest_report.json` |
| `research/` | Separate research service (no IG credentials, cannot trade): downloads free 1-min index data from Dukascopy and validates the intraday noise-area strategy before/after its 2024 publication |
| `DEPLOY.md` | Run it 24/5 on Railway, with phone-controllable kill, pause and dry-run switches |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # all green
python -m bot.backtest --synthetic          # random walk: should show NO edge
cp .env.example .env                        # add your IG DEMO API key + login
python -m bot.feasibility                   # which markets your equity can trade
python -m bot.live --once                   # one dry-run cycle on DEMO
```

**Getting an API key:** log in to IG, open the **demo** account, go to My IG → Settings → API keys and generate a key.

## Safety switches

| Action | How |
|---|---|
| Stop everything now | `touch KILL` or `BOT_KILL=1`: bot closes its positions and halts |
| Stop new entries | `touch PAUSE` or `BOT_PAUSE=1`. Remove it to resume |
| Clear a halt (human only) | `python -m bot.live --reset-halt` or `BOT_RESET_HALT=<code from logs>` |
| Go live | `IG_ENV=LIVE` + `safety.live_trading_confirmed: true` + `--live` / `BOT_LIVE_FLAG=1` (all three) |

No strategy guarantees profit. Backtest on real data first, run the demo for at least 6 weeks, then start live at 0.5% risk.
