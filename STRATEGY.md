# Strategy spec: Regime-Switching Confluence (IG spread bets, 4H)

This is the source of truth for trading logic. `bot/strategy.py` implements it, and the backtest and live bot run the same code.

## 1. Philosophy

- **Risk control first.** Most retail spread-bet and CFD accounts lose money. IG's own risk warning shows about 70% of retail accounts losing. The usual causes are oversized positions, missing stops, adding to losers and overtrading. The rules below rule those out.
- **Fewer, non-redundant indicators.** Every indicator has a separate job. Stacking more indicators of the same type (RSI, Stochastic, CCI, Williams %R all measure momentum) adds parameters to overfit, not information.
- **Deterministic rules, no AI in the trading path.** Every decision can be backtested, reproduced and audited.
- **Expect long flat spells and losing streaks.** Trend following typically wins 35–45% of trades and makes its money from a few large winners. That is normal, not a bug.

## 2. Markets and timeframe

| | |
|---|---|
| Account | IG spread betting (GBP per point) |
| Bars | 4-hour (`HOUR_4`), closed bars only |
| Universe | EURUSD, GBPUSD, UK100, US500, GER40, GOLD (DFB epics in `config.yaml`) |
| Correlation groups | `fx_usd`, `equity_index`, `metals`: max 1 open position per group |

Trade only the markets that `python -m bot.feasibility` marks **OK** for your equity (see §8).

## 3. Indicators and what each one does

| Group | Indicator | Job |
|---|---|---|
| Trend | EMA 20 / 50 / 200 | Direction, structure, long-term bias |
| Trend strength | ADX(14), +DI / −DI | Regime detection (trending vs ranging), directional pressure |
| Momentum | MACD (12,26,9) histogram | Momentum confirmation |
| Momentum | RSI(14) | Avoid chasing exhausted moves; weakness filter for mean reversion |
| Volatility | ATR(14) | Stop distance, trailing stop, position size |
| Volatility | ATR% percentile (250 bars) | Skip dead markets and news shocks |
| Volatility | Bollinger Bands (20, 2σ) | Range extremes and the mean for mean reversion |
| Structure | Donchian channel (20, prior bars) | Breakout trigger |
| Mean reversion | RSI(2) | Short-term oversold/overbought trigger |

**Data hygiene.** Broker feeds occasionally contain bad ticks. On live IG data, Germany 40 once showed a 7,285-point range in a single 4H bar against a typical 144. All three rules below compare against the median range of the *previous* 50 bars, so nothing looks ahead:
- Wicks longer than 6× that median are trimmed.
- Each bar counts at most 4× the median in ATR.
- Any bar wider than 10× the median blocks new entries for 3 bars.

Volume is excluded on purpose: IG's volume is tick volume and isn't reliable.

## 4. Regime filter (decided on every closed bar)

- **TREND:** ADX ≥ 22. Only Module A can enter.
- **RANGE:** ADX ≤ 18. Only Module B can enter.
- **NEUTRAL:** 18 < ADX < 22. No new entries. The dead zone cuts whipsaw.
- **Volatility gate:** new entries only when the ATR% percentile is between 5 and 95.

## 5. Module A: trend confluence

**Long.** Score 1 point for each condition below. Entry needs **bias + trigger + a score of at least 5 out of 6**.

1. **Bias (mandatory):** close > EMA200
2. EMA20 > EMA50
3. +DI > −DI
4. MACD histogram > 0
5. 50 ≤ RSI(14) ≤ 75
6. **Trigger (mandatory):** either
   - a **breakout:** close > highest high of the previous 20 bars, or
   - a **pullback:** the low touched EMA20 within the last 3 bars, and close > EMA20, and close > the previous bar's high

**Short:** the exact mirror (RSI band 25–50). Allowed only where `allow_short: true`. Equity indices default to long-only.

**Stop:** entry − 2.0 × ATR, placed server-side with the order.

**Exits:**
- Chandelier trailing stop: highest high since entry − 3.0 × ATR. It only ever moves in the trade's favour.
- Trend failure: EMA20 crosses back through EMA50, and the trade exits at the next bar's open.

## 6. Module B: range mean reversion

**Long.** All conditions are required:
- close < lower Bollinger Band
- RSI(2) < 10
- RSI(14) < 40
- close > EMA200 (never buy dips in a downtrend)

**Short:** the mirror (> upper band, RSI(2) > 90, RSI(14) > 60, close < EMA200), only where shorts are allowed.

**Stop:** 2.5 × ATR, placed server-side with the order.

**Exits:** whichever comes first:
- close back at the Bollinger mid-line
- RSI(2) > 70 (short: < 30)
- 6 bars have passed (time stop)
- the stop is hit

## 7. Risk and position sizing (`bot/risk.py`)

**Stake per point** = (equity × risk%) ÷ stop distance in points. It is rounded **down** to the size step. If IG's minimum stake would risk more than the cap, the trade is **skipped**, never rounded up.

| Rule | Default | Code hard limit |
|---|---|---|
| Risk per trade | 1.0% of equity | 2.0% |
| Max open positions | 3 | 5 |
| Max per correlation group | 1 | – |
| Max total open risk | 3.0% | 6.0% |
| Max margin used | 50% of equity | 70% |
| Daily loss limit | 3%: no new entries until the next day | 5% |
| Weekly loss limit | 6%: no new entries until the next week | 10% |
| Max drawdown from peak | 15%: close everything, halt, manual reset only | 25% |
| 4 losses in a row | risk halves until a winner | – |
| 6 losses in a row | 48-hour pause on new entries | – |
| Max orders per day | 8 | – |
| Friday after 16:00 UTC | no new entries (weekend gap risk) | – |
| Live phase 1 | risk capped at 0.5% until the human approves full risk | – |

## 8. Small-capital reality

IG's minimum stakes are around £0.50–£1 per point. A 2×ATR stop on 4H bars is typically tens of points on FX, US500 and UK100, and well over a hundred on GER40. The equity needed to trade a market at 1% risk is therefore often **in the low thousands of pounds per market**.

Below that, the bot correctly skips the trade. The engine logs these skips as `UNSIZEABLE`; `bot.feasibility` marks such markets `SKIP (too small)`. Your options, in order of preference:

1. Trade only the markets the feasibility report marks OK. FX majors and gold usually need the least.
2. Add capital before going live.
3. Raise `risk_per_trade_pct` towards 1.5–2%. This is a conscious decision: drawdowns grow proportionally.

Never "fix" this by rounding the stake up or tightening stops below 2×ATR to make the maths work.

## 9. Execution

- Decisions are made on bar close and orders go out as market orders immediately after. The backtest fills at the next bar's open.
- Every order carries `stopDistance` (server-side). Orders without one are refused in code.
- The spread filter skips an entry when the live spread is above `max_spread_points` (out-of-hours spreads).
- The bot manages only positions it opened. Manual positions are ignored.

## 10. What this strategy is NOT

No martingale, no averaging down, no grid, no adding to losers, no removing or widening stops. No trading on news headlines. No AI deciding trades.

## 11. Promotion path

1. **Backtest:** at least 3 years of data, real costs. Pass criteria: ≥ 100 trades, profit factor ≥ 1.3, max drawdown ≤ 20%, out-of-sample profit factor ≥ 1.15.
2. **Demo forward test:** at least 6 weeks and at least 30 trades. Live fills and spreads must be in line with backtest assumptions.
3. **Live at 0.5% risk:** for the first 30 trades.
4. **Full risk:** only by human decision (`live_full_risk_approved: true`).
