# Running the bot 24/5 on Railway

The bot is a long-running Python worker. Railway builds it from the `Dockerfile` and keeps it running. State, the bar cache and logs live on a persistent volume, so restarts are safe.

## One-time setup

1. Put this folder in a **private** GitHub repo.
2. In Railway, create a new project and deploy from that repo. `railway.json` sets one replica and a restart-on-failure policy.
3. Add a **volume** to the service, mounted at `/data`.
4. Set these service **Variables**:

| Variable | Value | Notes |
|---|---|---|
| `IG_ENV` | `DEMO` | Demo account only until the promotion criteria are met |
| `IG_API_KEY` | your demo API key | From My IG → Settings → API keys, while logged into **demo** |
| `IG_USERNAME` | demo username | |
| `IG_PASSWORD` | demo password | |
| `IG_ACCOUNT_ID` | spread bet account id | Optional |
| `STATE_DIR` | `/data/state` | |
| `LOG_DIR` | `/data/logs` | |
| `BOT_DRY_RUN` | `true` | Log-only. Set to `false` to send real **demo** orders |
| `BOT_MARKETS` | e.g. `EURUSD,GOLD` | Optional: only the markets `bot.feasibility` marks OK |

Until the IG credentials are set, the bot idles and logs `waiting for credentials`. Changing any variable redeploys the service.

## Controls from your phone (Railway app or dashboard → Variables)

| Want to… | Do this |
|---|---|
| Stop new trades, keep managing open ones | `BOT_PAUSE=1` |
| Close the bot's positions and stop | `BOT_KILL=1` |
| Resume after a kill or drawdown halt | Remove `BOT_KILL`, then set `BOT_RESET_HALT` to the code shown in the logs (`reset code: XXXXXX`) |
| Switch from log-only to demo orders | `BOT_DRY_RUN=false` |
| See what it's doing | Railway → Deployments → Logs. Every decision is also kept in `/data/logs/decisions.jsonl` |

Open positions always carry a stop held by IG. If the service stops, those stops still protect you. You can also close any position yourself in the IG app.

## Going live (only after the demo forward test passes)

LIVE needs all three of these. That's deliberate.

1. In the repo, set `safety.live_trading_confirmed: true` in `config.yaml` and commit it.
2. Set `IG_ENV=LIVE` with the **live** API key, username and password.
3. Set `BOT_LIVE_FLAG=1`.

Live risk is capped at 0.5% per trade until `safety.live_full_risk_approved: true` is committed.

**Never run more than one replica.** The bot also holds a lock on the volume as a second safeguard.
