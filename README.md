# First Pitch

Daily MLB scoreboard + Elo-based win-probability model, with a walk-forward
backtest report. All data comes from the free MLB statsapi (no API keys).

## Run locally

```bash
pip install -r requirements.txt
python mlb_ui.py
```

Open http://127.0.0.1:5000

## Deploy to Render (free tier)

1. Push this folder to a new GitHub repo (see `push_to_github.md` in this
   directory).
2. Sign in at https://render.com (GitHub sign-in is easiest).
3. New + → **Web Service** → connect the repo you just pushed.
4. Render auto-detects the Procfile. Confirm settings:
   - Environment: **Python**
   - Build command: `pip install -r requirements.txt`
   - Start command: (leave as detected from Procfile)
   - Instance type: **Free**
5. Click **Create Web Service**. First build takes ~3–5 min.
6. Your app will be live at `https://<your-service-name>.onrender.com`.

**Note:** Render's free tier spins the app down after 15 min of inactivity.
The first request after a sleep takes ~30 seconds to wake up; subsequent
requests are snappy. The fitted model cache is committed so cold boots
don't have to refit from scratch.

## Files

- `mlb_ui.py` — Flask app: `/`, `/edges`, `/market`, `/backtest`, `/export.csv`
- `mlb_model.py` — Elo + SP prediction model + walk-forward backtest
- `mlb_odds.py` — Pinnacle + Polymarket + The Odds API integration, EV + Kelly math
- `mlb_backtest_cache.json` — pre-fit model state (refit daily)

## Tabs

- **Schedule** `/` — every game with model prediction, Pinnacle odds comparison, EV
- **Edges** `/edges` — positive-EV moneyline plays sorted by edge, with Kelly stake sizing
- **Market** `/market` — Pinnacle limits (sharp-money proxy) + Polymarket futures
- **Model** `/backtest` — walk-forward backtest report with calibration diagram

## Optional: The Odds API integration

To add DraftKings / FanDuel / BetMGM / Caesars best-price lookup to the game cards and Edges tab,
set the environment variable `ODDS_API_KEY` to your key from https://the-odds-api.com. The Starter
tier is $30/mo and gives 20k requests, enough to poll hourly during the season. The integration is
already scaffolded in `mlb_odds.py` and activates automatically when the key is present. On Render,
set it under Settings → Environment → Add Environment Variable.
