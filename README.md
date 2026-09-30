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

- `mlb_ui.py` — Flask app: `/`, `/backtest`, `/export.csv`
- `mlb_model.py` — Elo + SP prediction model + walk-forward backtest
- `mlb_backtest_cache.json` — pre-fit model state (refit daily)
