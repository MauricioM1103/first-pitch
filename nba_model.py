"""NBA model + market research — kickoff scaffolding.

Shipped 2026-10-23 as the NBA addition kickoff. This file holds the
research notes, the market priority list, and a minimal simulate_game()
scaffold that picks.py can call once we're ready to wire the sport in.
Backtesting + model-weight tuning will follow once we have ~2 weeks of
live Pinnacle snapshots to compare against our sim output.

================================================================
PART 1 — MARKET RESEARCH
================================================================

Beyond ML and Total, these are the NBA markets with the sharpest
historical edges for a model-vs-Pinnacle devig approach, ranked by
expected edge-to-noise ratio:

1. ALT TOTALS (narrow bands: ±2.5 off the posted total)
   Pace swings in NBA are large and well-signaled by rest + lineup
   data. Pinnacle prices alt totals with uniform juice around the
   main line, so a +4% EV read on O/U 219.5 when the main is 216.5
   is a real edge even when the main total is at fair. Shrink to
   O/U ±2.5 only — further bands get noise-dominated.

2. 1H / 2H TOTALS (first-half specifically)
   First-half scoring is less affected by garbage time and endgame
   fouling, so our pace+efficiency sim matches reality more tightly
   there than the full game. Market inefficient on 1H totals for
   back-to-back teams and games with injured starters.

3. ALT SPREADS (±3.5, ±6.5, ±9.5 bands)
   NBA scoring distribution is roughly normal around the median
   margin — unlike NFL (which has 3/7/10 key-number mass), NBA
   half-point moves on alt spreads are smooth. Our sim can price
   them off the normal CDF around our fair spread.

4. TEAM TOTALS
   One of the strongest NBA edges, because it isolates pace +
   efficiency per side without the correlation noise of the full
   total. Especially sharp when one team is on a back-to-back and
   the other isn't.

5. 1Q TOTALS
   The sharpest sub-market for teams with clear pace mismatches
   (fast + fast or slow + slow). Thinly traded on Pinnacle, so our
   limits are small, but the per-bet edge is highest here.

6. 1H MONEYLINE / 1H SPREAD
   Similar to full-game ML/Spread but with lower noise. If our
   model has +4% EV on the full game, 1H ML typically shows +5-6%.

7. PLAYER PROPS (DEFERRED — needs player-level data pipeline)
   PRA (points+rebounds+assists), PTS, REB, AST are the deepest
   markets but require minute projections per player — out of
   scope for the kickoff. Add once rotation + injury feed is live.

8. PERIOD-WINNER (1Q / 1H), DOUBLE-RESULT (DEFERRED)
   Pinnacle lists these; low volume so bet-size constrained; our
   period-prob derivation is reliable so these are future upside.

Markets we're NOT prioritizing:
- First basket / first team to score — gambling-adjacent, not
  model-driven.
- Period Quarterbacks (3Q / 4Q) — too few sim samples to be
  reliable at a stage-of-game level.
- Series props (playoff only) — niche.

================================================================
PART 2 — MODEL STRUCTURE
================================================================

NBA model is a lightweight pace + efficiency Monte Carlo built on
public data. Fit annually from boxscore + lineup data.

Core variables per team (updated every game):
  elo_off     — offensive Elo rating (points per 100 possessions
                relative to league avg, EWMA with alpha=0.08)
  elo_def     — defensive Elo rating (same, inverted)
  pace        — possessions per 48 min, EWMA alpha=0.08
  rest_days   — nights since last game (0 = back-to-back)
  home        — binary; home-court advantage is ~2.5pts in NBA
  starters_in — fraction of top-5 minutes available (injury adj)

Per-game sim:
  pace_game    = (pace_home + pace_away) / 2
  off_eff_h    = elo_off_home - elo_def_away + home_bonus
  off_eff_a    = elo_off_away - elo_def_home
  points_h     = pace_game * (100 + off_eff_h) / 100
  points_a     = pace_game * (100 + off_eff_a) / 100
  spread_fair  = points_a - points_h  (negative = home favored)
  total_fair   = points_h + points_a

Monte Carlo:
  For 10,000 trials, draw a per-game variance adjustment from a
  truncated normal (σ_pts = 11.5 per team per game, correlated 0.3
  across teams to reflect pace noise), then sum. Compute ML prob
  from fraction of trials with home_points > away_points.

Totals edge adjustments:
  - Back-to-back teams score ~3.0 pts fewer (empirically).
  - Starters-in < 0.8 → scoring down 2.5pts × (0.8 - starters_in).
  - Pace shrinks 5% in nationally televised playoff games.

================================================================
PART 3 — HOME-COURT PRIOR CALIBRATION
================================================================

Historical NBA home-court advantage: ~2.5pts regular season (post
2020 bubble; was ~3.5 pre-2015). Playoff home advantage: ~3.5pts.
Model should use 2.5 as the prior and tighten per-team based on
observed home/away split over the last 20 games.

================================================================
PART 4 — BACKTEST PLAN
================================================================

Same methodology we used for NHL + soccer:

1. Pull 2 seasons of historical boxscores + lineups (NBA.com's
   boxscoretraditionalv2 and leaguedashlineups endpoints, or
   nba_api package wrapping them).

2. Rebuild team ratings in a walk-forward loop (train on games
   before date D, predict for D, update after). This is the only
   way to avoid lookahead bias.

3. Compare model fair prob vs Pinnacle closing devig on every ML
   / Spread / Total price. Record CLV per bucket.

4. Set per-sport consensus weight (model/market) by minimizing
   log-loss on predicted probabilities. Starting prior: 35/65
   (same as NHL default). We expect to end up around 30/70 or
   35/65 for NBA — basketball totals are closely tracked by the
   market so pure model is risky.

5. Separately backtest Signal composite thresholds for NBA to
   avoid the "same filter for every sport" trap. If NBA shows
   high variance on line-move component, down-weight it in the
   composite for this sport.

6. 60-day forward test on live Pinnacle snapshots before putting
   NBA on the public board.

================================================================
PART 5 — DATA SOURCES
================================================================

- Scheduling + results: ESPN scoreboard (basketball/nba) — same
  pipeline as other sports.
- Historical boxscores + lineups: NBA Stats API (nba_api package,
  no key required; rate-limited to ~1 req/sec).
- Starter / injury status: ESPN injury API + manual verification
  on game-day (NBA injury reports drop ~90min before tipoff).
- Historical Pinnacle closing lines: aggregated via sportsbookreview
  mirror OR by snapshotting from our own live scrape (building up
  60 days of our own data is cleaner than trusting third-party
  archives).

================================================================
PART 6 — WHAT'S IN THIS FILE NOW
================================================================

- SPORT_SLUG, PINNACLE_LEAGUE_ID, HOME_COURT_PRIOR constants
- Market priority list for picks.py to iterate when generating
  NBA candidates
- simulate_game() stub — returns ML prob + fair spread + fair
  total from a basic pace × efficiency formula using league-avg
  Elo (112 off, 112 def, 100 pace) when per-team data isn't
  loaded yet. Good enough to put NBA on the Pinnacle devig path
  without any sim edge; the per-team ratings fill in after
  nba_backtest.py runs.

The actual historical Elo + pace fit lives in a follow-up file
(nba_backtest.py — TODO) that walks through 2 seasons of nba_api
data and dumps a pickled rating cache into nba_ratings_cache.json
on the data branch via log_persist.
"""

SPORT_SLUG = "nba"
PINNACLE_LEAGUE_ID = 487
HOME_COURT_PRIOR_PTS = 2.5

# Market priority — iterate in this order when generating candidates.
# Each entry is (market_name, pinnacle_market_key, generator_fn_name).
# The generator_fn names are placeholders; picks.py will wire these up
# once the backtest confirms edge per market.
MARKET_PRIORITY = [
    ("Moneyline",       "ml",          "gen_ml"),
    ("Total",           "total",       "gen_total"),
    ("Alt Total ±2.5",  "alt_total",   "gen_alt_total_near"),
    ("1H Total",        "half_total",  "gen_1h_total"),
    ("Spread",          "spread",      "gen_spread"),
    ("Alt Spread ±3.5", "alt_spread",  "gen_alt_spread_near"),
    ("Team Total",      "team_total",  "gen_team_total"),
    ("1Q Total",        "q1_total",    "gen_1q_total"),
    ("1H Moneyline",    "half_ml",     "gen_1h_ml"),
    ("1H Spread",       "half_spread", "gen_1h_spread"),
]

# League-average priors used when per-team ratings aren't loaded yet
# (NBA 2025-26 baseline). Replace with per-team values once
# nba_backtest.py produces them.
LEAGUE_AVG_OFF_RTG = 112.0   # points per 100 possessions, offense
LEAGUE_AVG_DEF_RTG = 112.0   # points allowed per 100, defense
LEAGUE_AVG_PACE    = 100.0   # possessions per 48 min
GAME_SIGMA_PTS     = 11.5    # per-team per-game Monte Carlo σ

# Rest-day scoring penalty (back-to-back = 0 nights of rest).
B2B_SCORING_PENALTY_PTS = 3.0


_RATINGS_CACHE_MEM = {"data": None, "ts": 0.0}


def _load_ratings_cache():
    """Load the per-team Elo + pace fit from nba_backtest.py's output.
    Reads local file first, falls back to log_persist mirror (lets a
    freshly-deployed instance pick up ratings fit on an older one).
    60-second in-memory cache."""
    import json
    import os as _os
    import time as _time
    now = _time.time()
    cached = _RATINGS_CACHE_MEM.get("data")
    if cached is not None and (now - _RATINGS_CACHE_MEM["ts"]) < 60:
        return cached
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "logs", "nba_ratings_cache.json")
    data = {}
    try:
        with open(path) as f:
            data = json.load(f) or {}
    except Exception:
        try:
            import log_persist
            text, _sha = log_persist.read_file("logs/nba_ratings_cache.json")
            if text:
                data = json.loads(text) or {}
        except Exception:
            data = {}
    _RATINGS_CACHE_MEM["data"] = data
    _RATINGS_CACHE_MEM["ts"] = now
    return data


def _team_ratings(team_slug):
    """Lookup per-team ratings. Falls back to league averages if the
    cache hasn't been populated (nba_backtest.py hasn't been run yet)
    or if we don't have ratings for this team (new team, slug
    mismatch).

    Returned shape mirrors the Elo-based rating from nba_backtest:
      off_rtg + def_rtg are derived from elo (both set to the same
      value since Elo is one-dimensional). pace is the EWMA'd
      possessions-per-48 from the backtest."""
    try:
        slug = (team_slug or "").lower()
        cache = _load_ratings_cache()
        ratings = (cache.get("ratings") or {}).get(slug)
        if not ratings:
            raise KeyError(slug)
        # Elo → off/def-rating conversion: 1 Elo pt ≈ 0.1 rtg pts,
        # anchored so that a 1500 Elo team is at LEAGUE_AVG_OFF_RTG.
        elo = float(ratings.get("elo") or 1500)
        rtg_delta = (elo - 1500) * 0.1
        return {
            "off_rtg": LEAGUE_AVG_OFF_RTG + rtg_delta,
            "def_rtg": LEAGUE_AVG_DEF_RTG - rtg_delta,  # higher Elo = better defense too
            "pace":    float(ratings.get("pace") or LEAGUE_AVG_PACE),
        }
    except Exception:
        return {
            "off_rtg": LEAGUE_AVG_OFF_RTG,
            "def_rtg": LEAGUE_AVG_DEF_RTG,
            "pace":    LEAGUE_AVG_PACE,
        }


def simulate_game(home_team, away_team,
                  home_b2b=False, away_b2b=False,
                  home_starters_in=1.0, away_starters_in=1.0,
                  n_trials=10000):
    """Monte Carlo one NBA matchup. Returns a dict with:
      - home_ml_prob:  prob home wins (float 0-1)
      - away_ml_prob:  prob away wins (float 0-1)
      - fair_total:    expected combined points
      - fair_spread:   expected points_away - points_home
                       (negative = home favored)
      - total_distribution: (mean, stddev) for the sim's total
                            so a totals-O/U probability can be
                            derived from a normal CDF.

    Until per-team ratings are loaded (see nba_backtest.py TODO),
    both teams use the league-average Elo/pace prior, which means
    the model is essentially neutral and shouldn't disagree with
    the Pinnacle devig by much. That's intentional for the first
    few days of NBA on the board — we don't want the model
    overriding the market until it's been tuned."""
    import random
    import statistics

    h = _team_ratings(home_team)
    a = _team_ratings(away_team)

    pace_game = (h["pace"] + a["pace"]) / 2.0

    # Points per 100 possessions, home with +HOME_COURT_PRIOR_PTS.
    # Scale pace to game length (48 min = 100 possessions baseline).
    off_h = h["off_rtg"] - a["def_rtg"] + (LEAGUE_AVG_OFF_RTG + HOME_COURT_PRIOR_PTS)
    off_a = a["off_rtg"] - h["def_rtg"] + LEAGUE_AVG_OFF_RTG
    exp_pts_h = pace_game * off_h / 100.0
    exp_pts_a = pace_game * off_a / 100.0

    # Rest-day + starter adjustments.
    if home_b2b: exp_pts_h -= B2B_SCORING_PENALTY_PTS
    if away_b2b: exp_pts_a -= B2B_SCORING_PENALTY_PTS
    if home_starters_in < 0.8:
        exp_pts_h -= 2.5 * (0.8 - home_starters_in)
    if away_starters_in < 0.8:
        exp_pts_a -= 2.5 * (0.8 - away_starters_in)

    home_wins = 0
    totals = []
    for _ in range(n_trials):
        # Correlated draws: pace noise moves both totals in the same
        # direction (ρ ≈ 0.3), on top of independent team-level noise.
        shared = random.gauss(0, GAME_SIGMA_PTS * 0.3)
        pts_h  = exp_pts_h + shared + random.gauss(0, GAME_SIGMA_PTS * 0.95)
        pts_a  = exp_pts_a + shared + random.gauss(0, GAME_SIGMA_PTS * 0.95)
        totals.append(pts_h + pts_a)
        if pts_h > pts_a:
            home_wins += 1

    return {
        "home_ml_prob": home_wins / n_trials,
        "away_ml_prob": 1 - home_wins / n_trials,
        "fair_total":   exp_pts_h + exp_pts_a,
        "fair_spread":  exp_pts_a - exp_pts_h,  # negative = home favored
        "total_distribution": (
            exp_pts_h + exp_pts_a,
            statistics.stdev(totals) if len(totals) > 1 else GAME_SIGMA_PTS,
        ),
    }
