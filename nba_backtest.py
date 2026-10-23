"""NBA Elo + pace fit from historical ESPN scoreboards.

Walk-forward over the last N days of NBA games (default 365). For each
game: predict with the pre-game ratings, grade, then update ratings
with the actual result. Produces per-team off/def/pace cached to
logs/nba_ratings_cache.json (mirrored via log_persist so the live
simulate_game() call picks it up on the next render).

Design choices:

- Uses ESPN scoreboard (same source our grader already uses) instead
  of nba_api. No new dependency, no NBA.com rate limits.
- Elo is a plain 1-dim rating that absorbs offense + defense.
  Updates with a small K and a margin-of-victory multiplier
  (classic 538-style NBA Elo).
- Pace is updated per-team as a simple EWMA of possession estimates
  derived from final score + an assumed league-avg efficiency.
- Walk-forward only — no lookahead. Ratings as-of game D are used
  to predict game D's result, THEN updated.
- Report:
  * ML log-loss and Brier score (lower is better)
  * Margin MAE (points of error on predicted spread)
  * Total-points MAE (points of error on predicted total)

Usage:
  # one-shot CLI
  python nba_backtest.py --days 365

  # or hit /debug/nba-backtest on the Render service
"""
import json
import os
import time
from datetime import date, datetime, timedelta
from urllib.request import Request, urlopen

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ymd}&limit=100"
RATINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "logs", "nba_ratings_cache.json")

# Elo tuning constants. These are the 538 NBA defaults, which are
# well-studied and work out of the box; we can retune after the first
# backtest run shows the error metrics.
K_BASE               = 20.0     # Elo K-factor base
MOV_MULT_COEFF       = 2.2      # margin-of-victory multiplier scale
MOV_MULT_LOG_BASE    = 0.001    # log-margin dampening
HOME_ELO_BONUS       = 80.0     # ~2.5 pts in NBA
START_RATING         = 1500.0

# Pace prior + EWMA — scores per-team of ~113 at 100 possessions is the
# modern-era baseline (2024-25 season was 113.0 off rating / 100 poss).
LEAGUE_AVG_OFF_RTG   = 112.0    # pts per 100 possessions
LEAGUE_AVG_PACE      = 100.0    # possessions per 48 min
PACE_EWMA_ALPHA      = 0.08


def _http_get_json(url, timeout=20):
    req = Request(url, headers={"User-Agent": "cheeks-model/1.0 (nba-backtest)"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_day(ymd):
    """Pull one date's completed NBA games. ymd = 'YYYYMMDD'."""
    try:
        data = _http_get_json(ESPN_SCOREBOARD.format(ymd=ymd))
    except Exception:
        return []
    out = []
    for ev in (data.get("events") or []):
        try:
            comp = (ev.get("competitions") or [{}])[0]
            status = ((comp.get("status") or {}).get("type") or {}).get("state")
            if status != "post":
                continue  # skip in-progress / future
            competitors = comp.get("competitors") or []
            if len(competitors) != 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            out.append({
                "date":       ymd,
                "home_slug":  (home.get("team") or {}).get("abbreviation", "").lower(),
                "away_slug":  (away.get("team") or {}).get("abbreviation", "").lower(),
                "home_pts":   int(home.get("score") or 0),
                "away_pts":   int(away.get("score") or 0),
            })
        except Exception:
            continue
    return out


def _ewma(prev, new, alpha):
    return (1 - alpha) * prev + alpha * new


def _elo_win_prob(home_rating, away_rating, home_bonus=HOME_ELO_BONUS):
    diff = home_rating + home_bonus - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _mov_multiplier(point_diff, elo_diff):
    """538-style MOV multiplier that keeps Elo updates reasonable even
    for blowouts. point_diff is winner_pts - loser_pts (always > 0);
    elo_diff is winner_rating - loser_rating (can be negative if an
    underdog won)."""
    return ((abs(point_diff) + 3) ** 0.8) / (7.5 + MOV_MULT_LOG_BASE * elo_diff)


def _estimate_pace(home_pts, away_pts, home_pace_prev, away_pace_prev):
    """Estimate possessions from final score + league-average
    efficiency, assuming both teams had the same number of
    possessions in the game. Returns the estimate; caller EWMA-
    blends it per team."""
    total = home_pts + away_pts
    # possessions * 2 * league_avg_eff / 100 = total points
    # → possessions = total / (2 * eff / 100)
    poss = total / (2 * LEAGUE_AVG_OFF_RTG / 100.0)
    return poss  # same for both teams in this estimate


def run_backtest(days_back=365, verbose=False):
    """Walk-forward Elo + pace fit over the last `days_back` days of
    NBA games. Returns a dict with per-team ratings + per-market error
    metrics. Writes the ratings to logs/nba_ratings_cache.json."""
    ratings  = {}   # team_slug → {"elo": float, "pace": float}
    picks_predicted = 0
    log_loss_sum    = 0.0
    brier_sum       = 0.0
    margin_err_sum  = 0.0
    total_err_sum   = 0.0

    end_date   = date.today()
    start_date = end_date - timedelta(days=days_back)
    games_graded = 0

    cursor = start_date
    while cursor <= end_date:
        ymd = cursor.strftime("%Y%m%d")
        for g in _fetch_day(ymd):
            h, a = g["home_slug"], g["away_slug"]
            if not h or not a:
                continue
            for s in (h, a):
                if s not in ratings:
                    ratings[s] = {"elo": START_RATING, "pace": LEAGUE_AVG_PACE}

            # PREDICT (before updating with this game's result)
            prior_h = ratings[h]
            prior_a = ratings[a]
            p_home_win = _elo_win_prob(prior_h["elo"], prior_a["elo"])
            # Rough margin prediction from Elo diff (~ 1 pt per 28 Elo).
            fair_margin = (prior_h["elo"] + HOME_ELO_BONUS - prior_a["elo"]) / 28.0
            pace_pred = (prior_h["pace"] + prior_a["pace"]) / 2.0
            fair_total = pace_pred * 2 * LEAGUE_AVG_OFF_RTG / 100.0

            # GRADE
            actual_margin = g["home_pts"] - g["away_pts"]
            actual_total  = g["home_pts"] + g["away_pts"]
            home_won = actual_margin > 0

            log_loss_sum += -(home_won * (0.0 if p_home_win <= 0 else __import__("math").log(max(1e-6, p_home_win)))
                             + (1 - home_won) * (0.0 if p_home_win >= 1 else __import__("math").log(max(1e-6, 1 - p_home_win))))
            brier_sum    += (p_home_win - (1 if home_won else 0)) ** 2
            margin_err_sum += abs(fair_margin - actual_margin)
            total_err_sum  += abs(fair_total - actual_total)
            picks_predicted += 1

            # UPDATE ELO (home side perspective)
            if actual_margin > 0:
                winner, loser = h, a
                winner_margin = actual_margin
                winner_elo    = prior_h["elo"] + HOME_ELO_BONUS
                loser_elo     = prior_a["elo"]
                outcome       = 1.0
            else:
                winner, loser = a, h
                winner_margin = -actual_margin
                winner_elo    = prior_a["elo"]
                loser_elo     = prior_h["elo"] + HOME_ELO_BONUS
                outcome       = 0.0

            elo_diff   = winner_elo - loser_elo
            mov_mult   = _mov_multiplier(winner_margin, elo_diff)
            k          = K_BASE * mov_mult
            delta      = k * (outcome - p_home_win)
            # Zero-sum move (delta points flow home → away or vice versa).
            ratings[h]["elo"] = prior_h["elo"] + delta
            ratings[a]["elo"] = prior_a["elo"] - delta

            # UPDATE PACE (EWMA per team on the game's implied possessions)
            poss_estimate = _estimate_pace(g["home_pts"], g["away_pts"],
                                            prior_h["pace"], prior_a["pace"])
            ratings[h]["pace"] = _ewma(prior_h["pace"], poss_estimate, PACE_EWMA_ALPHA)
            ratings[a]["pace"] = _ewma(prior_a["pace"], poss_estimate, PACE_EWMA_ALPHA)

            games_graded += 1
            if verbose and games_graded % 50 == 0:
                print(f"  {games_graded} games processed...")

        cursor += timedelta(days=1)

    # Save ratings cache (local + log_persist mirror).
    try:
        os.makedirs(os.path.dirname(RATINGS_PATH), exist_ok=True)
        with open(RATINGS_PATH, "w") as f:
            json.dump({
                "fit_at":     datetime.utcnow().isoformat() + "Z",
                "days_back":  days_back,
                "games":      games_graded,
                "ratings":    ratings,
            }, f, indent=2)
    except Exception:
        pass
    try:
        import log_persist
        log_persist.write_file("logs/nba_ratings_cache.json",
                                 json.dumps({"fit_at": datetime.utcnow().isoformat() + "Z",
                                              "days_back": days_back,
                                              "games": games_graded,
                                              "ratings": ratings}))
    except Exception:
        pass

    n = max(1, picks_predicted)
    return {
        "games":              games_graded,
        "teams":              len(ratings),
        "log_loss":           round(log_loss_sum / n, 4),
        "brier":              round(brier_sum / n, 4),
        "margin_mae":         round(margin_err_sum / n, 2),
        "total_mae":          round(total_err_sum / n, 2),
        "top_teams":          sorted(
            [{"team": t, "elo": round(r["elo"]), "pace": round(r["pace"], 1)}
             for t, r in ratings.items()],
            key=lambda x: -x["elo"],
        )[:10],
        "ratings_path":       RATINGS_PATH,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    print(f"NBA backtest: last {args.days} days of ESPN games...")
    t0 = time.time()
    result = run_backtest(days_back=args.days, verbose=args.verbose)
    print(f"\nFit complete in {int(time.time() - t0)}s")
    print(f"  games:       {result['games']}")
    print(f"  teams:       {result['teams']}")
    print(f"  log-loss:    {result['log_loss']}  (lower = better; <0.65 = decent)")
    print(f"  Brier:       {result['brier']}     (lower = better; <0.24 = decent)")
    print(f"  margin MAE:  {result['margin_mae']} pts  (<10 = decent)")
    print(f"  total MAE:   {result['total_mae']} pts  (<13 = decent)")
    print(f"\n  Top 10 by Elo:")
    for t in result["top_teams"]:
        print(f"    {t['team'].upper():5s}  elo={t['elo']}  pace={t['pace']}")
    print(f"\n  Ratings cache: {result['ratings_path']}")
