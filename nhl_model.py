#!/usr/bin/env python3
"""NHL Poisson goal-scoring model.

Fetches current (and if empty, prior) season team summary stats from the
NHL public stats API (no key required). Projects per-matchup goal rates
and runs a Monte Carlo simulation similar to mlb_model but with hockey's
smaller OT/shootout coin-flip for tied games.

Public endpoints used:
  * https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId=YYYYZZZZ...
  * https://api-web.nhle.com/v1/standings/now

No API key. Cached locally for 24 hours to keep page loads fast.
"""
import json
import math
import os
import random
import time
from datetime import date
from urllib.error import URLError
from urllib.request import Request, urlopen

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CACHE_DIR, "nhl_team_stats_cache.json")
CACHE_MAX_AGE_H = 24

INITIAL_ELO = 1500.0
NHL_HFA_GOALS = 0.14  # ~0.14 goal home advantage per side historically
LEAGUE_AVG_GPG = 2.95  # NHL avg goals per team per game, recent seasons

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _fetch_json(url, timeout=15):
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _current_season_id():
    """Season id formatted YYYYZZZZ (e.g. 20242025 for 2024-25)."""
    d = date.today()
    if d.month >= 7:
        return int(f"{d.year}{d.year + 1}")
    return int(f"{d.year - 1}{d.year}")


def _prior_season_id(sid):
    """20242025 -> 20232024 (subtract 10001)."""
    return sid - 10001


def fetch_team_stats(season_id=None):
    season_id = season_id or _current_season_id()
    url = (f"https://api.nhle.com/stats/rest/en/team/summary"
           f"?cayenneExp=seasonId={season_id}%20and%20gameTypeId=2")
    try:
        data = _fetch_json(url)
    except Exception:
        return []
    out = []
    for r in data.get("data", []):
        gp = r.get("gamesPlayed", 0) or 0
        if not gp:
            continue
        out.append({
            "team_name": r.get("teamFullName", ""),
            "games": gp,
            "gf": r.get("goalsFor", 0),
            "ga": r.get("goalsAgainst", 0),
            "gf_per": r.get("goalsForPerGame", 0) or 0,
            "ga_per": r.get("goalsAgainstPerGame", 0) or 0,
            "points": r.get("points", 0),
            "wins": r.get("wins", 0),
            "losses": r.get("losses", 0),
            "ot_losses": r.get("otLosses", 0),
            "points_pct": r.get("pointPct", 0),
            "pp_pct": r.get("powerPlayPct", 0),
            "pk_pct": r.get("penaltyKillPct", 0),
            "faceoff_pct": r.get("faceoffWinPct", 0),
            "season_id": season_id,
        })
    return out


def get_or_fetch_team_stats():
    """Return {team_name: stats_dict}. Falls back to prior season if current empty."""
    now = time.time()
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                data = json.load(f)
            if now - data.get("ts", 0) < CACHE_MAX_AGE_H * 3600:
                return data["teams"]
        except Exception:
            pass

    sid = _current_season_id()
    teams = fetch_team_stats(sid)
    season_used = sid
    # Early-season fallback: if teams have < 3 games played, use prior season
    if not teams or teams[0]["games"] < 3:
        prior = _prior_season_id(sid)
        prior_teams = fetch_team_stats(prior)
        if prior_teams:
            teams = prior_teams
            season_used = prior

    out = {t["team_name"]: t for t in teams}
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({"ts": now, "teams": out, "season": season_used}, f)
    except OSError:
        pass
    return out


def project_lambdas(home_name, away_name, stats=None):
    """Return (λ_home_goals, λ_away_goals) from team season averages."""
    if stats is None:
        stats = get_or_fetch_team_stats()
    h = stats.get(home_name) or {}
    a = stats.get(away_name) or {}
    h_gf = h.get("gf_per") or LEAGUE_AVG_GPG
    h_ga = h.get("ga_per") or LEAGUE_AVG_GPG
    a_gf = a.get("gf_per") or LEAGUE_AVG_GPG
    a_ga = a.get("ga_per") or LEAGUE_AVG_GPG
    lam_h = ((h_gf + a_ga) / 2.0) + NHL_HFA_GOALS / 2
    lam_a = ((a_gf + h_ga) / 2.0) - NHL_HFA_GOALS / 2
    return max(0.3, lam_h), max(0.3, lam_a)


def simulate_match(home_name, away_name, n=10000, seed=None):
    """Monte Carlo simulate N games with Poisson goals + OT/SO coin-flip."""
    stats = get_or_fetch_team_stats()
    lam_h, lam_a = project_lambdas(home_name, away_name, stats)
    rng = random.Random(seed) if seed is not None else random

    def _pois(lam):
        if lam <= 0:
            return 0
        L = math.exp(-lam); k = 0; p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return k - 1

    home_wins = away_wins = ot_games = 0
    h_goals_sum = a_goals_sum = 0
    totals = []
    margins = []
    score_counts = {}

    for _ in range(n):
        hg = _pois(lam_h)
        ag = _pois(lam_a)
        if hg == ag:
            # Regulation tie → OT/shootout resolves. Small home edge.
            ot_games += 1
            if rng.random() < 0.52:
                hg += 1
            else:
                ag += 1
        if hg > ag:
            home_wins += 1
        else:
            away_wins += 1
        h_goals_sum += hg
        a_goals_sum += ag
        totals.append(hg + ag)
        margins.append(hg - ag)
        key = f"{hg}-{ag}"
        score_counts[key] = score_counts.get(key, 0) + 1

    top_scores = sorted(score_counts.items(), key=lambda kv: -kv[1])[:10]
    top_scores = [{"score": k, "pct": v / n * 100} for k, v in top_scores]

    return {
        "trials": n,
        "home_team": home_name,
        "away_team": away_name,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "home_win_pct": home_wins / n * 100,
        "away_win_pct": away_wins / n * 100,
        "draw_pct": 0.0,
        "ot_pct": ot_games / n * 100,
        "avg_home_goals": h_goals_sum / n,
        "avg_away_goals": a_goals_sum / n,
        "expected_total": lam_h + lam_a,
        "mean_total": sum(totals) / n,
        "margins": margins,
        "totals": totals,
        "most_likely_scores": top_scores,
    }


def predict_moneyline(home_name, away_name):
    """Return {home, away, draw: None} probability dict (NHL ML is 2-way inc OT/SO)."""
    sim = simulate_match(home_name, away_name, n=5000)
    return {"home": sim["home_win_pct"] / 100,
            "away": sim["away_win_pct"] / 100,
            "draw": None}


if __name__ == "__main__":
    stats = get_or_fetch_team_stats()
    print(f"Loaded {len(stats)} NHL teams")
    if stats:
        sample = list(stats.values())[0]
        print(f"Sample: {sample['team_name']} — GP {sample['games']}, "
              f"GF/G {sample['gf_per']:.2f}, GA/G {sample['ga_per']:.2f}, "
              f"Pts% {sample['points_pct']:.3f}")
