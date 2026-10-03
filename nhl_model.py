#!/usr/bin/env python3
"""NHL Elo backtest + Poisson goal-scoring simulator.

Two complementary pieces, both fed by NHL's public stats API (no key):

1. Elo rating backtest (new): walk-forward fit from historical game results
   across the last N seasons. Each team starts at 1500; after every game
   their rating updates by K * MOV_mult * (actual - expected). Season-
   boundary regression toward the mean (1/3). Writes a slim side-file
   (nhl_final_elo.json) alongside the full predictions cache so the live
   site can look up team Elo without pulling the predictions list.

2. Poisson goal simulator (existing): projects per-game goal rates by
   blending team season averages with team Elo diff, then draws N
   independent samples with OT/SO coin-flip for ties.

Public endpoints used:
  * https://api.nhle.com/stats/rest/en/team/summary?cayenneExp=seasonId=YYYYZZZZ...
  * https://api.nhle.com/stats/rest/en/game?cayenneExp=seasonId=YYYYZZZZ...
  * https://api-web.nhle.com/v1/standings/now
"""
import json
import math
import os
import random
import time
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(CACHE_DIR, "nhl_team_stats_cache.json")
BACKTEST_CACHE_PATH = os.path.join(CACHE_DIR, "nhl_backtest_cache.json")
ELO_ONLY_CACHE_PATH = os.path.join(CACHE_DIR, "nhl_final_elo.json")
CACHE_MAX_AGE_H = 24
BACKTEST_MAX_AGE_H = 24 * 7  # Elo fit refreshed weekly

INITIAL_ELO = 1500.0
NHL_HFA_GOALS = 0.14    # ~0.14 goal home advantage per side historically
NHL_HFA_ELO = 35.0      # ~35 Elo ~ 0.14 goal / small advantage
K_FACTOR = 6.0          # smaller K for hockey (82-game seasons, higher variance)
LEAGUE_AVG_GPG = 2.95   # NHL avg goals per team per game, recent seasons
BACKTEST_SEASONS = 5    # ~410 games/season * 5 = 2050 games for fit

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


def project_lambdas(home_name, away_name, stats=None, elo=None):
    """Return (λ_home_goals, λ_away_goals) blending team season averages
    with team Elo differential (so a hot team's recent form pushes the
    projection without needing a mid-season rate recompute).
    """
    if stats is None:
        stats = get_or_fetch_team_stats()
    if elo is None:
        elo = get_or_fit_final_elo()
    h = stats.get(home_name) or {}
    a = stats.get(away_name) or {}
    h_gf = h.get("gf_per") or LEAGUE_AVG_GPG
    h_ga = h.get("ga_per") or LEAGUE_AVG_GPG
    a_gf = a.get("gf_per") or LEAGUE_AVG_GPG
    a_ga = a.get("ga_per") or LEAGUE_AVG_GPG
    base_h = ((h_gf + a_ga) / 2.0) + NHL_HFA_GOALS / 2
    base_a = ((a_gf + h_ga) / 2.0) - NHL_HFA_GOALS / 2
    # Elo blend: 100 Elo diff ≈ 0.25 goal shift
    h_elo = elo.get(home_name, INITIAL_ELO) if elo else INITIAL_ELO
    a_elo = elo.get(away_name, INITIAL_ELO) if elo else INITIAL_ELO
    diff = (h_elo + NHL_HFA_ELO) - a_elo
    shift = diff / 400.0  # 100 Elo → 0.25 goal
    lam_h = base_h + shift / 2
    lam_a = base_a - shift / 2
    return max(0.3, lam_h), max(0.3, lam_a)


# ---------------------------------------------------------------------------
# Elo backtest (walk-forward fit from NHL.com game endpoint)
# ---------------------------------------------------------------------------

def _mov_mult(margin):
    return math.log(max(1, abs(margin)) + 1)


def _elo_update(h_elo, a_elo, h_goals, a_goals, k=K_FACTOR):
    """Update ratings based on actual result (OT/SO winners count as regular W)."""
    y = 1.0 if h_goals > a_goals else (0.5 if h_goals == a_goals else 0.0)
    exp_h = 1.0 / (1.0 + 10 ** (-((h_elo + NHL_HFA_ELO) - a_elo) / 400.0))
    mov = _mov_mult(h_goals - a_goals)
    delta = k * mov * (y - exp_h)
    return h_elo + delta, a_elo - delta


_TEAM_NAME_CACHE = {}


def _fetch_team_names():
    """Return {team_id: fullName}. One-off call to the team stats endpoint."""
    if _TEAM_NAME_CACHE:
        return _TEAM_NAME_CACHE
    try:
        url = "https://api.nhle.com/stats/rest/en/team?limit=100"
        data = _fetch_json(url, timeout=20)
    except Exception:
        return {}
    for t in data.get("data", []) or []:
        tid = t.get("id")
        name = t.get("fullName") or t.get("rawTricode")
        if tid is not None and name:
            _TEAM_NAME_CACHE[tid] = name
    return _TEAM_NAME_CACHE


def fetch_season_games(season_id):
    """Pull completed regular-season games (gameType=2) for a season.

    NHL endpoint uses `season` (not seasonId) and `visitingTeamId`/`visitingScore`
    (not awayTeamId/awayScore). Pages back 100 at a time until it stops returning.
    """
    team_names = _fetch_team_names()
    games = []
    page = 0
    while True:
        offset = page * 100
        url = (f"https://api.nhle.com/stats/rest/en/game"
               f"?cayenneExp=season={season_id}%20and%20gameType=2"
               f"&limit=100&start={offset}")
        try:
            data = _fetch_json(url, timeout=25)
        except Exception:
            break
        rows = data.get("data", []) or []
        if not rows:
            break
        for g in rows:
            h = g.get("homeTeamId"); a = g.get("visitingTeamId")
            hs = g.get("homeScore"); as_ = g.get("visitingScore")
            state = g.get("gameStateId")
            if h is None or a is None or hs is None or as_ is None:
                continue
            # gameStateId: 7 = Final / Official. Skip in-progress or scheduled.
            if state not in (6, 7, 8):
                continue
            games.append({
                "game_id":    g.get("id"),
                "date":       (g.get("gameDate") or "")[:10],
                "home_id":    h,
                "away_id":    a,
                "home_name":  team_names.get(h, f"TeamID {h}"),
                "away_name":  team_names.get(a, f"TeamID {a}"),
                "home_goals": hs,
                "away_goals": as_,
                "season":     season_id,
            })
        page += 1
        if page > 20:  # safety cap (82 games * 32 teams / 2 = 1312/season, 14 pages)
            break
    games.sort(key=lambda r: (r["date"], r["game_id"] or 0))
    return games


def run_backtest(num_seasons=BACKTEST_SEASONS, verbose=False):
    """Walk-forward Elo fit across N recent NHL seasons."""
    end_sid = _current_season_id()
    # Last fully-completed season (don't fit on partial season)
    first_sid = _prior_season_id(end_sid) - (num_seasons - 1) * 10001
    seasons = []
    sid = first_sid
    for _ in range(num_seasons):
        seasons.append(sid)
        sid += 10001

    elo = {}
    predictions = []
    first_scored = seasons[1] if len(seasons) > 1 else seasons[0]  # warmup = 1st season
    for sid in seasons:
        if elo:  # Season-boundary regression toward 1500
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        games = fetch_season_games(sid)
        if verbose:
            print(f"[nhl backtest] {sid}: {len(games)} games")
        for g in games:
            h, a = g["home_name"] or f"T{g['home_id']}", g["away_name"] or f"T{g['away_id']}"
            elo.setdefault(h, INITIAL_ELO); elo.setdefault(a, INITIAL_ELO)
            exp_h = 1.0 / (1.0 + 10 ** (-((elo[h] + NHL_HFA_ELO) - elo[a]) / 400.0))
            if sid >= first_scored:
                predictions.append({
                    "date": g["date"], "home": h, "away": a,
                    "p_home": exp_h,
                    "home_win": g["home_goals"] > g["away_goals"],
                    "home_goals": g["home_goals"], "away_goals": g["away_goals"],
                    "season": sid,
                })
            elo[h], elo[a] = _elo_update(elo[h], elo[a],
                                          g["home_goals"], g["away_goals"])

    final_elo = {t: round(v, 1) for t, v in elo.items()}
    # Accuracy on scored window
    n = len(predictions); correct = sum(1 for p in predictions
                                        if (p["p_home"] >= 0.5) == p["home_win"])
    ll = 0.0
    import math as _m
    for p in predictions:
        y = 1.0 if p["home_win"] else 0.0
        ph = max(1e-6, min(1 - 1e-6, p["p_home"]))
        ll -= y * _m.log(ph) + (1 - y) * _m.log(1 - ph)
    return {
        "final_elo": final_elo,
        "predictions": predictions,
        "n_scored": n,
        "accuracy": (correct / n) if n else None,
        "log_loss": (ll / n) if n else None,
        "seasons": seasons,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {"K": K_FACTOR, "HFA_ELO": NHL_HFA_ELO,
                        "INITIAL_ELO": INITIAL_ELO, "regression": 2/3},
    }


def _write_elo_only(result):
    try:
        with open(ELO_ONLY_CACHE_PATH, "w") as f:
            json.dump({"final_elo": result.get("final_elo", {}),
                       "generated_at": result.get("generated_at"),
                       "accuracy": result.get("accuracy"),
                       "log_loss": result.get("log_loss")}, f)
    except OSError:
        pass


def get_or_fit_final_elo(refresh=False):
    """Lightweight path: just the {team: elo} dict."""
    if not refresh and os.path.exists(ELO_ONLY_CACHE_PATH):
        try:
            with open(ELO_ONLY_CACHE_PATH) as f:
                data = json.load(f)
            gen = datetime.fromisoformat(data.get("generated_at", "1970-01-01"))
            if (datetime.now() - gen) < timedelta(hours=BACKTEST_MAX_AGE_H):
                return data.get("final_elo", {})
        except Exception:
            pass
    if not refresh and os.path.exists(BACKTEST_CACHE_PATH):
        try:
            with open(BACKTEST_CACHE_PATH) as f:
                data = json.load(f)
            final = dict(data.get("final_elo") or {})
            _write_elo_only(data)
            import gc
            del data
            gc.collect()
            return final
        except Exception:
            pass
    # Full fit
    try:
        result = run_backtest(verbose=False)
    except Exception:
        return {}
    try:
        with open(BACKTEST_CACHE_PATH, "w") as f:
            json.dump(result, f)
    except OSError:
        pass
    _write_elo_only(result)
    return result.get("final_elo", {})


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
