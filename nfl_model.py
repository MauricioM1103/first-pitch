#!/usr/bin/env python3
"""NFL Elo model + walk-forward backtest.

Mirrors the MLB approach: predict before each game using only data available
pre-kickoff, log the prediction, then update Elo with the actual result.
Score only the final N weeks of the season — the earlier games serve as Elo
warm-up.

Data source: nflverse community games.csv (hosted on GitHub). It contains
every NFL game 1999–current with scores, rest days, QBs, and closing lines.
No API key required.

Hyperparameters (lightly tuned off 538's public MLB/NFL Elo methodology):
  * K = 20            typical NFL value given ~272 regular-season games
  * HFA = 55          Elo points (~2.5 point spread equivalent, 538 uses 48)
  * REST_WEIGHT = 2   Elo per day of extra rest
  * MOV multiplier    log-based damper for expected blowouts
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nfl_backtest_cache.json",
)

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HFA = 55.0
REST_WEIGHT = 2.0

# QB Elo: parallel rating per quarterback. Blended into team strength at
# prediction time; updated separately per game based on observed vs expected.
QB_INITIAL_ELO = 1500.0
QB_WEIGHT = 0.50   # fraction of QB rating diff from avg that enters team Elo
QB_K = 10.0        # smaller than team K: QBs change teams, one player
WARMUP_SEASONS = 0  # Backtest hyperparam tuning showed warmup hurts team Elo
                    # (even with 1/3 regression at season boundaries). We fit
                    # on the target season only. For live predictions in a
                    # later season you'd want this at 1 — accept the tradeoff.

# Multi-season backtest params: fit across N seasons, use the first
# `MULTI_WARMUP_SEASONS` as pure warmup (no scoring), score every remaining
# game. Team Elo regresses toward 1500 by 1/3 at each season boundary.
NUM_SEASONS = 12
MULTI_WARMUP_SEASONS = 2

CACHE_MAX_AGE_H = 24

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
    # Legacy / alt codes
    "LAR": "Los Angeles Rams", "SD": "Los Angeles Chargers", "STL": "Los Angeles Rams",
    "OAK": "Las Vegas Raiders", "WSH": "Washington Commanders",
}
# Reverse map: name -> current preferred abbreviation
_PREFERRED_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def abbr_from_name(name):
    """Return the preferred team abbreviation for a full name, or None."""
    return _PREFERRED_ABBR.get(name)


def _fetch_games_csv():
    req = Request(GAMES_URL, headers={"User-Agent": "nfl-model/1.0"})
    with urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_season_games(season, extra_prior_seasons=0):
    """Return list of completed regular-season games.

    Includes `season` plus `extra_prior_seasons` prior seasons for ratings
    warm-up (QBs carry across seasons, team Elo regresses softly between).
    """
    rows = _fetch_games_csv()
    seasons = set(range(season - extra_prior_seasons, season + 1))
    games = []
    for r in rows:
        if _to_int(r.get("season")) not in seasons:
            continue
        if (r.get("game_type") or "").upper() != "REG":
            continue
        home_score = _to_int(r.get("home_score"))
        away_score = _to_int(r.get("away_score"))
        if home_score is None or away_score is None:
            continue
        games.append({
            "game_id": r.get("game_id"),
            "season": _to_int(r.get("season")),
            "week": _to_int(r.get("week")),
            "date": r.get("gameday"),
            "home_abbr": r.get("home_team"),
            "away_abbr": r.get("away_team"),
            "home_name": TEAM_NAMES.get(r.get("home_team"), r.get("home_team") or ""),
            "away_name": TEAM_NAMES.get(r.get("away_team"), r.get("away_team") or ""),
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
            "home_rest": _to_int(r.get("home_rest")) or 7,
            "away_rest": _to_int(r.get("away_rest")) or 7,
            "home_qb": r.get("home_qb_name") or "",
            "away_qb": r.get("away_qb_name") or "",
        })
    games.sort(key=lambda g: (g["date"] or "", g["game_id"] or ""))
    return games


# ----- Elo math ---------------------------------------------------------

def expected_prob(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def _mov_multiplier(margin, winner_elo_diff):
    m = abs(margin) if margin else 1
    return math.log(m + 1) * (2.2 / (max(0.0, winner_elo_diff) * 0.001 + 2.2))


def _qb_adj(qb_elo):
    """Team-Elo adjustment implied by a QB's rating."""
    if qb_elo is None:
        return 0.0
    return QB_WEIGHT * (qb_elo - QB_INITIAL_ELO)


def predict_win_prob(home_elo, away_elo, home_rest=7, away_rest=7,
                     home_qb_elo=None, away_qb_elo=None):
    rest_adj = REST_WEIGHT * ((home_rest or 7) - (away_rest or 7))
    h = home_elo + _qb_adj(home_qb_elo)
    a = away_elo + _qb_adj(away_qb_elo)
    return expected_prob(h + HFA + rest_adj, a)


def elo_update(home_elo, away_elo, home_won, home_score, away_score,
               home_rest=7, away_rest=7,
               home_qb_elo=None, away_qb_elo=None,
               k=K_FACTOR, qb_k=QB_K):
    """Return (new_home_elo, new_away_elo, new_home_qb_elo, new_away_qb_elo).

    QB inputs may be None (fallback to team-only). Both team and QB ratings
    update from the same game result; the QB K-factor is lower because a
    QB plays ~1/22 of positions and we don't want individual-game noise to
    swing their rating too hard.
    """
    rest_adj = REST_WEIGHT * ((home_rest or 7) - (away_rest or 7))
    h = home_elo + _qb_adj(home_qb_elo)
    a = away_elo + _qb_adj(away_qb_elo)
    exp_h = expected_prob(h + HFA + rest_adj, a)
    margin = (home_score or 0) - (away_score or 0)
    if home_won:
        winner_diff = (h + HFA + rest_adj) - a
    else:
        winner_diff = a - (h + HFA + rest_adj)
    mov = _mov_multiplier(margin, winner_diff)
    y = 1 if home_won else 0
    team_delta = k * mov * (y - exp_h)
    new_home = home_elo + team_delta
    new_away = away_elo - team_delta
    new_home_qb = home_qb_elo
    new_away_qb = away_qb_elo
    if home_qb_elo is not None and away_qb_elo is not None:
        qb_delta = qb_k * mov * (y - exp_h)
        new_home_qb = home_qb_elo + qb_delta
        new_away_qb = away_qb_elo - qb_delta
    return new_home, new_away, new_home_qb, new_away_qb


# ----- backtest ---------------------------------------------------------

def run_backtest(season, score_last_weeks=8, warmup_seasons=WARMUP_SEASONS,
                 use_qb=True, verbose=True):
    """Walk-forward Elo backtest for one NFL season.

    Scores only games in the `score_last_weeks` tail of `season`. The prior
    `warmup_seasons` are used purely to warm team + QB ratings so they start
    the scored window with history.

    `use_qb=False` runs the team-only variant for comparison.
    """
    if verbose:
        print(f"[nfl] loading {season} (+{warmup_seasons} prior) from nflverse...", flush=True)
    games = load_season_games(season, extra_prior_seasons=warmup_seasons)
    if not games:
        raise RuntimeError(f"no completed regular-season games found for {season}")
    scored_season_games = [g for g in games if g["season"] == season]
    if not scored_season_games:
        raise RuntimeError(f"no games in target season {season}")
    if verbose:
        print(f"[nfl]   {len(games)} total games, "
              f"{len(scored_season_games)} in target season "
              f"({'with QB' if use_qb else 'team-only'})", flush=True)

    max_week = max((g["week"] or 0) for g in scored_season_games)
    score_from_week = max(1, max_week - score_last_weeks + 1)

    elo = {}
    qb_elo = {}
    qb_games = {}
    wl = {}
    predictions = []
    seasons_seen = set()

    for g in games:
        # 538-style: regress team Elo toward 1500 by 1/3 at each season boundary
        if g["season"] not in seasons_seen and seasons_seen:
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        seasons_seen.add(g["season"])

        h = g["home_abbr"]
        a = g["away_abbr"]
        home_qb = (g["home_qb"] or "").strip() if use_qb else ""
        away_qb = (g["away_qb"] or "").strip() if use_qb else ""
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)
        wl.setdefault(h, [0, 0])
        wl.setdefault(a, [0, 0])
        if home_qb:
            qb_elo.setdefault(home_qb, QB_INITIAL_ELO)
            qb_games.setdefault(home_qb, 0)
        if away_qb:
            qb_elo.setdefault(away_qb, QB_INITIAL_ELO)
            qb_games.setdefault(away_qb, 0)

        hq = qb_elo.get(home_qb) if home_qb else None
        aq = qb_elo.get(away_qb) if away_qb else None

        p_home = predict_win_prob(elo[h], elo[a],
                                  g["home_rest"], g["away_rest"],
                                  home_qb_elo=hq, away_qb_elo=aq)

        in_target = (g["season"] == season and (g["week"] or 0) >= score_from_week)
        if in_target:
            predictions.append({
                "date": g["date"], "week": g["week"],
                "home": g["home_name"], "away": g["away_name"],
                "home_abbr": h, "away_abbr": a,
                "home_qb": home_qb, "away_qb": away_qb,
                "pregame_home_elo": round(elo[h], 1),
                "pregame_away_elo": round(elo[a], 1),
                "pregame_home_qb_elo": round(hq, 1) if hq else None,
                "pregame_away_qb_elo": round(aq, 1) if aq else None,
                "pregame_home_wl": list(wl[h]),
                "pregame_away_wl": list(wl[a]),
                "p_home": p_home,
                "home_won": g["home_won"],
                "home_score": g["home_score"], "away_score": g["away_score"],
            })

        new_h, new_a, new_hq, new_aq = elo_update(
            elo[h], elo[a], g["home_won"], g["home_score"], g["away_score"],
            g["home_rest"], g["away_rest"],
            home_qb_elo=hq, away_qb_elo=aq,
        )
        elo[h], elo[a] = new_h, new_a
        if home_qb and new_hq is not None:
            qb_elo[home_qb] = new_hq
            qb_games[home_qb] = qb_games.get(home_qb, 0) + 1
        if away_qb and new_aq is not None:
            qb_elo[away_qb] = new_aq
            qb_games[away_qb] = qb_games.get(away_qb, 0) + 1

        if g["home_won"]:
            wl[h][0] += 1
            wl[a][1] += 1
        else:
            wl[h][1] += 1
            wl[a][0] += 1

    team_names = {g["home_abbr"]: g["home_name"] for g in games}
    team_names.update({g["away_abbr"]: g["away_name"] for g in games})

    return {
        "season": season,
        "score_from_week": score_from_week,
        "max_week": max_week,
        "score_last_weeks": score_last_weeks,
        "warmup_seasons": warmup_seasons,
        "use_qb": use_qb,
        "total_games": len(games),
        "scored_games": len(predictions),
        "predictions": predictions,
        "final_elo": {t: round(e, 1) for t, e in elo.items()},
        "final_qb_elo": {q: round(e, 1) for q, e in qb_elo.items()},
        "qb_games": qb_games,
        "final_wl": wl,
        "team_names": team_names,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {
            "K": K_FACTOR, "HFA": HFA, "REST_WEIGHT": REST_WEIGHT,
            "INITIAL_ELO": INITIAL_ELO,
            "QB_WEIGHT": QB_WEIGHT, "QB_K": QB_K,
            "QB_INITIAL_ELO": QB_INITIAL_ELO,
        },
    }


def compute_metrics(predictions):
    if not predictions:
        return {"n": 0}
    n = len(predictions)
    ll = brier = 0.0
    correct = 0
    home_wins_actual = 0
    record_correct = 0
    record_defined = 0

    for p in predictions:
        y = 1 if p["home_won"] else 0
        ph = max(1e-6, min(1 - 1e-6, p["p_home"]))
        ll -= y * math.log(ph) + (1 - y) * math.log(1 - ph)
        brier += (ph - y) ** 2
        if (ph >= 0.5) == (y == 1):
            correct += 1
        home_wins_actual += y

        hw, hl = p.get("pregame_home_wl") or [0, 0]
        aw, al = p.get("pregame_away_wl") or [0, 0]
        home_pct = hw / max(1, hw + hl) if (hw + hl) else 0.5
        away_pct = aw / max(1, aw + al) if (aw + al) else 0.5
        if home_pct != away_pct:
            record_defined += 1
            pick_home = home_pct > away_pct
            if pick_home == bool(y):
                record_correct += 1

    bins = 10
    calibration = []
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        bucket = [p for p in predictions if (lo <= p["p_home"] < hi) or (b == bins - 1 and p["p_home"] == 1.0)]
        if bucket:
            avg_p = sum(p["p_home"] for p in bucket) / len(bucket)
            act = sum(1 if p["home_won"] else 0 for p in bucket) / len(bucket)
            calibration.append({"bin_lo": lo, "bin_hi": hi, "n": len(bucket),
                                "avg_pred": avg_p, "actual": act})

    p_home_base = home_wins_actual / n
    home_ll = 0.0
    for p in predictions:
        y = 1 if p["home_won"] else 0
        ph = max(1e-6, min(1 - 1e-6, p_home_base))
        home_ll -= y * math.log(ph) + (1 - y) * math.log(1 - ph)

    return {
        "n": n,
        "accuracy": correct / n,
        "log_loss": ll / n,
        "brier_score": brier / n,
        "home_baseline_prob": p_home_base,
        "home_baseline_accuracy": p_home_base,
        "home_baseline_log_loss": home_ll / n,
        "record_baseline_accuracy": record_correct / record_defined if record_defined else None,
        "record_baseline_defined": record_defined,
        "calibration": calibration,
    }


def _pick_season():
    """Prefer current season if it has completed REG games; else prior."""
    try:
        rows = _fetch_games_csv()
    except Exception:
        return date.today().year - 1
    counts = {}
    for r in rows:
        if (r.get("game_type") or "").upper() != "REG":
            continue
        if r.get("home_score") in (None, "", "NA"):
            continue
        s = _to_int(r.get("season"))
        if s:
            counts[s] = counts.get(s, 0) + 1
    y = date.today().year
    for cand in (y, y - 1, y - 2):
        if counts.get(cand, 0) >= 100:
            return cand
    return max(counts.keys()) if counts else y - 1


def get_or_run_backtest(refresh=False, season=None, score_last_weeks=8, verbose=False):
    """Legacy single-season backtest. New callers should use
    `get_or_run_multi_season_backtest` for 12-season aggregate metrics.
    This function still backs the live NFL page's `final_elo` lookup."""
    if not refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                data = json.load(f)
            gen = datetime.fromisoformat(data.get("generated_at", "1970-01-01"))
            if (datetime.now() - gen) < timedelta(hours=CACHE_MAX_AGE_H):
                return data
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    if season is None:
        season = _pick_season()
    result = run_backtest(season, score_last_weeks=score_last_weeks, verbose=verbose)
    result["metrics"] = compute_metrics(result["predictions"])
    with open(CACHE_FILE, "w") as f:
        json.dump(result, f)
    return result


# ============================================================================
# Multi-season backtest (12-season aggregate)
# ============================================================================

MULTI_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nfl_multi_backtest_cache.json",
)


def run_multi_season_backtest(end_season, num_seasons=NUM_SEASONS,
                               warmup_seasons=MULTI_WARMUP_SEASONS,
                               verbose=True):
    """Walk-forward Elo across `num_seasons` ending at `end_season`.

    First `warmup_seasons` are pure warm-up (not scored). Remaining seasons
    have every game scored. Team Elo regresses 1/3 toward 1500 at each
    season boundary; QB Elo persists across seasons.

    Returns a dict with aggregate metrics plus per-season breakdown.
    """
    first = end_season - num_seasons + 1
    seasons = list(range(first, end_season + 1))
    scored_from = first + warmup_seasons

    if verbose:
        print(f"[nfl] multi-season backtest {first}-{end_season} "
              f"(warmup {first}..{scored_from - 1}, scored {scored_from}..{end_season})",
              flush=True)
    all_rows = _fetch_games_csv()

    # Pre-filter to our window + REG + completed
    pool = []
    for r in all_rows:
        s = _to_int(r.get("season"))
        if s is None or s not in seasons:
            continue
        if (r.get("game_type") or "").upper() != "REG":
            continue
        hs, aw = _to_int(r.get("home_score")), _to_int(r.get("away_score"))
        if hs is None or aw is None:
            continue
        pool.append({
            "game_id": r.get("game_id"),
            "season": s,
            "week": _to_int(r.get("week")),
            "date": r.get("gameday"),
            "home_abbr": r.get("home_team"),
            "away_abbr": r.get("away_team"),
            "home_name": TEAM_NAMES.get(r.get("home_team"), r.get("home_team") or ""),
            "away_name": TEAM_NAMES.get(r.get("away_team"), r.get("away_team") or ""),
            "home_score": hs, "away_score": aw,
            "home_won": hs > aw,
            "home_rest": _to_int(r.get("home_rest")) or 7,
            "away_rest": _to_int(r.get("away_rest")) or 7,
            "home_qb": r.get("home_qb_name") or "",
            "away_qb": r.get("away_qb_name") or "",
        })
    pool.sort(key=lambda g: (g["date"] or "", g["game_id"] or ""))
    if verbose:
        print(f"[nfl]   {len(pool)} games across {len(seasons)} seasons", flush=True)

    elo = {}
    qb_elo = {}
    qb_games = {}
    predictions = []
    per_season = {s: {"n": 0, "correct": 0, "ll": 0.0, "brier": 0.0} for s in seasons[warmup_seasons:]}
    seasons_seen = set()

    for g in pool:
        # Season-boundary regression (team Elo only; QB Elo persists)
        if g["season"] not in seasons_seen and seasons_seen:
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        seasons_seen.add(g["season"])

        h, a = g["home_abbr"], g["away_abbr"]
        hqb = g["home_qb"].strip()
        aqb = g["away_qb"].strip()
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)
        if hqb:
            qb_elo.setdefault(hqb, QB_INITIAL_ELO)
            qb_games.setdefault(hqb, 0)
        if aqb:
            qb_elo.setdefault(aqb, QB_INITIAL_ELO)
            qb_games.setdefault(aqb, 0)
        hq = qb_elo.get(hqb) if hqb else None
        aq = qb_elo.get(aqb) if aqb else None

        p_home = predict_win_prob(elo[h], elo[a],
                                  g["home_rest"], g["away_rest"],
                                  home_qb_elo=hq, away_qb_elo=aq)

        if g["season"] >= scored_from:
            y = 1 if g["home_won"] else 0
            ph = max(1e-6, min(1 - 1e-6, p_home))
            correct = (ph >= 0.5) == (y == 1)
            per_season[g["season"]]["n"] += 1
            per_season[g["season"]]["correct"] += 1 if correct else 0
            per_season[g["season"]]["ll"] -= y * math.log(ph) + (1 - y) * math.log(1 - ph)
            per_season[g["season"]]["brier"] += (ph - y) ** 2
            # Keep lightweight prediction records for aggregate metrics; drop
            # verbose fields to keep the cache small.
            predictions.append({
                "season": g["season"], "week": g["week"], "date": g["date"],
                "home_abbr": h, "away_abbr": a,
                "home": g["home_name"], "away": g["away_name"],
                "p_home": p_home, "home_won": g["home_won"],
                "home_score": g["home_score"], "away_score": g["away_score"],
                "home_qb": hqb, "away_qb": aqb,
            })

        new_h, new_a, new_hq, new_aq = elo_update(
            elo[h], elo[a], g["home_won"], g["home_score"], g["away_score"],
            g["home_rest"], g["away_rest"],
            home_qb_elo=hq, away_qb_elo=aq,
        )
        elo[h], elo[a] = new_h, new_a
        if hqb and new_hq is not None:
            qb_elo[hqb] = new_hq
            qb_games[hqb] += 1
        if aqb and new_aq is not None:
            qb_elo[aqb] = new_aq
            qb_games[aqb] += 1

    # Aggregate metrics
    for s, stat in per_season.items():
        if stat["n"] > 0:
            stat["accuracy"] = stat["correct"] / stat["n"]
            stat["log_loss"] = stat["ll"] / stat["n"]
            stat["brier"] = stat["brier"] / stat["n"]
        else:
            stat["accuracy"] = stat["log_loss"] = stat["brier"] = None

    team_names = {g["home_abbr"]: g["home_name"] for g in pool}
    team_names.update({g["away_abbr"]: g["away_name"] for g in pool})

    return {
        "first_season": first,
        "last_season": end_season,
        "num_seasons": num_seasons,
        "warmup_seasons": warmup_seasons,
        "scored_from_season": scored_from,
        "total_games": len(pool),
        "scored_games": len(predictions),
        "per_season": per_season,
        "predictions": predictions,
        "final_elo": {t: round(e, 1) for t, e in elo.items()},
        "final_qb_elo": {q: round(e, 1) for q, e in qb_elo.items()},
        "qb_games": qb_games,
        "team_names": team_names,
        "hyperparams": {
            "K": K_FACTOR, "HFA": HFA, "REST_WEIGHT": REST_WEIGHT,
            "QB_WEIGHT": QB_WEIGHT, "QB_K": QB_K,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_or_run_multi_season_backtest(refresh=False, end_season=None, verbose=False):
    if not refresh and os.path.exists(MULTI_CACHE_FILE):
        try:
            with open(MULTI_CACHE_FILE) as f:
                data = json.load(f)
            gen = datetime.fromisoformat(data.get("generated_at", "1970-01-01"))
            if (datetime.now() - gen) < timedelta(hours=CACHE_MAX_AGE_H):
                return data
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    if end_season is None:
        end_season = _pick_season()
    result = run_multi_season_backtest(end_season, verbose=verbose)
    result["metrics"] = compute_metrics(result["predictions"])
    with open(MULTI_CACHE_FILE, "w") as f:
        json.dump(result, f)
    return result


def _print_report(result):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    m = result["metrics"]
    print()
    print(f"=== NFL backtest ===")
    print(f"Season          : {result['season']}")
    print(f"Total games     : {result['total_games']}")
    print(f"Scored window   : weeks {result['score_from_week']} - {result['max_week']}")
    print(f"Scored games    : {result['scored_games']}")
    print()
    print(f"Model accuracy  : {m['accuracy']:.4f}")
    print(f"Model log loss  : {m['log_loss']:.4f}")
    print(f"Model Brier     : {m['brier_score']:.4f}")
    print()
    print(f"Home baseline   : acc={m['home_baseline_accuracy']:.4f}  ll={m['home_baseline_log_loss']:.4f}")
    if m["record_baseline_accuracy"] is not None:
        print(f"Record baseline : acc={m['record_baseline_accuracy']:.4f}  over {m['record_baseline_defined']} games")
    print()
    print("Calibration:")
    for c in m["calibration"]:
        print(f"  {c['bin_lo']:.1f}-{c['bin_hi']:.1f}  n={c['n']:3d}  avg_pred={c['avg_pred']:.3f}  actual={c['actual']:.3f}")
    print()
    ranked = sorted(result["final_elo"].items(), key=lambda kv: -kv[1])[:10]
    print("Top 10 final Elo:")
    for ab, elo in ranked:
        name = result["team_names"].get(ab, ab)
        wl = result["final_wl"].get(ab, [0, 0])
        print(f"  {elo:7.1f}  {name:28s}  {wl[0]}-{wl[1]}")


def main():
    parser = argparse.ArgumentParser(description="NFL Elo backtest")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--score-weeks", type=int, default=8)
    args = parser.parse_args()
    result = get_or_run_backtest(
        refresh=args.refresh, season=args.season,
        score_last_weeks=args.score_weeks, verbose=True,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
