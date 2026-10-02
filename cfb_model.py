#!/usr/bin/env python3
"""NCAAF (college football) simulator — normal-distribution scoring from team Elo.

Mirror of nfl_model's approach, with higher scoring (CFB avg ≈ 28 PPG) and
wider scoring variance (σ ≈ 15). Historical fit uses the CollegeFootballData
(CFBD) public API when CFBD_API_KEY is set in the environment; without it,
every team defaults to 1500 Elo and the simulator shows a visible fallback
note (the sport still functions, just without differentiated team ratings).

Public endpoint used (when key present):
  * https://api.collegefootballdata.com/games?year=YYYY&seasonType=regular
  * https://api.collegefootballdata.com/stats/season?year=YYYY

Get a free CFBD key: https://collegefootballdata.com/key
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
CACHE_PATH = os.path.join(CACHE_DIR, "cfb_elo_cache.json")
CACHE_MAX_AGE_H = 24

INITIAL_ELO = 1500.0
HFA_ELO = 65.0            # CFB HFA ~ 3 pts ≈ 65 Elo
K_FACTOR = 20.0
PPG_BASE = 28.0           # CFB avg scoring per team ~ 28
SIGMA_POINTS = 15.0       # team scoring stddev — larger than NFL
NUM_SEASONS = 5           # keep CFBD call volume reasonable


def is_fitted():
    return bool(os.environ.get("CFBD_API_KEY"))


def _cfbd_fetch(path, params):
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        return None
    from urllib.parse import urlencode
    url = f"https://api.collegefootballdata.com{path}?{urlencode(params)}"
    req = Request(url, headers={
        "Authorization": f"Bearer {key}",
        "User-Agent": "betting-tools/1.0",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def _mov_mult(margin):
    return math.log(max(1, abs(margin)) + 1)


def _elo_update(h_elo, a_elo, h_pts, a_pts, k=K_FACTOR):
    """Update ratings based on actual margin."""
    y = 1.0 if h_pts > a_pts else (0.5 if h_pts == a_pts else 0.0)
    exp_h = 1.0 / (1.0 + 10 ** (-((h_elo + HFA_ELO) - a_elo) / 400.0))
    mov = _mov_mult(h_pts - a_pts)
    delta = k * mov * (y - exp_h)
    return h_elo + delta, a_elo - delta


def fit_team_elo(num_seasons=NUM_SEASONS, verbose=False):
    """Walk-forward Elo fit from CFBD game data. Returns {team_name: elo}.

    Falls back to {} if CFBD key not set or API fails.
    """
    if not is_fitted():
        return {}
    today = date.today()
    end_year = today.year if today.month >= 7 else today.year - 1
    years = list(range(end_year - num_seasons + 1, end_year + 1))

    elo = {}
    for year in years:
        games = _cfbd_fetch("/games", {"year": year, "seasonType": "regular"})
        if not games:
            continue
        # Season-boundary regression toward the mean
        if elo:
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        # Sort chronologically
        games.sort(key=lambda g: (g.get("week") or 0, g.get("start_date") or ""))
        for g in games:
            h_name = g.get("home_team"); a_name = g.get("away_team")
            h_pts = g.get("home_points"); a_pts = g.get("away_points")
            if h_pts is None or a_pts is None:
                continue
            elo.setdefault(h_name, INITIAL_ELO)
            elo.setdefault(a_name, INITIAL_ELO)
            elo[h_name], elo[a_name] = _elo_update(
                elo[h_name], elo[a_name], h_pts, a_pts
            )
        if verbose:
            print(f"[cfb] {year}: {len(games)} games, {len(elo)} teams fit")
    return elo


def get_or_fit_team_elo():
    """Cached Elo fit — refreshed daily when CFBD key is present."""
    now = time.time()
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                data = json.load(f)
            if now - data.get("ts", 0) < CACHE_MAX_AGE_H * 3600:
                return data["elo"]
        except Exception:
            pass
    elo = fit_team_elo()
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({"ts": now, "elo": elo, "fitted": bool(elo)}, f)
    except OSError:
        pass
    return elo


def project_points(home_name, away_name):
    """Return (proj_home_pts, proj_away_pts) from team Elo + HFA."""
    elo = get_or_fit_team_elo()
    h_elo = elo.get(home_name, INITIAL_ELO)
    a_elo = elo.get(away_name, INITIAL_ELO)
    diff = (h_elo + HFA_ELO) - a_elo
    # Scale: 100 Elo ≈ 5 CFB points
    edge = diff / 20.0
    proj_h = PPG_BASE + edge / 2
    proj_a = PPG_BASE - edge / 2
    return proj_h, proj_a


def simulate_match(home_name, away_name, n=10000, seed=None):
    """Normal-distribution scoring simulator."""
    rng = random.Random(seed) if seed is not None else random
    proj_h, proj_a = project_points(home_name, away_name)

    home_wins = away_wins = ties = 0
    margins = []
    totals = []
    for _ in range(n):
        hs = max(0.0, rng.gauss(proj_h, SIGMA_POINTS))
        asc = max(0.0, rng.gauss(proj_a, SIGMA_POINTS))
        m = hs - asc
        margins.append(m)
        totals.append(hs + asc)
        if abs(m) < 0.5:
            ties += 1
        elif hs > asc:
            home_wins += 1
        else:
            away_wins += 1

    return {
        "trials": n,
        "home_team": home_name,
        "away_team": away_name,
        "home_win_pct": home_wins / n * 100,
        "away_win_pct": away_wins / n * 100,
        "draw_pct": ties / n * 100,
        "avg_home_points": proj_h,
        "avg_away_points": proj_a,
        "expected_total": proj_h + proj_a,
        "mean_total": sum(totals) / n,
        "margins": margins,
        "fitted": is_fitted(),
    }


def predict_moneyline(home_name, away_name):
    sim = simulate_match(home_name, away_name, n=5000)
    return {"home": sim["home_win_pct"] / 100,
            "away": sim["away_win_pct"] / 100,
            "draw": None}


if __name__ == "__main__":
    print(f"CFBD key present: {is_fitted()}")
    elo = get_or_fit_team_elo()
    print(f"Teams fit: {len(elo)}")
    if elo:
        top = sorted(elo.items(), key=lambda kv: -kv[1])[:10]
        for n, e in top:
            print(f"  {e:7.1f}  {n}")
