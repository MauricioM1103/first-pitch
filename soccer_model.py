#!/usr/bin/env python3
"""Soccer 3-way Elo model + walk-forward backtest.

Mirrors the MLB and NFL approach: predict before each match using only
pre-kickoff data, log the prediction, then update Elo with the actual
result. Score the last N scored seasons after a warm-up period.

Data source: football-data.co.uk CSV files. European leagues (EPL,
La Liga, Serie A, Bundesliga, Ligue 1) use the "main" per-season
format; Liga MX uses the "extra" bundle (all seasons in one file).
No API keys, no scraping — just well-known public CSVs.

3-way probability construction:
  * Standard Elo gives P(home not lose) as the usual sigmoid.
  * We split that into P(home) and P(draw) using a draw factor that
    peaks when the two teams are rated similarly (close matches draw
    more often, blowouts almost never draw).
  * K-factor applies with a log(|margin|+1) multiplier for blowouts.

Runs standalone:
    python soccer_model.py --league epl
    python soccer_model.py --league epl --refresh
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- model hyperparameters ---------------------------------------------
INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HFA = 100.0           # DEFAULT home-advantage Elo; per-league values override

# Per-league HFA tuned off empirical home-win rates. Historical home-win %
# for the backtest window (2013-2025 Elo+DC fit):
#   EPL        ~45%  → HFA ≈ 85  (lowest — biggest road teams, crowds vary)
#   La Liga    ~47%  → HFA ≈ 95
#   Serie A    ~47%  → HFA ≈ 95
#   Bundesliga ~45%  → HFA ≈ 85
#   Ligue 1    ~46%  → HFA ≈ 90
#   Liga MX    ~52%  → HFA ≈ 130 (big altitude + travel edges)
# Also: international friendlies have smaller home edge (~55 Elo) than
# qualifiers / competitive matches (~110 Elo). These numbers are empirical.
_PER_LEAGUE_HFA = {
    "epl":        85.0,
    "laliga":     95.0,
    "seriea":     95.0,
    "bundesliga": 85.0,
    "ligue1":     90.0,
    "ligamx":    130.0,
    "ucl":        90.0,
    "europa":     90.0,
    "international_friendly":   55.0,
    "international_competitive":110.0,
}


def hfa_for(slug):
    """Return the per-league HFA in Elo, or the default HFA if unknown."""
    return _PER_LEAGUE_HFA.get(slug, HFA)


def predict_international_match(home_team, away_team, is_friendly=False,
                                 draw_factor=None):
    """Project an international match using national-team Elo + per-type HFA.

    Falls back to league-prior Dixon-Coles (via simulate_match with slug=epl)
    when either side isn't in the national Elo snapshot. Returns a dict
    shaped like simulate_match's output so the picks consensus layer and
    the MC page can read it with no branching.
    """
    try:
        import national_team_elo
    except Exception:
        return None
    h_elo = national_team_elo.get_national_elo(home_team)
    a_elo = national_team_elo.get_national_elo(away_team)
    if h_elo is None or a_elo is None:
        return None  # unknown nation — caller should fall back

    hfa = hfa_for("international_friendly" if is_friendly
                  else "international_competitive")
    p_h, p_d, p_a = predict_3way(h_elo, a_elo, hfa=hfa)
    # Project goal totals from Elo diff: ~1.3 goals per team as a baseline
    # for international competitive, nudged by the Elo diff (100 Elo ≈ 0.25 g).
    base = 1.25
    shift = ((h_elo + hfa) - a_elo) / 400.0  # 100 Elo → +0.25 goal
    lam_h = max(0.3, base + shift / 2)
    lam_a = max(0.3, base - shift / 2)
    mat = dixon_coles_matrix(lam_h, lam_a)
    btts_yes = sum(mat[hh][aa] for hh in range(1, DC_MAX_GOALS + 1)
                                 for aa in range(1, DC_MAX_GOALS + 1))
    home_wn  = sum(mat[hh][0] for hh in range(1, DC_MAX_GOALS + 1))
    away_wn  = sum(mat[0][aa] for aa in range(1, DC_MAX_GOALS + 1))
    marg_h = [sum(mat[hh][aa] for aa in range(DC_MAX_GOALS + 1))
              for hh in range(DC_MAX_GOALS + 1)]
    marg_a = [sum(mat[hh][aa] for hh in range(DC_MAX_GOALS + 1))
              for aa in range(DC_MAX_GOALS + 1)]
    def _tt_over(marg, line):
        return sum(p for g, p in enumerate(marg) if g > line) * 100
    return {
        "home_win_pct": p_h * 100,
        "draw_pct":     p_d * 100,
        "away_win_pct": p_a * 100,
        "btts_yes_pct": btts_yes * 100,
        "btts_no_pct":  (1 - btts_yes) * 100,
        "home_wn_pct":  home_wn * 100,
        "away_wn_pct":  away_wn * 100,
        "home_tt_over_1_5": _tt_over(marg_h, 1.5),
        "home_tt_over_2_5": _tt_over(marg_h, 2.5),
        "away_tt_over_1_5": _tt_over(marg_a, 1.5),
        "away_tt_over_2_5": _tt_over(marg_a, 2.5),
        "avg_home_goals": lam_h,
        "avg_away_goals": lam_a,
        "expected_total": lam_h + lam_a,
        "most_likely_scores": _top_scores_from_mat(mat, 10),
        "lam_home": lam_h, "lam_away": lam_a,
        "home_elo": h_elo, "away_elo": a_elo, "hfa_used": hfa,
        "engine":   "national-team Elo + DC",
    }


def _top_scores_from_mat(mat, n=10):
    scores = []
    for h in range(DC_MAX_GOALS + 1):
        for a in range(DC_MAX_GOALS + 1):
            scores.append((h, a, mat[h][a]))
    scores.sort(key=lambda s: -s[2])
    return [{"score": f"{h}-{a}", "pct": p * 100} for h, a, p in scores[:n]]


# K-factor downweight for friendlies — the review flagged that we should
# move national-team ratings LESS on a friendly than on a World Cup qualifier.
# Picks pipeline will pass is_friendly=True for internationals.
K_FRIENDLY_FACTOR = 0.4
DRAW_FACTOR = 0.28    # empirical draw rate at team parity
NUM_SEASONS = 12
WARMUP_SEASONS = 2
CACHE_MAX_AGE_H = 24

LEAGUES = {
    "epl": {
        "name": "England - Premier League",
        "code": "E0",
        "format": "main",
        "start_season": None,  # auto-pick
    },
    "laliga": {
        "name": "Spain - La Liga",
        "code": "SP1",
        "format": "main",
        "start_season": None,
    },
    "bundesliga": {
        "name": "Germany - Bundesliga",
        "code": "D1",
        "format": "main",
        "start_season": None,
    },
    "seriea": {
        "name": "Italy - Serie A",
        "code": "I1",
        "format": "main",
        "start_season": None,
    },
    "ligue1": {
        "name": "France - Ligue 1",
        "code": "F1",
        "format": "main",
        "start_season": None,
    },
    "ligamx": {
        "name": "Mexico - Liga MX",
        "code": "MEX",
        "format": "extra",
        "start_season": None,
    },
}


def cache_file(slug):
    return os.path.join(CACHE_DIR, f"soccer_{slug}_backtest_cache.json")


# ============================================================================
# fetch
# ============================================================================

def _fetch_text(url, timeout=30):
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    })
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _season_code(year):
    """2024 -> '2425' (football-data.co.uk main format)."""
    a = year % 100
    b = (year + 1) % 100
    return f"{a:02d}{b:02d}"


def _parse_date(s):
    """football-data.co.uk uses dd/mm/yy or dd/mm/yyyy."""
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_main_csv(text, season_year):
    """Main format: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, ..."""
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        # Handle BOM in first column
        home = r.get("HomeTeam") or r.get("﻿HomeTeam") or r.get(" HomeTeam")
        away = r.get("AwayTeam") or r.get("﻿AwayTeam") or r.get(" AwayTeam")
        if not home or not away:
            continue
        d = _parse_date(r.get("Date") or r.get("﻿Date"))
        if not d:
            continue
        try:
            hg = int(r.get("FTHG") or 0)
            ag = int(r.get("FTAG") or 0)
        except (TypeError, ValueError):
            continue
        res = r.get("FTR") or ("H" if hg > ag else ("A" if ag > hg else "D"))
        rows.append({
            "date": d.isoformat(),
            "home": home.strip(),
            "away": away.strip(),
            "hg": hg,
            "ag": ag,
            "res": res,
            "season": season_year,
        })
    return rows


def _parse_extra_csv(text):
    """Extra format: Country, League, Season, Date, Home, Away, HG, AG, Res, ..."""
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        # BOM stripping
        country = r.get("Country") or r.get("﻿Country")
        home = r.get("Home")
        away = r.get("Away")
        if not home or not away:
            continue
        d = _parse_date(r.get("Date"))
        if not d:
            continue
        try:
            hg = int(r.get("HG") or 0)
            ag = int(r.get("AG") or 0)
        except (TypeError, ValueError):
            continue
        res = r.get("Res") or ("H" if hg > ag else ("A" if ag > hg else "D"))
        # Season like "2024/2025" -> use starting year
        s = r.get("Season", "")
        try:
            season_year = int(s.split("/")[0]) if "/" in s else int(s)
        except (ValueError, IndexError):
            season_year = d.year
        rows.append({
            "date": d.isoformat(),
            "home": home.strip(),
            "away": away.strip(),
            "hg": hg,
            "ag": ag,
            "res": res,
            "season": season_year,
        })
    return rows


def load_league_matches(league_cfg, end_season, num_seasons=NUM_SEASONS, verbose=False):
    """Return sorted list of matches for the league across the past N seasons."""
    code = league_cfg["code"]
    fmt = league_cfg["format"]
    first = end_season - num_seasons + 1
    matches = []

    if fmt == "main":
        # One request per season
        def _one(y):
            url = f"https://www.football-data.co.uk/mmz4281/{_season_code(y)}/{code}.csv"
            try:
                text = _fetch_text(url)
                return y, _parse_main_csv(text, y)
            except Exception as e:
                return y, []

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_one, y): y for y in range(first, end_season + 1)}
            for f in as_completed(futs):
                y, rows = f.result()
                if verbose:
                    print(f"[soccer:{code}]   {y}/{y+1}: {len(rows)} matches", flush=True)
                matches.extend(rows)
    else:
        # Extra: one bundle, filter by season
        url = f"https://www.football-data.co.uk/new/{code}.csv"
        try:
            text = _fetch_text(url)
            all_rows = _parse_extra_csv(text)
        except Exception as e:
            all_rows = []
        matches = [r for r in all_rows if r["season"] in range(first, end_season + 1)]
        if verbose:
            per = {}
            for r in matches:
                per[r["season"]] = per.get(r["season"], 0) + 1
            for s in sorted(per):
                print(f"[soccer:{code}]   {s}/{s+1}: {per[s]} matches", flush=True)

    matches.sort(key=lambda r: (r["date"], r["home"]))
    return matches


# ============================================================================
# model math
# ============================================================================

def _p_home_not_lose(home_elo, away_elo, hfa=HFA):
    diff = (home_elo + hfa) - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def predict_3way(home_elo, away_elo, hfa=HFA):
    """Return (p_home, p_draw, p_away) summing to 1.

    hfa: home-field advantage in Elo. Pass soccer_model.hfa_for(league_slug)
    so each league gets its empirically-fitted edge instead of a flat 100.
    """
    p_hw = _p_home_not_lose(home_elo, away_elo, hfa)
    # Draw probability peaks at p_hw = 0.5, drops to zero at extremes
    p_draw = DRAW_FACTOR * (1.0 - abs(2.0 * p_hw - 1.0))
    p_home = p_hw - p_draw / 2.0
    p_away = 1.0 - p_hw - p_draw / 2.0
    p_home = max(0.0, p_home)
    p_draw = max(0.0, p_draw)
    p_away = max(0.0, p_away)
    s = p_home + p_draw + p_away
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_home / s, p_draw / s, p_away / s


def _mov_multiplier(margin):
    m = max(1, abs(margin))
    return math.log(m + 1)


def elo_update(home_elo, away_elo, result, hg, ag, k=K_FACTOR,
               hfa=HFA, is_friendly=False):
    """Update ratings based on actual result (H/D/A) with MOV multiplier.

    is_friendly: when True (international friendlies), apply the friendly
    K-factor downweight — these shouldn't move national-team ratings as
    much as a World Cup qualifier or Nations League match.
    """
    y = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
    exp_h = _p_home_not_lose(home_elo, away_elo, hfa)
    mov = _mov_multiplier(hg - ag)
    effective_k = k * (K_FRIENDLY_FACTOR if is_friendly else 1.0)
    delta = effective_k * mov * (y - exp_h)
    return home_elo + delta, away_elo - delta


# ============================================================================
# backtest
# ============================================================================

def run_multi_season_backtest(league_slug, end_season, num_seasons=NUM_SEASONS,
                                warmup_seasons=WARMUP_SEASONS, verbose=True):
    league = LEAGUES[league_slug]
    first = end_season - num_seasons + 1
    seasons = list(range(first, end_season + 1))
    scored_from = first + warmup_seasons

    if verbose:
        print(f"[soccer:{league_slug}] {first}-{end_season} "
              f"(warmup {first}..{scored_from - 1}, scored {scored_from}..{end_season})",
              flush=True)
    t0 = time.time()
    matches = load_league_matches(league, end_season, num_seasons, verbose)
    if verbose:
        print(f"[soccer:{league_slug}] {len(matches)} matches in {time.time()-t0:.1f}s",
              flush=True)

    elo = {}
    # Walk-forward team goal rates for Dixon-Coles projections (BTTS/totals use these)
    goal_stats = {}  # team -> {"gs": int, "ga": int, "matches": int}
    predictions = []
    per_season = {s: {"n": 0, "correct": 0, "ll": 0.0, "brier": 0.0,
                      "n_dc": 0, "ll_dc": 0.0, "correct_dc": 0}
                  for s in seasons[warmup_seasons:]}
    seasons_seen = set()
    prior = _LEAGUE_GOAL_PRIORS.get(league_slug, 1.35)

    def _team_rate(team, kind):
        s = goal_stats.get(team)
        if not s or s["matches"] < 3:
            return prior
        return s[kind] / s["matches"]

    for m in matches:
        s_year = m["season"]
        if s_year not in seasons_seen and seasons_seen:
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
            goal_stats = {}  # DC rates reset each season for recency
        seasons_seen.add(s_year)

        h, a = m["home"], m["away"]
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)

        # Primary prediction: Elo-based 3-way (slight accuracy edge over DC
        # in backtests). Dixon-Coles still runs in parallel for comparison
        # and powers BTTS/totals predictions where its low-score correction
        # genuinely helps.
        p_h, p_d, p_a = predict_3way(elo[h], elo[a], hfa=hfa_for(league_slug))

        # Dixon-Coles comparison prediction
        hr_scored = _team_rate(h, "gs")
        hr_conceded = _team_rate(h, "ga")
        ar_scored = _team_rate(a, "gs")
        ar_conceded = _team_rate(a, "ga")
        lam_h = max(0.1, (hr_scored + ar_conceded) / 2.0 * 1.15)
        lam_a = max(0.1, (ar_scored + hr_conceded) / 2.0 * 0.90)
        dc_mat = dixon_coles_matrix(lam_h, lam_a, DC_RHO_DEFAULT)
        p_h_dc = sum(dc_mat[hh][aa] for hh in range(DC_MAX_GOALS + 1)
                                      for aa in range(hh))
        p_d_dc = sum(dc_mat[k][k] for k in range(DC_MAX_GOALS + 1))
        p_a_dc = max(0.0, 1.0 - p_h_dc - p_d_dc)

        if s_year >= scored_from:
            res = m["res"]
            y_h = 1.0 if res == "H" else 0.0
            y_d = 1.0 if res == "D" else 0.0
            y_a = 1.0 if res == "A" else 0.0

            p_hh = max(1e-6, min(1 - 1e-6, p_h))
            p_dd = max(1e-6, min(1 - 1e-6, p_d))
            p_aa = max(1e-6, min(1 - 1e-6, p_a))
            ll = -(y_h * math.log(p_hh) + y_d * math.log(p_dd) + y_a * math.log(p_aa))
            brier = ((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2) / 3
            pred_pick = max(("H", p_h), ("D", p_d), ("A", p_a), key=lambda x: x[1])[0]
            correct = (pred_pick == res)

            per_season[s_year]["n"] += 1
            per_season[s_year]["correct"] += 1 if correct else 0
            per_season[s_year]["ll"] += ll
            per_season[s_year]["brier"] += brier

            # DC comparison metrics
            p_hh_dc = max(1e-6, min(1 - 1e-6, p_h_dc))
            p_dd_dc = max(1e-6, min(1 - 1e-6, p_d_dc))
            p_aa_dc = max(1e-6, min(1 - 1e-6, p_a_dc))
            ll_dc = -(y_h * math.log(p_hh_dc) + y_d * math.log(p_dd_dc) + y_a * math.log(p_aa_dc))
            pick_dc = max(("H", p_h_dc), ("D", p_d_dc), ("A", p_a_dc), key=lambda x: x[1])[0]
            per_season[s_year]["n_dc"] += 1
            per_season[s_year]["correct_dc"] += 1 if (pick_dc == res) else 0
            per_season[s_year]["ll_dc"] += ll_dc

            predictions.append({
                "date": m["date"],
                "season": s_year,
                "home": h, "away": a,
                "pregame_home_elo": round(elo[h], 1),
                "pregame_away_elo": round(elo[a], 1),
                "lam_home": round(lam_h, 2),
                "lam_away": round(lam_a, 2),
                "p_home": p_h, "p_draw": p_d, "p_away": p_a,
                "p_home_dc": p_h_dc, "p_draw_dc": p_d_dc, "p_away_dc": p_a_dc,
                "result": res,
                "hg": m["hg"], "ag": m["ag"],
            })

        elo[h], elo[a] = elo_update(elo[h], elo[a], m["res"], m["hg"], m["ag"],
                                     hfa=hfa_for(league_slug))
        gs_h = goal_stats.setdefault(h, {"gs": 0, "ga": 0, "matches": 0})
        gs_a = goal_stats.setdefault(a, {"gs": 0, "ga": 0, "matches": 0})
        gs_h["gs"] += m["hg"]; gs_h["ga"] += m["ag"]; gs_h["matches"] += 1
        gs_a["gs"] += m["ag"]; gs_a["ga"] += m["hg"]; gs_a["matches"] += 1

    for s, stat in per_season.items():
        if stat["n"] > 0:
            stat["accuracy"] = stat["correct"] / stat["n"]
            stat["log_loss"] = stat["ll"] / stat["n"]
            stat["brier"] = stat["brier"] / stat["n"]
            # DC parallel metrics for comparison on backtest page
            n_dc = stat.get("n_dc", 0)
            if n_dc:
                stat["accuracy_dc"] = stat["correct_dc"] / n_dc
                stat["log_loss_dc"] = stat["ll_dc"] / n_dc
        else:
            stat["accuracy"] = stat["log_loss"] = stat["brier"] = None

    return {
        "league_slug": league_slug,
        "league_name": league["name"],
        "first_season": first,
        "last_season": end_season,
        "num_seasons": num_seasons,
        "warmup_seasons": warmup_seasons,
        "scored_from_season": scored_from,
        "total_matches": len(matches),
        "scored_matches": len(predictions),
        "per_season": per_season,
        "predictions": predictions,
        "final_elo": {t: round(e, 1) for t, e in elo.items()},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {
            "K": K_FACTOR, "HFA": HFA, "DRAW_FACTOR": DRAW_FACTOR,
            "INITIAL_ELO": INITIAL_ELO,
        },
    }


def compute_metrics(predictions):
    if not predictions:
        return {"n": 0}
    n = len(predictions)
    ll = brier = 0.0
    correct = 0
    y_counts = {"H": 0, "D": 0, "A": 0}
    home_baseline_correct = 0

    for p in predictions:
        res = p["result"]
        y_h = 1.0 if res == "H" else 0.0
        y_d = 1.0 if res == "D" else 0.0
        y_a = 1.0 if res == "A" else 0.0
        p_h = max(1e-6, min(1 - 1e-6, p["p_home"]))
        p_d = max(1e-6, min(1 - 1e-6, p["p_draw"]))
        p_a = max(1e-6, min(1 - 1e-6, p["p_away"]))
        ll -= y_h * math.log(p_h) + y_d * math.log(p_d) + y_a * math.log(p_a)
        brier += ((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2) / 3
        pred = max(("H", p_h), ("D", p_d), ("A", p_a), key=lambda x: x[1])[0]
        if pred == res:
            correct += 1
        y_counts[res] += 1
        if res == "H":
            home_baseline_correct += 1

    # Home baseline uses empirical home-win rate as probability
    p_home_base = y_counts["H"] / n
    p_draw_base = y_counts["D"] / n
    p_away_base = y_counts["A"] / n
    home_ll = 0.0
    for p in predictions:
        res = p["result"]
        y_h = 1.0 if res == "H" else 0.0
        y_d = 1.0 if res == "D" else 0.0
        y_a = 1.0 if res == "A" else 0.0
        home_ll -= (y_h * math.log(max(1e-6, p_home_base))
                    + y_d * math.log(max(1e-6, p_draw_base))
                    + y_a * math.log(max(1e-6, p_away_base)))

    return {
        "n": n,
        "accuracy": correct / n,
        "log_loss": ll / n,
        "brier_score": brier / n,
        "home_baseline_accuracy": p_home_base,
        "home_baseline_log_loss": home_ll / n,
        "home_rate": p_home_base,
        "draw_rate": p_draw_base,
        "away_rate": p_away_base,
    }


# ============================================================================
# caching
# ============================================================================

_GOAL_RATES_CACHE = {}
_GOAL_RATES_TTL_S = 6 * 3600


def get_team_goal_rates(league_slug, as_of_date=None):
    """Return {team_name: {gs_per_match, ga_per_match, matches}} for the current
    season up to as_of_date (default: use whole current season).

    Used to estimate BTTS and over/under probabilities per matchup.
    """
    cache_key = (league_slug, as_of_date or "latest")
    now = time.time()
    hit = _GOAL_RATES_CACHE.get(cache_key)
    if hit and now - hit[1] < _GOAL_RATES_TTL_S:
        return hit[0]
    end_season = _pick_season()
    try:
        matches = load_league_matches(LEAGUES[league_slug], end_season, num_seasons=1)
    except Exception:
        matches = []
    out = {}
    for m in matches:
        if as_of_date and m["date"] >= as_of_date:
            continue
        for team, scored, allowed in [(m["home"], m["hg"], m["ag"]),
                                       (m["away"], m["ag"], m["hg"])]:
            e = out.setdefault(team, {"gs": 0, "ga": 0, "matches": 0})
            e["gs"] += scored
            e["ga"] += allowed
            e["matches"] += 1
    rates = {
        t: {
            "gs_per_match": s["gs"] / max(s["matches"], 1),
            "ga_per_match": s["ga"] / max(s["matches"], 1),
            "matches": s["matches"],
        }
        for t, s in out.items()
    }
    _GOAL_RATES_CACHE[cache_key] = (rates, now)
    return rates


# League average goals per team per match — fallback when a team has no data
_LEAGUE_GOAL_PRIORS = {
    "epl": 1.40, "laliga": 1.25, "ligamx": 1.45, "bundesliga": 1.50,
    "seriea": 1.35, "ligue1": 1.30,
}


# ============================================================================
# Dixon-Coles — joint-distribution model with low-score correction.
# ============================================================================
# Classical independent-Poisson under-predicts 0-0, 1-0, 0-1, 1-1. Dixon-Coles
# (1997) applies a correction factor τ to those four cells. Everything else
# uses the independent Poisson pmf. ρ is the correlation parameter — negative
# in practice (−0.1 to −0.2 for most leagues).

DC_RHO_DEFAULT = -0.15
DC_MAX_GOALS = 10


def _dc_tau(h, a, lam_h, lam_a, rho):
    """Low-score correction factor."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_matrix(lam_home, lam_away, rho=DC_RHO_DEFAULT, max_goals=DC_MAX_GOALS):
    """Return a (max_goals+1) × (max_goals+1) joint pmf matrix.

    Row index = home goals, column index = away goals. Rows/cols sum close
    to the Poisson marginals, with the four low-score cells adjusted and
    the whole matrix renormalized so entries sum to 1.
    """
    try:
        pmf_h = [math.exp(-lam_home) * (lam_home ** k) / math.factorial(k)
                 for k in range(max_goals + 1)]
        pmf_a = [math.exp(-lam_away) * (lam_away ** k) / math.factorial(k)
                 for k in range(max_goals + 1)]
    except (OverflowError, ValueError):
        # Fallback uniform if lambdas blow up
        pmf_h = [1.0 / (max_goals + 1)] * (max_goals + 1)
        pmf_a = pmf_h[:]
    mat = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            mat[h][a] = pmf_h[h] * pmf_a[a] * _dc_tau(h, a, lam_home, lam_away, rho)
    total = sum(sum(row) for row in mat)
    if total <= 0:
        return mat
    return [[c / total for c in row] for row in mat]


def _project_lambdas(home_team, away_team, league_slug):
    """Compute (λ_home, λ_away) from team historical goal rates with HFA.

    When xG per-match rates are available (soccer_xg snapshot), blend them
    60/40 with the realized goals rate: xG is sharper (removes finishing
    variance) but goals are the actual outcome and shouldn't be ignored.
    """
    rates = get_team_goal_rates(league_slug)
    prior = _LEAGUE_GOAL_PRIORS.get(league_slug, 1.35)
    default = {"gs_per_match": prior, "ga_per_match": prior, "matches": 0}
    hr = rates.get(home_team, default)
    ar = rates.get(away_team, default)

    # xG blend when snapshot has both teams
    try:
        import soccer_xg
        h_xg = soccer_xg.get_team_xg_rates(home_team, league_slug)
        a_xg = soccer_xg.get_team_xg_rates(away_team, league_slug)
    except Exception:
        h_xg = a_xg = None

    def _blend(goals_rate, xg_rate, weight_xg=0.6):
        if xg_rate is None:
            return goals_rate
        return weight_xg * xg_rate + (1 - weight_xg) * goals_rate

    h_gs = _blend(hr["gs_per_match"], (h_xg or {}).get("xg_per_match"))
    h_ga = _blend(hr["ga_per_match"], (h_xg or {}).get("xga_per_match"))
    a_gs = _blend(ar["gs_per_match"], (a_xg or {}).get("xg_per_match"))
    a_ga = _blend(ar["ga_per_match"], (a_xg or {}).get("xga_per_match"))

    lam_home = (h_gs + a_ga) / 2.0 * 1.15
    lam_away = (a_gs + h_ga) / 2.0 * 0.90
    return max(0.1, lam_home), max(0.1, lam_away)


def predict_3way_dc(home_team, away_team, league_slug, rho=DC_RHO_DEFAULT):
    """3-way outcome probabilities under Dixon-Coles.

    Returns (p_home, p_draw, p_away).
    """
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    p_home = sum(mat[h][a] for h in range(DC_MAX_GOALS + 1)
                             for a in range(h))
    p_draw = sum(mat[h][h] for h in range(DC_MAX_GOALS + 1))
    p_away = 1.0 - p_home - p_draw
    return p_home, p_draw, p_away


def predict_total_dc(home_team, away_team, league_slug, line,
                     rho=DC_RHO_DEFAULT):
    """Over/under total goals probability under Dixon-Coles."""
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    p_over = 0.0
    for h in range(DC_MAX_GOALS + 1):
        for a in range(DC_MAX_GOALS + 1):
            if h + a > line:
                p_over += mat[h][a]
    return {"over": p_over, "under": 1.0 - p_over,
            "expected_total": lam_h + lam_a}


def simulate_match(home_team, away_team, league_slug, n=10000,
                   rho=DC_RHO_DEFAULT, seed=None):
    """Monte Carlo simulate a match using the Dixon-Coles joint pmf.

    Returns a dict with:
        * trials, home_win_pct, draw_pct, away_win_pct
        * btts_pct (both teams score)
        * cs_pct (clean sheet probabilities per side)
        * total_buckets (histogram of total-goal lines)
        * avg_home_goals, avg_away_goals
        * expected_total (lam_home + lam_away)
        * most_likely_scores (top 10 scorelines by trial frequency)
    """
    import random
    rng = random.Random(seed) if seed is not None else random

    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)

    # Flatten joint pmf into a cumulative array for inverse-CDF sampling
    cells = []
    cum = 0.0
    for hh in range(DC_MAX_GOALS + 1):
        for aa in range(DC_MAX_GOALS + 1):
            p = mat[hh][aa]
            if p > 0:
                cum += p
                cells.append((cum, hh, aa))

    home_wins = draws = away_wins = 0
    btts_yes = 0
    cs_home = cs_away = 0  # home clean sheet / away clean sheet
    total_sum_h = total_sum_a = 0
    score_counts = {}
    buckets = {"over_1_5": 0, "over_2_5": 0, "over_3_5": 0,
               "under_2_5": 0, "under_3_5": 0}

    for _ in range(n):
        r = rng.random() * cum
        # Binary search would be nice; cells is up to 121 so linear is fine
        hh = aa = 0
        for c, h_g, a_g in cells:
            if r <= c:
                hh, aa = h_g, a_g
                break

        if hh > aa:
            home_wins += 1
        elif hh < aa:
            away_wins += 1
        else:
            draws += 1
        if hh >= 1 and aa >= 1:
            btts_yes += 1
        if aa == 0:
            cs_home += 1
        if hh == 0:
            cs_away += 1
        total_sum_h += hh
        total_sum_a += aa
        total = hh + aa
        if total > 1: buckets["over_1_5"] += 1
        if total > 2: buckets["over_2_5"] += 1
        if total > 3: buckets["over_3_5"] += 1
        if total < 3: buckets["under_2_5"] += 1
        if total < 4: buckets["under_3_5"] += 1

        score_key = f"{hh}-{aa}"
        score_counts[score_key] = score_counts.get(score_key, 0) + 1

    top_scores = sorted(score_counts.items(), key=lambda kv: -kv[1])[:10]
    top_scores = [{"score": k, "pct": v / n * 100} for k, v in top_scores]

    # Win-to-nil (team wins AND opponent fails to score). Derivable
    # analytically from the already-computed matrix for free; we track it
    # alongside BTTS so picks for either market can read from one sim.
    home_wn_mat_prob = sum(mat[hh][0] for hh in range(1, DC_MAX_GOALS + 1))
    away_wn_mat_prob = sum(mat[0][aa] for aa in range(1, DC_MAX_GOALS + 1))

    # Team-total marginals (useful for Over/Under 1.5 / 2.5 team-total picks)
    marg_home = [sum(mat[hh][aa] for aa in range(DC_MAX_GOALS + 1))
                 for hh in range(DC_MAX_GOALS + 1)]
    marg_away = [sum(mat[hh][aa] for hh in range(DC_MAX_GOALS + 1))
                 for aa in range(DC_MAX_GOALS + 1)]
    def _tt_over(marg, line):
        return sum(p for g, p in enumerate(marg) if g > line)

    return {
        "trials": n,
        "home_team": home_team,
        "away_team": away_team,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "home_win_pct": home_wins / n * 100,
        "draw_pct": draws / n * 100,
        "away_win_pct": away_wins / n * 100,
        "btts_yes_pct": btts_yes / n * 100,
        "btts_no_pct": (n - btts_yes) / n * 100,
        "cs_home_pct": cs_home / n * 100,
        "cs_away_pct": cs_away / n * 100,
        "home_wn_pct": home_wn_mat_prob * 100,
        "away_wn_pct": away_wn_mat_prob * 100,
        "home_tt_over_1_5": _tt_over(marg_home, 1.5) * 100,
        "home_tt_over_2_5": _tt_over(marg_home, 2.5) * 100,
        "away_tt_over_1_5": _tt_over(marg_away, 1.5) * 100,
        "away_tt_over_2_5": _tt_over(marg_away, 2.5) * 100,
        "avg_home_goals": total_sum_h / n,
        "avg_away_goals": total_sum_a / n,
        "expected_total": lam_h + lam_a,
        "totals": {k: v / n * 100 for k, v in buckets.items()},
        "most_likely_scores": top_scores,
    }


def predict_win_to_nil(home_team, away_team, league_slug, rho=DC_RHO_DEFAULT):
    """P(home wins AND keeps clean sheet) and P(away wins AND clean sheet).

    Derived directly from the Dixon-Coles joint pmf — no extra sim needed.
    These are strong +EV markets at Pinnacle when a dominant side meets
    a weak attack; most books overprice them because the "win AND clean
    sheet" conjunction feels less likely than it statistically is.
    """
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    home_wn = sum(mat[hh][0] for hh in range(1, DC_MAX_GOALS + 1))
    away_wn = sum(mat[0][aa] for aa in range(1, DC_MAX_GOALS + 1))
    return {
        "home_wn": home_wn,
        "away_wn": away_wn,
        "home_wn_no": 1.0 - home_wn,
        "away_wn_no": 1.0 - away_wn,
    }


def predict_team_total(home_team, away_team, league_slug, side, line,
                       rho=DC_RHO_DEFAULT):
    """P(side's goals over X / under X) using the DC marginal.

    side: "home" or "away"
    line: half-integer like 0.5, 1.5, 2.5 (whole numbers can push)
    """
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    # Marginal for the chosen side
    if side == "home":
        marg = [sum(mat[hh][aa] for aa in range(DC_MAX_GOALS + 1))
                for hh in range(DC_MAX_GOALS + 1)]
    else:
        marg = [sum(mat[hh][aa] for hh in range(DC_MAX_GOALS + 1))
                for aa in range(DC_MAX_GOALS + 1)]
    over = sum(p for g, p in enumerate(marg) if g > line)
    return {"over": over, "under": 1.0 - over,
            "expected": lam_h if side == "home" else lam_a}


def predict_correct_score(home_team, away_team, league_slug,
                          rho=DC_RHO_DEFAULT, top_n=10):
    """Top-N most-likely exact scorelines with probabilities."""
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    scores = []
    for hh in range(DC_MAX_GOALS + 1):
        for aa in range(DC_MAX_GOALS + 1):
            scores.append((hh, aa, mat[hh][aa]))
    scores.sort(key=lambda s: -s[2])
    return [{"score": f"{h}-{a}", "prob": p} for h, a, p in scores[:top_n]]


def predict_btts(home_team, away_team, league_slug, rho=DC_RHO_DEFAULT):
    """BTTS probability (Dixon-Coles version).

    P(BTTS Yes) = sum over (h ≥ 1 and a ≥ 1) of joint pmf. More accurate than
    the independent-Poisson identity because of the low-score correction.
    """
    lam_h, lam_a = _project_lambdas(home_team, away_team, league_slug)
    mat = dixon_coles_matrix(lam_h, lam_a, rho)
    p_yes = sum(mat[h][a] for h in range(1, DC_MAX_GOALS + 1)
                             for a in range(1, DC_MAX_GOALS + 1))
    return {"yes": p_yes, "no": 1.0 - p_yes}


def _pick_season():
    """Current soccer "season" year (year that this season started)."""
    today = date.today()
    # Most European leagues start Aug and end May. If month >= 7, season starts this year.
    if today.month >= 7:
        return today.year
    return today.year - 1


def elo_only_cache_file(slug):
    """Tiny side-file holding just final_elo — safe to load from constrained
    hosts (Render free tier) without pulling the full ~1-5MB predictions list."""
    return os.path.join(CACHE_DIR, f"soccer_{slug}_final_elo.json")


def _write_elo_only_cache(league_slug, result):
    """Write a slim JSON next to the full backtest cache for lightweight lookups."""
    try:
        with open(elo_only_cache_file(league_slug), "w") as f:
            json.dump({
                "final_elo":    result.get("final_elo", {}),
                "generated_at": result.get("generated_at"),
                "league_slug":  league_slug,
            }, f)
    except OSError:
        pass


def get_final_elo(league_slug):
    """Lightweight final_elo lookup. Reads a tiny side-file instead of the
    full backtest JSON (which can be several MB). Falls back to deriving the
    side-file from the full cache, else runs the backtest (expensive).
    """
    slim = elo_only_cache_file(league_slug)
    if os.path.exists(slim):
        try:
            with open(slim) as f:
                return (json.load(f) or {}).get("final_elo", {})
        except (OSError, json.JSONDecodeError):
            pass
    full = cache_file(league_slug)
    if os.path.exists(full):
        try:
            with open(full) as f:
                data = json.load(f)
            elo = data.get("final_elo", {}) or {}
            _write_elo_only_cache(league_slug, data)
            import gc
            del data
            gc.collect()
            return elo
        except (OSError, json.JSONDecodeError):
            pass
    # Last resort: run the backtest
    state = get_or_run_backtest(league_slug)
    return (state or {}).get("final_elo", {})


def get_or_run_backtest(league_slug, refresh=False, verbose=False):
    path = cache_file(league_slug)
    if not refresh and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            gen = datetime.fromisoformat(data.get("generated_at", "1970-01-01"))
            if (datetime.now() - gen) < timedelta(hours=CACHE_MAX_AGE_H):
                return data
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    end_season = _pick_season()
    result = run_multi_season_backtest(league_slug, end_season, verbose=verbose)
    result["metrics"] = compute_metrics(result["predictions"])
    with open(path, "w") as f:
        json.dump(result, f)
    _write_elo_only_cache(league_slug, result)
    return result


# ============================================================================
# CLI
# ============================================================================

def _print_report(result):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    m = result["metrics"]
    print()
    print(f"=== Soccer backtest: {result['league_name']} ===")
    print(f"Window        : {result['first_season']}/{result['first_season']+1} .. "
          f"{result['last_season']}/{result['last_season']+1}")
    print(f"Total matches : {result['total_matches']}")
    print(f"Scored        : {result['scored_matches']} "
          f"(from {result['scored_from_season']}/{result['scored_from_season']+1})")
    print()
    print(f"Model accuracy  : {m['accuracy']:.4f}")
    print(f"Model log loss  : {m['log_loss']:.4f}")
    print(f"Model Brier     : {m['brier_score']:.4f}")
    print()
    print(f"Empirical rates : H={m['home_rate']:.3f} D={m['draw_rate']:.3f} A={m['away_rate']:.3f}")
    print(f"Home baseline   : acc={m['home_baseline_accuracy']:.4f} ll={m['home_baseline_log_loss']:.4f}")
    print()
    print("Per-season:")
    for s in sorted(result["per_season"]):
        st = result["per_season"][s]
        if st["n"]:
            print(f"  {s}/{s+1}: n={st['n']:3d}  acc={st['accuracy']:.4f}  "
                  f"ll={st['log_loss']:.4f}  brier={st['brier']:.4f}")

    # Top 10 teams by Elo
    ranked = sorted(result["final_elo"].items(), key=lambda kv: -kv[1])[:10]
    print()
    print("Top 10 final Elo:")
    for t, e in ranked:
        print(f"  {e:7.1f}  {t}")


def main():
    parser = argparse.ArgumentParser(description="Soccer league backtest")
    parser.add_argument("--league", default="epl", choices=list(LEAGUES),
                        help="League slug (epl, laliga, bundesliga, seriea, ligue1, ligamx)")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = get_or_run_backtest(args.league, refresh=args.refresh, verbose=True)
    _print_report(result)


if __name__ == "__main__":
    main()
