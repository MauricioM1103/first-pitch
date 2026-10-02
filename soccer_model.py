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
HFA = 100.0           # soccer home advantage ≈ 100 Elo (~2 goals Elo-equivalent)
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

def _p_home_not_lose(home_elo, away_elo):
    diff = (home_elo + HFA) - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def predict_3way(home_elo, away_elo):
    """Return (p_home, p_draw, p_away) summing to 1."""
    p_hw = _p_home_not_lose(home_elo, away_elo)
    # Draw probability peaks at p_hw = 0.5, drops to zero at extremes
    p_draw = DRAW_FACTOR * (1.0 - abs(2.0 * p_hw - 1.0))
    p_home = p_hw - p_draw / 2.0
    p_away = 1.0 - p_hw - p_draw / 2.0
    # Clip and normalize
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


def elo_update(home_elo, away_elo, result, hg, ag, k=K_FACTOR):
    """Update ratings based on actual result (H/D/A) with MOV multiplier."""
    y = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
    exp_h = _p_home_not_lose(home_elo, away_elo)
    # Treat draw as y=0.5 in the win-prob frame
    mov = _mov_multiplier(hg - ag)
    delta = k * mov * (y - exp_h)
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
    predictions = []
    per_season = {s: {"n": 0, "correct": 0, "ll": 0.0, "brier": 0.0}
                  for s in seasons[warmup_seasons:]}
    seasons_seen = set()

    for m in matches:
        s_year = m["season"]
        # Season-boundary regression
        if s_year not in seasons_seen and seasons_seen:
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        seasons_seen.add(s_year)

        h, a = m["home"], m["away"]
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)

        p_h, p_d, p_a = predict_3way(elo[h], elo[a])

        if s_year >= scored_from:
            # 3-way log loss + Brier
            res = m["res"]
            y_h = 1.0 if res == "H" else 0.0
            y_d = 1.0 if res == "D" else 0.0
            y_a = 1.0 if res == "A" else 0.0

            p_hh = max(1e-6, min(1 - 1e-6, p_h))
            p_dd = max(1e-6, min(1 - 1e-6, p_d))
            p_aa = max(1e-6, min(1 - 1e-6, p_a))

            ll = -(y_h * math.log(p_hh) + y_d * math.log(p_dd) + y_a * math.log(p_aa))
            brier = ((p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2) / 3

            # Accuracy: argmax prediction matches actual
            pred_pick = max(("H", p_h), ("D", p_d), ("A", p_a), key=lambda x: x[1])[0]
            correct = (pred_pick == res)

            per_season[s_year]["n"] += 1
            per_season[s_year]["correct"] += 1 if correct else 0
            per_season[s_year]["ll"] += ll
            per_season[s_year]["brier"] += brier

            predictions.append({
                "date": m["date"],
                "season": s_year,
                "home": h, "away": a,
                "pregame_home_elo": round(elo[h], 1),
                "pregame_away_elo": round(elo[a], 1),
                "p_home": p_h, "p_draw": p_d, "p_away": p_a,
                "result": res,
                "hg": m["hg"], "ag": m["ag"],
            })

        elo[h], elo[a] = elo_update(elo[h], elo[a], m["res"], m["hg"], m["ag"])

    for s, stat in per_season.items():
        if stat["n"] > 0:
            stat["accuracy"] = stat["correct"] / stat["n"]
            stat["log_loss"] = stat["ll"] / stat["n"]
            stat["brier"] = stat["brier"] / stat["n"]
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


def predict_btts(home_team, away_team, league_slug):
    """Probability that both teams score at least once.

    Blends team-level attack rate with opponent defense rate to project
    expected goals per side, then applies the independent-Poisson identity:
        P(team scores ≥ 1) = 1 − exp(−λ)
        P(BTTS Yes) = P(home scores) × P(away scores)
    Home teams get a +15% attack boost, away a −10% penalty (standard HFA).
    """
    rates = get_team_goal_rates(league_slug)
    prior = _LEAGUE_GOAL_PRIORS.get(league_slug, 1.35)
    default = {"gs_per_match": prior, "ga_per_match": prior, "matches": 0}
    hr = rates.get(home_team, default)
    ar = rates.get(away_team, default)
    # Blend own attack with opponent defense
    lam_home = (hr["gs_per_match"] + ar["ga_per_match"]) / 2.0 * 1.15
    lam_away = (ar["gs_per_match"] + hr["ga_per_match"]) / 2.0 * 0.90
    p_home_scores = 1.0 - math.exp(-lam_home)
    p_away_scores = 1.0 - math.exp(-lam_away)
    p_yes = p_home_scores * p_away_scores
    return {"yes": p_yes, "no": 1.0 - p_yes}


def _pick_season():
    """Current soccer "season" year (year that this season started)."""
    today = date.today()
    # Most European leagues start Aug and end May. If month >= 7, season starts this year.
    if today.month >= 7:
        return today.year
    return today.year - 1


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
