#!/usr/bin/env python3
"""MLB win-probability model — Elo + starting-pitcher adjustment.

Walk-forward backtest on a full regular season:
  * Elo ratings warm up across the whole season, updated after every game.
  * For each game, prediction is made BEFORE the game using stats that were
    available prior to first pitch (pitcher season-to-date, Elo pre-game).
  * The last N days of the season are scored against actual outcomes.

Data source: MLB statsapi. No API keys, no scraping HTML.

Run standalone:
    python mlb_model.py              # cached (rerun once/day)
    python mlb_model.py --refresh    # force refit
    python mlb_model.py --score-days 60 --season 2025

Programmatic:
    from mlb_model import get_or_run_backtest, predict_win_prob, load_pitcher_stats_asof
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

STATSAPI = "https://statsapi.mlb.com/api/v1"
SCHEDULE_URL = (
    STATSAPI + "/schedule?sportId=1&startDate={start}&endDate={end}"
    "&hydrate=probablePitcher"
)
PITCHER_LOG_URL = (
    STATSAPI + "/people/{pid}/stats?stats=gameLog&season={season}"
    "&group=pitching&sportId=1"
)

# Cache next to the source file so it ships with the repo (Linux hosts have
# a different ~ than Windows, and Render's filesystem is ephemeral otherwise).
CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mlb_backtest_cache.json",
)

# ---- model hyperparameters ---------------------------------------------
INITIAL_ELO = 1500.0
K_FACTOR = 5.0
HFA = 24.0                # home-field advantage, Elo points (~54% base)
LEAGUE_ERA = 4.20
LEAGUE_WHIP = 1.30
ERA_WEIGHT = 22.0         # Elo points per 1 ERA below league avg
WHIP_WEIGHT = 40.0        # Elo points per 1 WHIP below league avg
IP_FULL = 40.0            # innings for full workload weight
CACHE_MAX_AGE_H = 24


# ============================================================================
# fetch primitives
# ============================================================================

def _fetch(url):
    req = Request(url, headers={"User-Agent": "mlb-model/1.0"})
    with urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _parse_ip(ip_str):
    """MLB writes IP as '5.1' meaning 5 1/3 innings — decimal is thirds."""
    s = str(ip_str)
    if "." in s:
        whole, frac = s.split(".")
        try:
            return int(whole) + int(frac) / 3.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ============================================================================
# season & pitcher data loaders
# ============================================================================

def load_season_games(season):
    """Return sorted list of completed regular-season games for `season`."""
    d = _fetch(SCHEDULE_URL.format(start=f"{season}-01-01", end=f"{season}-12-31"))
    games = []
    for db in d.get("dates", []):
        for g in db.get("games", []):
            if g.get("gameType") != "R":
                continue
            status = (g.get("status") or {}).get("codedGameState")
            if status != "F":
                continue
            home = g.get("teams", {}).get("home", {}) or {}
            away = g.get("teams", {}).get("away", {}) or {}
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            if home.get("score") is None or away.get("score") is None:
                continue
            games.append({
                "date": g.get("officialDate"),
                "gamePk": g.get("gamePk"),
                "home_id": (home.get("team") or {}).get("id"),
                "away_id": (away.get("team") or {}).get("id"),
                "home_name": (home.get("team") or {}).get("name"),
                "away_name": (away.get("team") or {}).get("name"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_won": bool(home.get("isWinner")),
                "home_sp_id": hp.get("id"),
                "away_sp_id": ap.get("id"),
                "home_sp_name": hp.get("fullName"),
                "away_sp_name": ap.get("fullName"),
            })
    games.sort(key=lambda g: (g["date"], g["gamePk"]))
    return games


def load_pitcher_gamelogs(pitcher_ids, season, workers=16):
    """Fetch each pitcher's per-outing log in parallel.

    Returns {pitcher_id: [outing, ...]} where each outing has date, er, ip, so, bb, h.
    """
    def _one(pid):
        try:
            d = _fetch(PITCHER_LOG_URL.format(pid=pid, season=season))
        except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
            return pid, []
        outings = []
        for s in (d.get("stats") or []):
            for split in (s.get("splits") or []):
                stat = split.get("stat") or {}
                dt = split.get("date")
                if not dt:
                    continue
                outings.append({
                    "date": dt,
                    "er": stat.get("earnedRuns") or 0,
                    "ip": _parse_ip(stat.get("inningsPitched", 0)),
                    "so": stat.get("strikeOuts") or 0,
                    "bb": stat.get("baseOnBalls") or 0,
                    "h": stat.get("hits") or 0,
                })
        outings.sort(key=lambda o: o["date"])
        return pid, outings

    out = {}
    ids = [p for p in pitcher_ids if p]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, pid) for pid in ids]
        for f in as_completed(futs):
            pid, outings = f.result()
            out[pid] = outings
    return out


def load_pitcher_stats_asof(pid, season, cutoff_date):
    """Convenience: fetch one pitcher's log (cached upstream ideally) and slice.

    Used by the UI at prediction time for today's starters.
    """
    if not pid:
        return None
    try:
        d = _fetch(PITCHER_LOG_URL.format(pid=pid, season=season))
    except Exception:
        return None
    outings = []
    for s in (d.get("stats") or []):
        for split in (s.get("splits") or []):
            stat = split.get("stat") or {}
            dt = split.get("date")
            if not dt:
                continue
            outings.append({
                "date": dt,
                "er": stat.get("earnedRuns") or 0,
                "ip": _parse_ip(stat.get("inningsPitched", 0)),
                "so": stat.get("strikeOuts") or 0,
                "bb": stat.get("baseOnBalls") or 0,
                "h": stat.get("hits") or 0,
            })
    return pitcher_stats_asof(outings, cutoff_date)


# ============================================================================
# as-of pitcher aggregation (no look-ahead)
# ============================================================================

def pitcher_stats_asof(outings, cutoff_date):
    """Aggregate all outings STRICTLY BEFORE cutoff_date into rate stats.

    Returns dict with ip, era, whip, k9, bb9. era/whip/etc are None if no IP.
    """
    er = ip = so = bb = h = 0
    for o in outings:
        if o["date"] < cutoff_date:
            er += o["er"]
            ip += o["ip"]
            so += o["so"]
            bb += o["bb"]
            h += o["h"]
    if ip <= 0:
        return {"ip": 0.0, "era": None, "whip": None, "k9": None, "bb9": None}
    return {
        "ip": ip,
        "era": er * 9.0 / ip,
        "whip": (h + bb) / ip,
        "k9": so * 9.0 / ip,
        "bb9": bb * 9.0 / ip,
    }


# ============================================================================
# Elo + prediction
# ============================================================================

def expected_prob(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier(margin, winner_elo_diff):
    """FiveThirtyEight-style MOV multiplier; dampens updates for expected blowouts."""
    m = abs(margin) if margin else 1
    return math.log(m + 1) * (2.2 / (max(0.0, winner_elo_diff) * 0.001 + 2.2))


def sp_boost(sp_stats):
    """Elo-point adjustment for the starter's season-to-date rate stats."""
    if not sp_stats or (sp_stats.get("ip") or 0) <= 0:
        return 0.0
    weight = min(1.0, sp_stats["ip"] / IP_FULL)
    era_c = (LEAGUE_ERA - sp_stats["era"]) * ERA_WEIGHT
    whip_c = (LEAGUE_WHIP - sp_stats["whip"]) * WHIP_WEIGHT
    return weight * (era_c + whip_c) / 2.0


def predict_win_prob(home_elo, away_elo, home_sp_stats, away_sp_stats):
    """Return P(home team wins). Uses Elo + HFA + SP adjustment."""
    h_adj = home_elo + HFA + sp_boost(home_sp_stats)
    a_adj = away_elo + sp_boost(away_sp_stats)
    return expected_prob(h_adj, a_adj)


def elo_update(elo_home, elo_away, home_won, home_score, away_score, k=K_FACTOR):
    exp_h = expected_prob(elo_home + HFA, elo_away)
    margin = (home_score or 0) - (away_score or 0)
    if home_won:
        winner_diff = (elo_home + HFA) - elo_away
    else:
        winner_diff = elo_away - (elo_home + HFA)
    mov = _mov_multiplier(margin, winner_diff)
    delta = k * mov * ((1 if home_won else 0) - exp_h)
    return elo_home + delta, elo_away - delta


# ============================================================================
# Totals / Run-Line / F5 scoring model (Poisson)
# ============================================================================
#
# Project runs scored per side from team offense, opponent starting pitcher,
# and opponent team pitching (bullpen proxy). Then compute market probabilities
# from independent Poisson distributions over each side's runs.
#
# Poisson is slightly under-dispersed vs real MLB (clumping in innings), but
# as a bettor-grade first-pass model it's defensible, and it's analytical so
# no simulation noise.

LEAGUE_RPG = 4.50              # team runs per full game, league-wide
LEAGUE_F5_PER_TEAM = 2.20      # team runs per first 5 innings
MAX_RUNS_FULL = 25             # Poisson summation cap, full game
MAX_RUNS_F5 = 15               # Poisson summation cap, F5


def _sp_blend(sp_era, sp_ip, team_era):
    """Blend SP ERA with team pitching staff ERA by SP's workload."""
    if sp_era is None:
        return team_era
    weight = min(1.0, (sp_ip or 0) / 40.0)
    return weight * sp_era + (1 - weight) * (team_era if team_era is not None else sp_era)


def project_runs(team_rpg, opp_sp_era, opp_sp_ip, opp_team_era, scope="full"):
    """Expected runs for a team.

    scope: 'full' (whole game, SP+bullpen blend) or 'f5' (SP dominates).
    Returns None if inputs are insufficient.
    """
    if team_rpg is None or opp_team_era is None:
        return None
    if scope == "f5":
        # F5 is almost entirely SP's innings; blend heavily toward SP
        pitcher_rate = _sp_blend(opp_sp_era, opp_sp_ip, opp_team_era)
        base = LEAGUE_F5_PER_TEAM
    else:
        # Full game: SP ~5-6 IP, bullpen ~3-4 IP, 60/40 blend
        sp_rate = _sp_blend(opp_sp_era, opp_sp_ip, opp_team_era)
        pitcher_rate = 0.60 * sp_rate + 0.40 * opp_team_era
        base = LEAGUE_RPG
    offense_factor = team_rpg / LEAGUE_RPG
    pitching_factor = pitcher_rate / LEAGUE_ERA
    return base * offense_factor * pitching_factor


def _poisson_pmf(k, lam):
    if lam is None or lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def _poisson_cdf_array(lam, max_k):
    """Return [pmf(0), pmf(1), ..., pmf(max_k)]."""
    return [_poisson_pmf(k, lam) for k in range(max_k + 1)]


def _joint_summary(lam_home, lam_away, max_k):
    """Compute joint distribution summaries needed for market probabilities.

    Returns dict with:
      p_total_over[t]       = P(home + away > t)       for t in 0..2*max_k
      p_margin_ge[m + max_k] = P(home - away >= m)     for m in -max_k..max_k
    """
    pmf_h = _poisson_cdf_array(lam_home, max_k)
    pmf_a = _poisson_cdf_array(lam_away, max_k)
    N = max_k + 1

    # P(home - away = m) for m in -max_k..max_k
    diff_pmf = [0.0] * (2 * N - 1)  # index 0 = margin -max_k, index N-1 = 0
    # P(home + away = s) for s in 0..2*max_k
    sum_pmf = [0.0] * (2 * max_k + 1)

    for h in range(N):
        ph = pmf_h[h]
        if ph == 0:
            continue
        for a in range(N):
            pa = pmf_a[a]
            if pa == 0:
                continue
            joint = ph * pa
            diff_pmf[h - a + max_k] += joint
            sum_pmf[h + a] += joint

    # Build P(home + away > t)
    cum_from_bottom = 0.0
    p_total_over = [0.0] * (2 * max_k + 1)
    for t in range(2 * max_k, -1, -1):
        cum_from_bottom += sum_pmf[t]
        # P(total > t) = sum of pmf[t+1..]
    # Redo correctly:
    p_total_over = [0.0] * (2 * max_k + 1)
    running = 0.0
    for t in range(2 * max_k, -1, -1):
        p_total_over[t] = running
        running += sum_pmf[t]

    # P(home margin >= m)
    p_margin_ge = [0.0] * (2 * N - 1)
    running = 0.0
    for i in range(2 * N - 2, -1, -1):
        running += diff_pmf[i]
        p_margin_ge[i] = running

    return {
        "p_total_over": p_total_over,
        "p_margin_ge": p_margin_ge,
        "sum_pmf": sum_pmf,
        "diff_pmf": diff_pmf,
        "max_k": max_k,
    }


def prob_total_over(line, lam_home, lam_away, scope="full"):
    """P(home_runs + away_runs > line). Handles X.5 and X.0 lines."""
    max_k = MAX_RUNS_FULL if scope == "full" else MAX_RUNS_F5
    j = _joint_summary(lam_home, lam_away, max_k)
    # For a .5 line (e.g. 8.5), P(total > 8.5) = P(total >= 9) = p_total_over[8]
    # For a .0 line (e.g. 8), P(total > 8) = p_total_over[8] excluding the push.
    floor = int(math.floor(line))
    if line - floor >= 0.5:
        return j["p_total_over"][floor]
    # X.0 line: exclude pushes (bet is refunded, so EV uses p_win conditional)
    return j["p_total_over"][floor]


def prob_total_push(line, lam_home, lam_away, scope="full"):
    """P(push on total). Non-zero only when line is a whole number."""
    if line - math.floor(line) >= 0.5:
        return 0.0
    max_k = MAX_RUNS_FULL if scope == "full" else MAX_RUNS_F5
    j = _joint_summary(lam_home, lam_away, max_k)
    return j["sum_pmf"][int(line)]


def prob_home_margin_ge(k, lam_home, lam_away, scope="full"):
    """P(home_runs - away_runs >= k)."""
    max_k = MAX_RUNS_FULL if scope == "full" else MAX_RUNS_F5
    j = _joint_summary(lam_home, lam_away, max_k)
    idx = k + max_k
    if idx < 0:
        return 1.0
    if idx > 2 * max_k:
        return 0.0
    return j["p_margin_ge"][idx]


def predict_nrfi(game_ctx):
    """P(No Runs First Inning) + P(Yes Runs First Inning) for a game.

    1st-inning scoring runs about 12% higher per inning than other innings
    because the leadoff hitter + top of the order faces the SP fresh. We
    scale each team's full-game expected runs by that ratio: 1st-inning
    lambda ≈ full_game_lambda × (0.12 / 9) roughly ≈ 11.5% of the per-game
    run expectation, which lines up with the empirical ~0.52 R/team/1st.

    P(NRFI) = P(home scores 0 in B1) × P(away scores 0 in T1)
            = exp(-λ_home_1st) × exp(-λ_away_1st)
    """
    lam_h_full = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                              game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                              scope="full")
    lam_a_full = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                              game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                              scope="full")
    if lam_h_full is None or lam_a_full is None:
        return None
    # First-inning rate: slightly over 1/9 of the full-game rate (leadoff
    # bump). 0.123 is calibrated against ~0.52 R/team per 1st on 4.25 RPG.
    FIRST_INNING_FACTOR = 0.123
    lam_h_1 = lam_h_full * FIRST_INNING_FACTOR
    lam_a_1 = lam_a_full * FIRST_INNING_FACTOR
    p_h_0 = math.exp(-lam_h_1)
    p_a_0 = math.exp(-lam_a_1)
    p_nrfi = p_h_0 * p_a_0
    return {
        "lambda_home_1st": lam_h_1,
        "lambda_away_1st": lam_a_1,
        "p_nrfi": p_nrfi,
        "p_yrfi": 1.0 - p_nrfi,
    }


def predict_full_total(game_ctx, line):
    """Return dict with lambda_home, lambda_away, expected_total, p_over, p_under."""
    lam_h = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                         game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                         scope="full")
    lam_a = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                         game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                         scope="full")
    if lam_h is None or lam_a is None or line is None:
        return None
    p_over = prob_total_over(line, lam_h, lam_a, "full")
    p_push = prob_total_push(line, lam_h, lam_a, "full")
    return {
        "lambda_home": lam_h, "lambda_away": lam_a,
        "expected_total": lam_h + lam_a,
        "p_over": p_over, "p_under": 1 - p_over - p_push,
        "p_push": p_push,
    }


def predict_f5_total(game_ctx, line):
    lam_h = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                         game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                         scope="f5")
    lam_a = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                         game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                         scope="f5")
    if lam_h is None or lam_a is None or line is None:
        return None
    p_over = prob_total_over(line, lam_h, lam_a, "f5")
    p_push = prob_total_push(line, lam_h, lam_a, "f5")
    return {
        "lambda_home": lam_h, "lambda_away": lam_a,
        "expected_total": lam_h + lam_a,
        "p_over": p_over, "p_under": 1 - p_over - p_push,
        "p_push": p_push,
    }


def predict_run_line(game_ctx, line=1.5):
    """Return dict with p_home_covers (home -1.5), p_away_covers (away +1.5).
    Since 1.5 is non-integer, there is no push. Home covers iff margin >= 2."""
    lam_h = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                         game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                         scope="full")
    lam_a = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                         game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                         scope="full")
    if lam_h is None or lam_a is None:
        return None
    # home -1.5 wins if home_runs - away_runs >= 2
    threshold = int(math.ceil(line + 0.5))  # 1.5 -> 2
    p_home_covers = prob_home_margin_ge(threshold, lam_h, lam_a, "full")
    return {
        "lambda_home": lam_h, "lambda_away": lam_a,
        "line_home": -line, "line_away": +line,
        "p_home_covers": p_home_covers,
        "p_away_covers": 1 - p_home_covers,
    }


def simulate_game(lam_home, lam_away, n_sims=10000, seed=None):
    """Monte Carlo simulate N games using independent Poisson scoring.

    Returns a dict with win probs, run distributions, and market-ready
    probabilities for totals and the run line. Pure stdlib — no numpy.
    """
    import random
    rng = random.Random(seed) if seed is not None else random

    def _pois(lam):
        """Knuth's algorithm for Poisson(λ) sample. Fast for small λ."""
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return k - 1

    home_scores = []
    away_scores = []
    totals = []
    margins = []
    home_wins = away_wins = 0
    extras = 0
    for _ in range(n_sims):
        h = _pois(lam_home)
        a = _pois(lam_away)
        if h == a:
            # Extra innings — resolve coin flip with small home advantage
            extras += 1
            if rng.random() < 0.53:
                h += 1
            else:
                a += 1
        if h > a:
            home_wins += 1
        else:
            away_wins += 1
        home_scores.append(h)
        away_scores.append(a)
        totals.append(h + a)
        margins.append(h - a)

    totals_sorted = sorted(totals)
    margins_sorted = sorted(margins)

    def _p_over(line):
        """P(total > line). line can be X.5 or X.0."""
        # count sims where total > line (strict)
        hi_count = sum(1 for t in totals if t > line)
        return hi_count / n_sims

    def _p_margin_ge(k):
        return sum(1 for m in margins if m >= k) / n_sims

    return {
        "n_sims": n_sims,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "p_home": home_wins / n_sims,
        "p_away": away_wins / n_sims,
        "mean_total": sum(totals) / n_sims,
        "median_total": totals_sorted[n_sims // 2],
        "mean_margin": sum(margins) / n_sims,
        "median_margin": margins_sorted[n_sims // 2],
        "extras_pct": extras / n_sims,
        "p_over_fn": _p_over,
        "p_margin_ge_fn": _p_margin_ge,
        # histograms (binned later in the UI)
        "totals": totals,
        "margins": margins,
    }


def simulate_from_game_ctx(game_ctx, n_sims=10000, scope="full", seed=None):
    """Convenience: project λ from team stats, then simulate."""
    lam_h = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                         game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                         scope=scope)
    lam_a = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                         game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                         scope=scope)
    if lam_h is None or lam_a is None:
        return None
    return simulate_game(lam_h, lam_a, n_sims=n_sims, seed=seed)


def predict_f5_moneyline(game_ctx):
    """F5 can tie — return p_home / p_away / p_push for 2-way-with-tie-push markets."""
    lam_h = project_runs(game_ctx.get("home_rpg"), game_ctx.get("away_sp_era"),
                         game_ctx.get("away_sp_ip"), game_ctx.get("away_team_era"),
                         scope="f5")
    lam_a = project_runs(game_ctx.get("away_rpg"), game_ctx.get("home_sp_era"),
                         game_ctx.get("home_sp_ip"), game_ctx.get("home_team_era"),
                         scope="f5")
    if lam_h is None or lam_a is None:
        return None
    p_home = prob_home_margin_ge(1, lam_h, lam_a, "f5")  # margin >= 1
    p_away = prob_home_margin_ge(-100, lam_h, lam_a, "f5") - prob_home_margin_ge(0, lam_h, lam_a, "f5")
    # Clean up: p_away = P(margin < 0) = 1 - P(margin >= 0) = 1 - p_home - p_tie
    p_tie_or_home = prob_home_margin_ge(0, lam_h, lam_a, "f5")
    p_away = 1 - p_tie_or_home
    p_tie = p_tie_or_home - p_home
    # numerical cleanup
    p_tie = max(0.0, min(1.0, p_tie))
    return {
        "lambda_home": lam_h, "lambda_away": lam_a,
        "p_home": p_home, "p_away": p_away, "p_tie": p_tie,
    }


# ============================================================================
# backtest
# ============================================================================

def run_backtest(season, score_last_n_days=60, verbose=True):
    if verbose:
        print(f"[model] loading {season} regular-season schedule...", flush=True)
    games = load_season_games(season)
    if not games:
        raise RuntimeError(f"no completed regular-season games found for {season}")
    if verbose:
        print(f"[model]   {len(games)} completed games", flush=True)

    pitcher_ids = set()
    for g in games:
        if g["home_sp_id"]:
            pitcher_ids.add(g["home_sp_id"])
        if g["away_sp_id"]:
            pitcher_ids.add(g["away_sp_id"])
    if verbose:
        print(f"[model] fetching gameLogs for {len(pitcher_ids)} pitchers...", flush=True)

    t0 = time.time()
    logs = load_pitcher_gamelogs(pitcher_ids, season)
    if verbose:
        print(f"[model]   done in {time.time() - t0:.1f}s", flush=True)

    end_date = games[-1]["date"]
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    score_from = (end_dt - timedelta(days=score_last_n_days)).isoformat()

    elo = {}
    predictions = []
    win_tally = {}  # team_id -> (wins, losses) running

    for g in games:
        h = g["home_id"]
        a = g["away_id"]
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)
        win_tally.setdefault(h, [0, 0])
        win_tally.setdefault(a, [0, 0])

        # === pregame features (no look-ahead) ===
        home_sp = pitcher_stats_asof(logs.get(g["home_sp_id"], []), g["date"]) if g["home_sp_id"] else None
        away_sp = pitcher_stats_asof(logs.get(g["away_sp_id"], []), g["date"]) if g["away_sp_id"] else None
        pregame_home_wl = tuple(win_tally[h])
        pregame_away_wl = tuple(win_tally[a])

        p_home = predict_win_prob(elo[h], elo[a], home_sp, away_sp)

        if g["date"] >= score_from:
            predictions.append({
                "date": g["date"],
                "gamePk": g["gamePk"],
                "home": g["home_name"], "away": g["away_name"],
                "home_id": h, "away_id": a,
                "home_sp": g["home_sp_name"], "away_sp": g["away_sp_name"],
                "home_sp_era": home_sp["era"] if home_sp else None,
                "away_sp_era": away_sp["era"] if away_sp else None,
                "home_sp_ip": home_sp["ip"] if home_sp else None,
                "away_sp_ip": away_sp["ip"] if away_sp else None,
                "pregame_home_wl": list(pregame_home_wl),
                "pregame_away_wl": list(pregame_away_wl),
                "pregame_home_elo": round(elo[h], 1),
                "pregame_away_elo": round(elo[a], 1),
                "p_home": p_home,
                "home_won": g["home_won"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
            })

        # === update state AFTER the prediction has been logged ===
        elo[h], elo[a] = elo_update(
            elo[h], elo[a], g["home_won"], g["home_score"], g["away_score"]
        )
        if g["home_won"]:
            win_tally[h][0] += 1
            win_tally[a][1] += 1
        else:
            win_tally[h][1] += 1
            win_tally[a][0] += 1

    # nice final Elo list sorted for the report
    team_names = {}
    for g in games:
        team_names[g["home_id"]] = g["home_name"]
        team_names[g["away_id"]] = g["away_name"]

    return {
        "season": season,
        "score_from": score_from,
        "score_to": end_date,
        "score_last_n_days": score_last_n_days,
        "total_games": len(games),
        "scored_games": len(predictions),
        "predictions": predictions,
        "final_elo": {str(tid): round(elo[tid], 1) for tid in elo},
        "final_wl": {str(tid): win_tally[tid] for tid in win_tally},
        "team_names": {str(tid): team_names.get(tid, "") for tid in team_names},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {
            "K": K_FACTOR, "HFA": HFA, "ERA_WEIGHT": ERA_WEIGHT,
            "WHIP_WEIGHT": WHIP_WEIGHT, "IP_FULL": IP_FULL,
            "LEAGUE_ERA": LEAGUE_ERA, "LEAGUE_WHIP": LEAGUE_WHIP,
        },
    }


# ============================================================================
# metrics
# ============================================================================

def compute_metrics(predictions):
    if not predictions:
        return {"n": 0}
    n = len(predictions)
    ll = brier = 0.0
    correct = 0
    # rolling record baseline: pick team with better wins - losses
    record_correct = 0
    record_defined = 0
    home_wins_actual = 0

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

    # calibration bins (10 bins from 0 to 1)
    bins = 10
    calibration = []
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        if b == bins - 1:
            bucket = [p for p in predictions if lo <= p["p_home"] <= hi]
        else:
            bucket = [p for p in predictions if lo <= p["p_home"] < hi]
        if bucket:
            avg_p = sum(p["p_home"] for p in bucket) / len(bucket)
            act = sum(1 if p["home_won"] else 0 for p in bucket) / len(bucket)
            calibration.append({
                "bin_lo": lo, "bin_hi": hi,
                "n": len(bucket), "avg_pred": avg_p, "actual": act,
            })

    # home-baseline log loss at empirical home-win rate
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
        "home_baseline_accuracy": home_wins_actual / n,
        "home_baseline_log_loss": home_ll / n,
        "record_baseline_accuracy": (
            record_correct / record_defined if record_defined else None
        ),
        "record_baseline_defined": record_defined,
        "calibration": calibration,
    }


# ============================================================================
# season picking & caching
# ============================================================================

def _pick_season():
    """Prefer this year if it has substantial completed regular-season data."""
    y = date.today().year
    for candidate in (y, y - 1):
        try:
            d = _fetch(SCHEDULE_URL.format(
                start=f"{candidate}-03-01", end=f"{candidate}-11-15",
            ))
            n_final = 0
            for db in d.get("dates", []):
                for g in db.get("games", []):
                    if g.get("gameType") == "R" and (g.get("status") or {}).get("codedGameState") == "F":
                        n_final += 1
                        if n_final >= 800:
                            return candidate
            if n_final >= 800:
                return candidate
        except Exception:
            continue
    return y - 1


def get_or_run_backtest(refresh=False, season=None, score_last_n_days=60, verbose=False):
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
    result = run_backtest(season, score_last_n_days=score_last_n_days, verbose=verbose)
    result["metrics"] = compute_metrics(result["predictions"])
    with open(CACHE_FILE, "w") as f:
        json.dump(result, f)
    return result


# ============================================================================
# Multi-season backtest (12 regular seasons aggregate)
# ============================================================================

NUM_SEASONS = 12
MULTI_WARMUP_SEASONS = 2

MULTI_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mlb_multi_backtest_cache.json",
)


def _load_pitcher_gamelogs_multi(pid_season_pairs, workers=16):
    """Fetch pitcher gameLogs for a set of (pid, season) pairs in parallel.

    Returns {(pid, season): [outing, ...]}.
    """
    def _one(args):
        pid, season = args
        try:
            d = _fetch(PITCHER_LOG_URL.format(pid=pid, season=season))
        except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
            return (pid, season), []
        outings = []
        for s in (d.get("stats") or []):
            for split in (s.get("splits") or []):
                stat = split.get("stat") or {}
                dt = split.get("date")
                if not dt:
                    continue
                outings.append({
                    "date": dt,
                    "er": stat.get("earnedRuns") or 0,
                    "ip": _parse_ip(stat.get("inningsPitched", 0)),
                    "so": stat.get("strikeOuts") or 0,
                    "bb": stat.get("baseOnBalls") or 0,
                    "h": stat.get("hits") or 0,
                })
        outings.sort(key=lambda o: o["date"])
        return (pid, season), outings

    out = {}
    pairs = [p for p in pid_season_pairs if p[0]]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p) for p in pairs]
        for f in as_completed(futs):
            key, outings = f.result()
            out[key] = outings
    return out


def run_multi_season_backtest(end_season, num_seasons=NUM_SEASONS,
                               warmup_seasons=MULTI_WARMUP_SEASONS,
                               verbose=True):
    """Walk-forward Elo + SP backtest across `num_seasons` ending at `end_season`.

    Team Elo persists across seasons (regressed 1/3 toward 1500 at each boundary).
    Starting-pitcher stats reset per season (statsapi gameLog is per-season).

    Returns aggregate metrics plus per-season breakdown.
    """
    first = end_season - num_seasons + 1
    seasons = list(range(first, end_season + 1))
    scored_from = first + warmup_seasons

    if verbose:
        print(f"[mlb] multi-season {first}-{end_season} "
              f"(warmup {first}..{scored_from - 1}, scored {scored_from}..{end_season})",
              flush=True)

    # 1. Fetch schedules in parallel
    t0 = time.time()
    all_games = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(load_season_games, s): s for s in seasons}
        for f in as_completed(futs):
            s = futs[f]
            try:
                g = f.result()
            except Exception as e:
                if verbose:
                    print(f"[mlb]   {s}: FAILED {e}", flush=True)
                g = []
            if verbose:
                print(f"[mlb]   {s}: {len(g)} games", flush=True)
            all_games.extend(g)
    all_games.sort(key=lambda g: (g["date"], g["gamePk"]))
    if verbose:
        print(f"[mlb]   {len(all_games)} total games, "
              f"schedules fetched in {time.time()-t0:.1f}s", flush=True)

    # 2. Collect unique (pitcher, season) pairs
    pid_season_pairs = set()
    for g in all_games:
        s_year = int(g["date"][:4])
        for pid in (g["home_sp_id"], g["away_sp_id"]):
            if pid:
                pid_season_pairs.add((pid, s_year))
    if verbose:
        print(f"[mlb] fetching {len(pid_season_pairs)} pitcher-season logs...",
              flush=True)

    t0 = time.time()
    logs = _load_pitcher_gamelogs_multi(pid_season_pairs)
    if verbose:
        print(f"[mlb]   gameLogs fetched in {time.time()-t0:.1f}s", flush=True)

    # 3. Walk forward
    elo = {}
    predictions = []
    per_season = {s: {"n": 0, "correct": 0, "ll": 0.0, "brier": 0.0} for s in seasons[warmup_seasons:]}
    seasons_seen = set()

    for g in all_games:
        s_year = int(g["date"][:4])
        if s_year not in seasons_seen and seasons_seen:
            # Season boundary: regress team Elo toward 1500 by 1/3
            for t in list(elo.keys()):
                elo[t] = INITIAL_ELO + (2.0 / 3.0) * (elo[t] - INITIAL_ELO)
        seasons_seen.add(s_year)

        h = g["home_id"]
        a = g["away_id"]
        elo.setdefault(h, INITIAL_ELO)
        elo.setdefault(a, INITIAL_ELO)

        home_sp = pitcher_stats_asof(
            logs.get((g["home_sp_id"], s_year), []), g["date"]
        ) if g["home_sp_id"] else None
        away_sp = pitcher_stats_asof(
            logs.get((g["away_sp_id"], s_year), []), g["date"]
        ) if g["away_sp_id"] else None

        p_home = predict_win_prob(elo[h], elo[a], home_sp, away_sp)

        if s_year >= scored_from:
            y = 1 if g["home_won"] else 0
            ph = max(1e-6, min(1 - 1e-6, p_home))
            correct = (ph >= 0.5) == (y == 1)
            per_season[s_year]["n"] += 1
            per_season[s_year]["correct"] += 1 if correct else 0
            per_season[s_year]["ll"] -= y * math.log(ph) + (1 - y) * math.log(1 - ph)
            per_season[s_year]["brier"] += (ph - y) ** 2
            predictions.append({
                "season": s_year,
                "date": g["date"],
                "home": g["home_name"], "away": g["away_name"],
                "home_id": h, "away_id": a,
                "home_sp": g["home_sp_name"], "away_sp": g["away_sp_name"],
                "p_home": p_home, "home_won": g["home_won"],
                "home_score": g["home_score"], "away_score": g["away_score"],
            })

        elo[h], elo[a] = elo_update(
            elo[h], elo[a], g["home_won"], g["home_score"], g["away_score"]
        )

    for s, stat in per_season.items():
        if stat["n"] > 0:
            stat["accuracy"] = stat["correct"] / stat["n"]
            stat["log_loss"] = stat["ll"] / stat["n"]
            stat["brier"] = stat["brier"] / stat["n"]
        else:
            stat["accuracy"] = stat["log_loss"] = stat["brier"] = None

    team_names = {g["home_id"]: g["home_name"] for g in all_games}
    team_names.update({g["away_id"]: g["away_name"] for g in all_games})

    return {
        "first_season": first,
        "last_season": end_season,
        "num_seasons": num_seasons,
        "warmup_seasons": warmup_seasons,
        "scored_from_season": scored_from,
        "total_games": len(all_games),
        "scored_games": len(predictions),
        "per_season": per_season,
        "predictions": predictions,
        "final_elo": {str(tid): round(e, 1) for tid, e in elo.items()},
        "team_names": {str(tid): team_names.get(tid, "") for tid in team_names},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {
            "K": K_FACTOR, "HFA": HFA,
            "ERA_WEIGHT": ERA_WEIGHT, "WHIP_WEIGHT": WHIP_WEIGHT,
            "IP_FULL": IP_FULL,
            "LEAGUE_ERA": LEAGUE_ERA, "LEAGUE_WHIP": LEAGUE_WHIP,
        },
    }


ELO_ONLY_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mlb_final_elo.json",
)


def _write_elo_only_cache(result):
    """Slim side-file with just the lookup tables, so get_final_state() can
    serve live predictions without pulling the full 6 MB cache file."""
    try:
        with open(ELO_ONLY_CACHE_FILE, "w") as f:
            json.dump({
                "final_elo":   result.get("final_elo", {}),
                "generated_at": result.get("generated_at"),
            }, f)
    except OSError:
        pass


def get_final_state():
    """Lightweight {final_elo} loader. Reads the slim side-file; if missing,
    derives it from the full backtest cache (and writes the side-file for
    next time); if the full cache is missing too, falls back to running the
    multi-season backtest.
    """
    if os.path.exists(ELO_ONLY_CACHE_FILE):
        try:
            with open(ELO_ONLY_CACHE_FILE) as f:
                return json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            pass
    if os.path.exists(MULTI_CACHE_FILE):
        try:
            with open(MULTI_CACHE_FILE) as f:
                data = json.load(f)
            out = {
                "final_elo":    data.get("final_elo", {}),
                "generated_at": data.get("generated_at"),
            }
            _write_elo_only_cache(data)
            import gc
            del data
            gc.collect()
            return out
        except (OSError, json.JSONDecodeError):
            pass
    state = get_or_run_multi_season_backtest()
    return {
        "final_elo":    (state or {}).get("final_elo", {}),
        "generated_at": (state or {}).get("generated_at"),
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
    _write_elo_only_cache(result)
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
    print(f"=== Backtest report ===")
    print(f"Season                : {result['season']}")
    print(f"Total games (fit)     : {result['total_games']}")
    print(f"Scored window         : {result['score_from']}  ->  {result['score_to']}")
    print(f"Scored games          : {result['scored_games']}")
    print()
    print(f"Model accuracy        : {m['accuracy']:.4f}")
    print(f"Model log loss        : {m['log_loss']:.4f}")
    print(f"Model Brier score     : {m['brier_score']:.4f}")
    print()
    print(f"Home-team baseline    : acc={m['home_baseline_accuracy']:.4f}  ll={m['home_baseline_log_loss']:.4f}")
    if m["record_baseline_accuracy"] is not None:
        print(f"Better-record baseline: acc={m['record_baseline_accuracy']:.4f}  (over {m['record_baseline_defined']} decidable games)")
    print()
    print("Calibration (pred bin -> observed win rate):")
    print("  bin         n     avg_pred  actual")
    for c in m["calibration"]:
        print(f"  {c['bin_lo']:.2f}-{c['bin_hi']:.2f}  {c['n']:5d}   {c['avg_pred']:.3f}    {c['actual']:.3f}")
    print()
    # top 10 teams by final Elo
    elo = result["final_elo"]
    names = result["team_names"]
    ranked = sorted(elo.items(), key=lambda kv: -kv[1])[:10]
    print("Top 10 final Elo:")
    for tid, r in ranked:
        wl = result["final_wl"].get(tid, [0, 0])
        print(f"  {r:7.1f}   {names.get(tid, tid):28s}  {wl[0]}-{wl[1]}")


def main():
    parser = argparse.ArgumentParser(description="MLB Elo+SP backtest")
    parser.add_argument("--refresh", action="store_true", help="force refit, ignore cache")
    parser.add_argument("--season", type=int, default=None, help="season year (default: auto)")
    parser.add_argument("--score-days", type=int, default=60, help="score predictions from last N days (default 60)")
    args = parser.parse_args()

    result = get_or_run_backtest(
        refresh=args.refresh, season=args.season,
        score_last_n_days=args.score_days, verbose=True,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
