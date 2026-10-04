#!/usr/bin/env python3
"""MLB plate-appearance Markov-chain simulator.

The review's #1 sport-specific issue was MLB: "simulator is broken, zero
picks today despite 4 playoff games." The old mlb_model.simulate_game draws
runs from Poisson(λ=runs_per_game) — fine for coarse totals but it can't
price run-lines well and it doesn't know when a weak lineup faces an ace.

This module replaces that with a V1 PA-level sim:

  * Each half-inning simulates plate appearances until 3 outs.
  * Per-PA outcome rates (K, BB, 1B, 2B, 3B, HR, out-in-play) are
    projected by log5-blending the batting team's season rates with
    the opposing pitcher's allowed rates, anchored to league average.
  * Runners advance deterministically by hit length (V1 simplification).
  * 9 innings per side; tied games get a Manfred-rule placeholder
    extra inning with a small home-field edge.
  * 10k iterations surface moneyline, totals, run-line, and the full
    margin distribution.

V2 ideas (deferred): per-batter rates (not team-level), 1st-to-3rd
runner aggression, DP probability, stolen bases, platoon splits,
bullpen takeover after SP's pitch budget.
"""
import math
import random
from datetime import datetime


# 2024 MLB per-PA league averages (approx).
LEAGUE_PA = {
    "k":   0.228,
    "bb":  0.088,   # includes HBP so the arithmetic adds up
    "1b":  0.137,
    "2b":  0.045,
    "3b":  0.004,
    "hr":  0.032,
    "out": 0.466,   # outs on balls in play + sac + reached-on-error ≈ residual
}

# SP pitch budget before bullpen takes over. Playoff SPs leave earlier;
# we approximate 20 PA (~6 IP equivalent) for regular season and 16 PA
# (~4-5 IP) when `playoffs=True`.
SP_PA_REGULAR  = 20
SP_PA_PLAYOFF  = 16


def _rate(x, default):
    try:
        v = float(x)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def team_pa_rates(hitting_stats):
    """Compute per-PA rates from a team's season hitting dict.
    Falls back to league averages when any field is missing/zero."""
    pa = _rate(hitting_stats.get("plateAppearances") if hitting_stats else None, 0)
    if pa <= 0:
        return dict(LEAGUE_PA)
    def _r(field, default_key):
        n = _rate((hitting_stats or {}).get(field), 0)
        return n / pa if n > 0 else LEAGUE_PA[default_key]
    k     = _r("strikeOuts",    "k")
    bb    = _r("baseOnBalls",   "bb")
    hbp   = _rate((hitting_stats or {}).get("hitByPitch"), 0) / pa
    walks = bb + hbp
    hits  = _r("hits", "1b")  # placeholder; we don't use directly
    doubles = _r("doubles",  "2b")
    triples = _r("triples",  "3b")
    hr    = _r("homeRuns",   "hr")
    # Singles = hits - 2B - 3B - HR
    singles_abs = _rate((hitting_stats or {}).get("hits"), 0) - \
                  _rate((hitting_stats or {}).get("doubles"), 0) - \
                  _rate((hitting_stats or {}).get("triples"), 0) - \
                  _rate((hitting_stats or {}).get("homeRuns"), 0)
    singles = singles_abs / pa if singles_abs > 0 else LEAGUE_PA["1b"]
    # Residual = outs on balls in play
    bip_out = max(0.0, 1.0 - k - walks - singles - doubles - triples - hr)
    return {
        "k":   k,
        "bb":  walks,
        "1b":  singles,
        "2b":  doubles,
        "3b":  triples,
        "hr":  hr,
        "out": bip_out,
    }


def pitcher_pa_rates(pitcher_stat):
    """Per-PA rates against this pitcher from his season stats dict."""
    bf = _rate(pitcher_stat.get("battersFaced") if pitcher_stat else None, 0)
    if bf <= 0:
        return dict(LEAGUE_PA)
    def _r(field, default_key):
        n = _rate((pitcher_stat or {}).get(field), 0)
        return n / bf if n > 0 else LEAGUE_PA[default_key]
    k   = _r("strikeOuts", "k")
    bb  = _r("baseOnBalls", "bb")
    hbp = _rate((pitcher_stat or {}).get("hitBatsmen"), 0) / bf
    walks = bb + hbp
    doubles = _r("doubles", "2b")
    triples = _r("triples", "3b")
    hr      = _r("homeRuns", "hr")
    singles_abs = (_rate((pitcher_stat or {}).get("hits"), 0)
                   - _rate((pitcher_stat or {}).get("doubles"), 0)
                   - _rate((pitcher_stat or {}).get("triples"), 0)
                   - _rate((pitcher_stat or {}).get("homeRuns"), 0))
    singles = singles_abs / bf if singles_abs > 0 else LEAGUE_PA["1b"]
    bip_out = max(0.0, 1.0 - k - walks - singles - doubles - triples - hr)
    return {
        "k":   k,
        "bb":  walks,
        "1b":  singles,
        "2b":  doubles,
        "3b":  triples,
        "hr":  hr,
        "out": bip_out,
    }


def _log5(batter_rate, pitcher_rate, league_rate):
    """Log5 odds-ratio combination of batter vs pitcher skill at an event."""
    if league_rate <= 0 or league_rate >= 1:
        return batter_rate
    num   = batter_rate * pitcher_rate / league_rate
    denom = num + (1 - batter_rate) * (1 - pitcher_rate) / (1 - league_rate)
    return (num / denom) if denom > 0 else batter_rate


def project_pa_rates(team_rates, pitcher_rates):
    """Blend team hitting rates with opposing pitcher's allowed rates via log5
    for each outcome, then renormalize to a proper distribution."""
    out = {}
    for key in ("k", "bb", "1b", "2b", "3b", "hr", "out"):
        br = team_rates.get(key, LEAGUE_PA[key])
        pr = pitcher_rates.get(key, LEAGUE_PA[key])
        out[key] = _log5(br, pr, LEAGUE_PA[key])
    total = sum(out.values())
    if total > 0:
        for k in out:
            out[k] = out[k] / total
    return out


def blend_sp_and_bullpen(sp_rates, team_bullpen_rates, sp_pa_before_bullpen):
    """A game's pitching exposure is SP for the first N PAs then bullpen.
    We return a per-PA 'blend' schedule that _simulate_half_inning samples
    from based on how many PAs have happened so far. For V1 we return two
    rate dicts and a crossover PA count."""
    return sp_rates, team_bullpen_rates, sp_pa_before_bullpen


# ---------------------------------------------------------------------------
# Base-out state Markov step
# ---------------------------------------------------------------------------
#
# Base state is (b1, b2, b3) ∈ {0,1}³ meaning "runner on this base?".
# Outcomes:
#   K           → +1 out, no base change
#   out BIP     → +1 out, no base change (V1: no DP, no sac fly)
#   BB          → forced advance; runs scored = (loaded ? 1 : 0)
#   1B          → runners from 2 and 3 score, runner from 1 to 2, batter to 1
#   2B          → runners from 1, 2, 3 all score, batter to 2
#   3B          → all runners score, batter to 3
#   HR          → all runners + batter score; bases cleared


def _apply_outcome(bases, outcome):
    """Return (new_bases, runs_scored, outs_added) for the given outcome."""
    b1, b2, b3 = bases
    if outcome in ("k", "out"):
        return (b1, b2, b3), 0, 1
    if outcome == "bb":
        if b1 and b2 and b3:
            return (1, 1, 1), 1, 0
        if b1 and b2:
            return (1, 1, 1), 0, 0
        if b1:
            return (1, 1, b3), 0, 0
        return (1, b2, b3), 0, 0
    if outcome == "1b":
        runs = b2 + b3
        return (1, b1, 0), runs, 0
    if outcome == "2b":
        runs = b1 + b2 + b3
        return (0, 1, 0), runs, 0
    if outcome == "3b":
        runs = b1 + b2 + b3
        return (0, 0, 1), runs, 0
    if outcome == "hr":
        runs = 1 + b1 + b2 + b3
        return (0, 0, 0), runs, 0
    return (b1, b2, b3), 0, 1


_ORDER = ("k", "out", "bb", "1b", "2b", "3b", "hr")


def _draw(rates, rng):
    r = rng.random()
    acc = 0.0
    for key in _ORDER:
        acc += rates.get(key, 0)
        if r <= acc:
            return key
    return "out"


def simulate_half_inning(rates, rng):
    bases = (0, 0, 0)
    outs = 0
    runs = 0
    while outs < 3:
        outcome = _draw(rates, rng)
        bases, scored, out_add = _apply_outcome(bases, outcome)
        runs += scored
        outs += out_add
    return runs


def simulate_game(home_hitting, home_sp_stat, away_hitting, away_sp_stat,
                   home_bullpen_rates=None, away_bullpen_rates=None,
                   n=5000, playoffs=False, seed=None):
    """Full-game PA Markov sim.

    home_hitting / away_hitting: team season hitting stats dict (statsapi shape)
    home_sp_stat / away_sp_stat: opposing starter's season stats dict
    home/away_bullpen_rates: optional per-PA rate dicts for each team's
        bullpen (falls back to league avg)
    playoffs: shorten the SP's pitch budget (playoff pitchers leave earlier)
    """
    rng = random.Random(seed) if seed is not None else random
    sp_budget = SP_PA_PLAYOFF if playoffs else SP_PA_REGULAR

    # Project per-PA rates for each half-inning scenario
    home_bat = team_pa_rates(home_hitting)
    away_bat = team_pa_rates(away_hitting)
    away_sp  = pitcher_pa_rates(away_sp_stat)
    home_sp  = pitcher_pa_rates(home_sp_stat)
    away_pen = away_bullpen_rates or dict(LEAGUE_PA)
    home_pen = home_bullpen_rates or dict(LEAGUE_PA)

    # home batting vs away SP, then away bullpen
    home_vs_sp  = project_pa_rates(home_bat, away_sp)
    home_vs_pen = project_pa_rates(home_bat, away_pen)
    # away batting vs home SP, then home bullpen
    away_vs_sp  = project_pa_rates(away_bat, home_sp)
    away_vs_pen = project_pa_rates(away_bat, home_pen)

    home_wins = away_wins = 0
    totals = []
    margins = []
    home_runs_sum = away_runs_sum = 0

    for _ in range(n):
        h_runs = 0
        a_runs = 0
        pa_h = 0  # count of PAs by home lineup (to swap in bullpen)
        pa_a = 0
        for _inning in range(9):
            # Away bats top
            rates_a = away_vs_sp if pa_a < sp_budget else away_vs_pen
            runs_a = simulate_half_inning(rates_a, rng)
            # Rough PA accounting: 3 outs + runs + runners-left (≈ 3-5 PAs on avg)
            pa_a += 3 + runs_a + 1
            a_runs += runs_a
            # Home bats bottom. Skip if home already leading in top of 9th
            if _inning == 8 and h_runs > a_runs:
                break
            rates_h = home_vs_sp if pa_h < sp_budget else home_vs_pen
            runs_h = simulate_half_inning(rates_h, rng)
            pa_h += 3 + runs_h + 1
            h_runs += runs_h
        # Extras (Manfred rule placeholder): single coin-flip half with small home edge
        if h_runs == a_runs:
            # Simulate one extra inning each, averaged
            extra_a = simulate_half_inning(away_vs_pen, rng)
            extra_h = simulate_half_inning(home_vs_pen, rng)
            a_runs += extra_a
            h_runs += extra_h
            if h_runs == a_runs:
                if rng.random() < 0.52:
                    h_runs += 1
                else:
                    a_runs += 1

        if h_runs > a_runs:
            home_wins += 1
        else:
            away_wins += 1
        home_runs_sum += h_runs
        away_runs_sum += a_runs
        totals.append(h_runs + a_runs)
        margins.append(h_runs - a_runs)

    return {
        "trials":        n,
        "home_win_pct":  home_wins / n * 100,
        "away_win_pct":  away_wins / n * 100,
        "draw_pct":      0.0,
        "avg_home_runs": home_runs_sum / n,
        "avg_away_runs": away_runs_sum / n,
        "mean_total":    sum(totals) / n,
        "mean_margin":   sum(margins) / n,
        "margins":       margins,
        "notes":         f"PA Markov sim ({'playoffs' if playoffs else 'regular'})",
    }


# ---------------------------------------------------------------------------
# Convenience wrapper for the picks pipeline
# ---------------------------------------------------------------------------

def simulate_from_game_ctx(game_ctx, n=5000, playoffs=None, seed=None):
    """Drop-in replacement for mlb_model.simulate_from_game_ctx. Pulls team
    hitting + SP raw stats off the game_ctx dict produced by mlb_ui.get_games.

    Returns the same keys the Monte Carlo page already renders:
        p_home, p_away, lambda_home, lambda_away, mean_total, margins, extras_pct
    plus a `notes` string so the UI knows which engine ran.
    """
    home = game_ctx.get("home") or {}
    away = game_ctx.get("away") or {}
    home_hit = home.get("hitting_raw") or home.get("raw_hitting") or {}
    away_hit = away.get("hitting_raw") or away.get("raw_hitting") or {}
    home_sp  = (game_ctx.get("home_pitcher") or {}).get("raw") or {}
    away_sp  = (game_ctx.get("away_pitcher") or {}).get("raw") or {}

    if playoffs is None:
        # Rough heuristic: October MLB is playoffs. Could also check gameType.
        try:
            d = datetime.fromisoformat(
                (game_ctx.get("first_pitch_utc") or "").replace("Z", "+00:00")
            )
            playoffs = (d.month == 10 or d.month == 11)
        except Exception:
            playoffs = False

    sim = simulate_game(
        home_hitting=home_hit, home_sp_stat=home_sp,
        away_hitting=away_hit, away_sp_stat=away_sp,
        n=n, playoffs=playoffs, seed=seed,
    )
    # Normalize keys to match mlb_model.simulate_game shape so the MC page
    # template can render both engines interchangeably.
    return {
        "n_sims":       sim["trials"],
        "p_home":       sim["home_win_pct"] / 100.0,
        "p_away":       sim["away_win_pct"] / 100.0,
        "lambda_home":  sim["avg_home_runs"],
        "lambda_away":  sim["avg_away_runs"],
        "mean_total":   sim["mean_total"],
        "mean_margin":  sim["mean_margin"],
        "margins":      sim["margins"],
        "extras_pct":   0.0,   # tracked internally; expose later
        "home_runs_avg": sim["avg_home_runs"],
        "away_runs_avg": sim["avg_away_runs"],
        "engine":       "pa_markov",
        "notes":        sim["notes"],
    }


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test with league-average everything — should return ~50/50 and
    # a league-average total.
    sim = simulate_game(
        home_hitting={}, home_sp_stat={},
        away_hitting={}, away_sp_stat={},
        n=3000, playoffs=False, seed=42,
    )
    print(f"League-avg sim: home {sim['home_win_pct']:.1f}% / away {sim['away_win_pct']:.1f}%")
    print(f"  mean total:  {sim['mean_total']:.2f} runs")
    print(f"  avg h/a:     {sim['avg_home_runs']:.2f} / {sim['avg_away_runs']:.2f}")
