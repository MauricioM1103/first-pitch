#!/usr/bin/env python3
"""NHL goalie adjustments for the Poisson sim.

Review flagged: 'Add confirmed starting goalies; it's the biggest single
factor in NHL pricing.' A pulled-goalie model that treats every team's
expected shots-to-goals conversion as league-average will mis-price
games where the real starter is e.g. Shesterkin (0.935) vs league-avg
(0.905).

NHL public API endpoint `/v1/gamecenter/{gameId}/landing` returns a
`matchup.goalieSeasonStats.goalies` list per team. There's no explicit
'projected starter' flag, so we use the goalie with the most games
played this season (team's #1 by rotation) as a proxy. For teams whose
#1 and backup have comparable GP, the fetched teamTotals.savePctg still
blends both so the aggregate is reasonable.

Lambda adjustment formula (applied in nhl_model.project_lambdas when a
goalie adjustment dict is supplied):

    λ_against_team = λ_base * (1 - team_savePct) / (1 - LEAGUE_AVG)

A team allowing more shots-into-net than league avg (lower savePct) sees
opponent λ raise; elite goaltending (higher savePct) suppresses opponent λ.
"""
import json
import os
import time
from urllib.request import Request, urlopen


LEAGUE_AVG_SAVE_PCT = 0.905   # 2024-25 NHL league average
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Cache the per-day lookup since each /picks render would otherwise hit
# ~15 game-center endpoints.
_GAME_GOALIES_CACHE = {}  # (date_str,) -> {home_team: {team_savePct, projected_name, projected_savePct}, ...}
_CACHE_TTL_S = 60 * 60    # 1 hour


def _fetch_json(url, timeout=15):
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _norm_name(d):
    if isinstance(d, dict):
        return d.get("default") or d.get("en") or ""
    return str(d or "")


def _project_starter(goalies, team_id):
    """Pick the goalie we think will start: team's #1 by games played this
    season. Returns dict or None."""
    cands = [g for g in (goalies or []) if g.get("teamId") == team_id]
    cands = [g for g in cands if g.get("gamesPlayed")]
    if not cands:
        return None
    cands.sort(key=lambda g: -(g.get("gamesPlayed") or 0))
    return cands[0]


def fetch_daily_goalie_stats(date_str):
    """Return {team_short_name: {projected_savePct, projected_name, team_savePct,
                                  team_gaa, team_record}} for all today's games.

    Falls back silently to {} on any error — nhl_model treats missing data
    as 'no adjustment, use team season Poisson base'.
    """
    now = time.time()
    hit = _GAME_GOALIES_CACHE.get(date_str)
    if hit and now - hit[1] < _CACHE_TTL_S:
        return hit[0]

    out = {}
    try:
        sched = _fetch_json(f"https://api-web.nhle.com/v1/schedule/{date_str}")
    except Exception:
        _GAME_GOALIES_CACHE[date_str] = (out, now)
        return out
    games_today = []
    for day in sched.get("gameWeek", []):
        if day.get("date") == date_str:
            games_today = day.get("games", []) or []
            break
    for g in games_today:
        state = g.get("gameState", "")
        if state in ("OFF", "FINAL"):
            continue  # already played
        gid = g.get("id")
        if not gid:
            continue
        try:
            landing = _fetch_json(
                f"https://api-web.nhle.com/v1/gamecenter/{gid}/landing"
            )
        except Exception:
            continue
        matchup = landing.get("matchup") or {}
        gcomp = matchup.get("goalieComparison") or {}
        goalies_list = (matchup.get("goalieSeasonStats") or {}).get("goalies") or []

        home_team = g.get("homeTeam") or {}
        away_team = g.get("awayTeam") or {}
        for side_key, team in (("home", home_team), ("away", away_team)):
            team_id = team.get("id")
            team_name_short = team.get("abbrev") or ""
            team_full = (team.get("placeName") or {}).get("default", "") + " " + \
                        (team.get("commonName") or {}).get("default", "")
            team_full = team_full.strip() or team_name_short
            td = gcomp.get(f"{side_key}Team") or {}
            tt = td.get("teamTotals") or {}
            starter = _project_starter(goalies_list, team_id)
            out[team_name_short] = {
                "team_short":       team_name_short,
                "team_full":        team_full,
                "projected_name":   _norm_name((starter or {}).get("name")) if starter else None,
                "projected_savePct": (starter or {}).get("savePctg"),
                "projected_gaa":    (starter or {}).get("goalsAgainstAvg"),
                "projected_record": f"{(starter or {}).get('wins', 0)}-{(starter or {}).get('losses', 0)}-{(starter or {}).get('otLosses', 0)}" if starter else None,
                "team_savePct":     tt.get("savePctg"),
                "team_gaa":         tt.get("gaa"),
                "team_record":      tt.get("record"),
            }
            # Also key by full-team-name lookup since our models use full names
            if team_full and team_full != team_name_short:
                out[team_full] = out[team_name_short]
    _GAME_GOALIES_CACHE[date_str] = (out, now)
    return out


def goalie_adjustment_factor(save_pct, league_avg=LEAGUE_AVG_SAVE_PCT):
    """Return the multiplicative factor to apply to opponent's base λ.

    Values >1 mean opponents score MORE against this team (poor goaltending);
    values <1 mean opponents score LESS (elite goaltending). Clamped to a
    reasonable band so a bad early-season sample doesn't wildly shift λ.
    """
    if save_pct is None or save_pct <= 0:
        return 1.0
    try:
        sp = float(save_pct)
    except (TypeError, ValueError):
        return 1.0
    denom = max(0.001, 1 - league_avg)
    num = max(0.001, 1 - sp)
    factor = num / denom
    return max(0.70, min(1.30, factor))


def project_goalie_adjustments(home_team, away_team, date_str=None):
    """Return {home_factor, away_factor, home_goalie, away_goalie} for a game.

    home_factor scales the HOME team's conceded λ (so applies to AWAY team's
    scoring). Symmetric for away_factor. When no stats are available for a
    team, that side's factor stays at 1.0 (no adjustment).
    """
    if date_str is None:
        from datetime import date
        date_str = date.today().isoformat()
    stats = fetch_daily_goalie_stats(date_str)
    home_stats = stats.get(home_team) or {}
    away_stats = stats.get(away_team) or {}
    # Prefer projected-starter savePct, fall back to team totals
    home_sp = home_stats.get("projected_savePct") or home_stats.get("team_savePct")
    away_sp = away_stats.get("projected_savePct") or away_stats.get("team_savePct")
    return {
        "home_factor": goalie_adjustment_factor(home_sp),
        "away_factor": goalie_adjustment_factor(away_sp),
        "home_goalie": home_stats.get("projected_name"),
        "away_goalie": away_stats.get("projected_name"),
        "home_save_pct": home_sp,
        "away_save_pct": away_sp,
    }


if __name__ == "__main__":
    from datetime import date
    stats = fetch_daily_goalie_stats(date.today().isoformat())
    print(f"got stats for {len(stats)} team references today")
    for team, s in list(stats.items())[:10]:
        if s.get("projected_name"):
            print(f"  {team:30s}: {s['projected_name']} savePct={s['projected_savePct']}, team savePct={s['team_savePct']}")
