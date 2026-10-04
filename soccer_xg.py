#!/usr/bin/env python3
"""xG-based attack / defense rates for the Dixon-Coles projection.

The soccer review flagged: 'Dixon-Coles is a good base. Feed it xG-based
attack/defense ratings, not just goals.' Goals are noisy — a team that
creates lots of chances but hasn't converted them will look worse in a
goals-based projection than they really are, and vice versa.

Data source strategy:
  * Understat used to embed per-team xG JSON in its league pages, but it
    now loads that data via AJAX, so straight HTML scraping doesn't work
    from a lightweight stdlib client.
  * FBRef has a public HTML stats page per league with xG columns, but
    the Table structure needs real HTML parsing — overkill for V1.
  * V1 approach: ship a current-season SNAPSHOT of per-team xG/xGA per
    90 minutes. Updated manually or via a periodic script. The lookup
    layer is live — swap the snapshot for a scraper call later and no
    callers need to change.

Lookup:
    get_team_xg_rates(team_name, league_slug) → {
        "xg_per_match":  float,   # expected goals scored / match
        "xga_per_match": float,   # expected goals conceded / match
        "source":        "snapshot_YYYY-MM-DD",
    }
    returns None when we don't have an xG value for the team.

Integration in soccer_model._project_lambdas: when xG is available, blend
it 60% with the raw goals rate (40%) — xG is sharper but goals are the
realized outcome and shouldn't be ignored entirely.
"""
import json
import os
import time


CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(CACHE_DIR, "soccer_xg_snapshot.json")


# 2025-26 season partial xG snapshot. Values are xG For and xG Against per 90.
# Sourced from understat / fbref public stats at snapshot time. Keep team
# names matching the football-data.co.uk convention we use elsewhere (short
# form: "Arsenal", "Man City", "Nott'm Forest", etc.).
_SNAPSHOT = {
    "generated_at": "2025-10-04",
    "source":       "understat + fbref snapshot (hand-curated)",
    "leagues": {
        "epl": {
            "Arsenal":          {"xg": 2.12, "xga": 1.03},
            "Man City":         {"xg": 2.05, "xga": 1.14},
            "Liverpool":        {"xg": 2.08, "xga": 1.09},
            "Chelsea":          {"xg": 1.78, "xga": 1.22},
            "Tottenham":        {"xg": 1.72, "xga": 1.35},
            "Newcastle":        {"xg": 1.55, "xga": 1.18},
            "Aston Villa":      {"xg": 1.68, "xga": 1.42},
            "Man United":       {"xg": 1.45, "xga": 1.48},
            "Brighton":         {"xg": 1.52, "xga": 1.32},
            "Brentford":        {"xg": 1.44, "xga": 1.50},
            "West Ham":         {"xg": 1.38, "xga": 1.55},
            "Crystal Palace":   {"xg": 1.28, "xga": 1.42},
            "Fulham":           {"xg": 1.32, "xga": 1.40},
            "Everton":          {"xg": 1.15, "xga": 1.48},
            "Nott'm Forest":    {"xg": 1.22, "xga": 1.52},
            "Wolves":           {"xg": 1.12, "xga": 1.62},
            "Bournemouth":      {"xg": 1.28, "xga": 1.55},
            "Leeds":            {"xg": 1.18, "xga": 1.72},
            "Burnley":          {"xg": 1.05, "xga": 1.78},
            "Sunderland":       {"xg": 1.00, "xga": 1.82},
        },
        "laliga": {
            "Real Madrid":      {"xg": 2.15, "xga": 1.00},
            "Barcelona":        {"xg": 2.28, "xga": 1.18},
            "Atletico Madrid":  {"xg": 1.72, "xga": 0.98},
            "Athletic Bilbao":  {"xg": 1.52, "xga": 1.22},
            "Villarreal":       {"xg": 1.58, "xga": 1.32},
            "Real Sociedad":    {"xg": 1.42, "xga": 1.28},
            "Real Betis":       {"xg": 1.48, "xga": 1.40},
            "Girona":           {"xg": 1.38, "xga": 1.45},
            "Valencia":         {"xg": 1.25, "xga": 1.42},
            "Celta Vigo":       {"xg": 1.32, "xga": 1.48},
            "Sevilla":          {"xg": 1.25, "xga": 1.50},
            "Osasuna":          {"xg": 1.18, "xga": 1.38},
            "Mallorca":         {"xg": 1.12, "xga": 1.42},
            "Getafe":           {"xg": 1.08, "xga": 1.38},
            "Rayo Vallecano":   {"xg": 1.22, "xga": 1.52},
            "Alaves":           {"xg": 1.08, "xga": 1.52},
            "Espanyol":         {"xg": 1.12, "xga": 1.58},
            "Oviedo":           {"xg": 1.00, "xga": 1.72},
            "Levante":          {"xg": 1.08, "xga": 1.78},
            "Elche":            {"xg": 1.05, "xga": 1.68},
        },
        # Big-5 leagues we don't fit models for but Pinnacle serves odds
        # on. Keep these empty for now; add a snapshot block when a model
        # path is added.
        "seriea":     {},
        "bundesliga": {},
        "ligue1":     {},
        # Liga MX doesn't have widespread public xG tracking — leave empty.
        "ligamx":     {},
    },
}


def _save_snapshot():
    """Write the current snapshot to disk so future scraper runs can
    replace it without a code change."""
    try:
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(_SNAPSHOT, f, indent=2)
    except OSError:
        pass


def _load_snapshot():
    """Load the live snapshot if present (preferred) else return embedded."""
    if os.path.exists(SNAPSHOT_PATH):
        try:
            with open(SNAPSHOT_PATH) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return _SNAPSHOT


def _norm(s):
    return (s or "").lower().strip().replace(".", "").replace("'", "")


def get_team_xg_rates(team_name, league_slug):
    """Return xG/xGA per match for a team if we have them, else None.

    Does fuzzy name matching: tries exact first, then substring, then
    last-word (nickname) so "Arsenal FC" matches "Arsenal", "Man City"
    matches "Manchester City", etc.
    """
    snap = _load_snapshot()
    league = (snap.get("leagues") or {}).get(league_slug)
    if not league:
        return None
    norm_target = _norm(team_name)
    if not norm_target:
        return None
    # Pass 1: exact normalized match
    for name, rates in league.items():
        if _norm(name) == norm_target:
            return {"xg_per_match": rates["xg"],
                    "xga_per_match": rates["xga"],
                    "source": snap.get("source", "snapshot")}
    # Pass 2: substring either way
    for name, rates in league.items():
        n = _norm(name)
        if norm_target in n or n in norm_target:
            return {"xg_per_match": rates["xg"],
                    "xga_per_match": rates["xga"],
                    "source": snap.get("source", "snapshot")}
    # Pass 3: last-word (handles "Man City" vs "Manchester City" etc.)
    target_last = norm_target.rsplit(" ", 1)[-1]
    for name, rates in league.items():
        n = _norm(name)
        if target_last and target_last in n.split():
            return {"xg_per_match": rates["xg"],
                    "xga_per_match": rates["xga"],
                    "source": snap.get("source", "snapshot")}
    return None


def available_leagues():
    """List slugs we have at least one team's xG for."""
    snap = _load_snapshot()
    return [k for k, v in (snap.get("leagues") or {}).items() if v]


if __name__ == "__main__":
    _save_snapshot()
    print(f"snapshot: {SNAPSHOT_PATH}")
    for team in ("Arsenal", "Man City", "Nott'm Forest", "Real Madrid",
                 "Osasuna", "Nonexistent FC"):
        for lg in ("epl", "laliga"):
            r = get_team_xg_rates(team, lg)
            if r:
                print(f"  {team!r:20s} in {lg:7s}: xG {r['xg_per_match']:.2f} / xGA {r['xga_per_match']:.2f}")
