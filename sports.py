#!/usr/bin/env python3
"""Sport + league registry.

Each entry maps a URL slug to the Pinnacle league ID (for free lines) and the
Odds API sport key (for DK/FanDuel/BetMGM/Caesars comparison).

MLB keeps its dedicated model path (mlb_odds.py). All other sports are served
by generic_odds.py: Pinnacle devigged fair probability vs. US book prices.
"""

SPORTS = [
    # ---- MLB uses its own dedicated model path, listed here for the nav ----
    {
        "slug": "mlb",
        "name": "MLB",
        "pinnacle_league_id": 246,
        "odds_api_key": "baseball_mlb",
        "ml_outcomes": 2,
        "has_halves": False,          # period 1 = F5, handled by mlb_odds
        "spread_label": "Run Line",
        "team_sport": True,
        "dedicated": True,            # routed through MLB-specific pages
        "home_route": "/",            # Schedule home
    },
    # ---- American Football ----
    {
        "slug": "nfl",
        "name": "NFL",
        "pinnacle_league_id": 889,
        "odds_api_key": "americanfootball_nfl",
        "ml_outcomes": 2,
        "has_halves": True,           # period 1 = 1st half
        "spread_label": "Spread",
        "team_sport": True,
    },
    # ---- MMA ----
    {
        "slug": "ufc",
        "name": "UFC",
        "pinnacle_league_id": 1624,
        "odds_api_key": "mma_mixed_martial_arts",
        "ml_outcomes": 2,
        "has_halves": False,
        "team_sport": False,          # participants are fighters
    },
    # ---- Soccer ----
    {
        "slug": "epl",
        "name": "EPL",
        "pinnacle_league_id": 1980,
        "odds_api_key": "soccer_epl",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "laliga",
        "name": "La Liga",
        "pinnacle_league_id": 2196,
        "odds_api_key": "soccer_spain_la_liga",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "ligamx",
        "name": "Liga MX",
        "pinnacle_league_id": 2242,
        "odds_api_key": "soccer_mexico_ligamx",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "ucl",
        "name": "Champions Lg",
        "pinnacle_league_id": 2627,
        "odds_api_key": "soccer_uefa_champs_league",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "europa",
        "name": "Europa Lg",
        "pinnacle_league_id": 2630,
        "odds_api_key": "soccer_uefa_europa_league",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "international",
        "name": "Int'l",
        "pinnacle_league_id": 2117,   # International Friendlies
        "odds_api_key": None,         # Odds API doesn't have a clean friendlies key
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
]


def by_slug(slug):
    for s in SPORTS:
        if s["slug"] == slug:
            return s
    return None


def all_slugs():
    return [s["slug"] for s in SPORTS]
