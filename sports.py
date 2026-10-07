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
        "espn_scoreboard": "baseball/mlb",
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
        "espn_scoreboard": "football/nfl",
        "ml_outcomes": 2,
        "has_halves": True,           # period 1 = 1st half
        "spread_label": "Spread",
        "team_sport": True,
    },
    {
        "slug": "ncaaf",
        "name": "NCAAF",
        "pinnacle_league_id": 880,
        "odds_api_key": "americanfootball_ncaaf",
        "espn_scoreboard": "football/college-football",
        "ml_outcomes": 2,
        "has_halves": True,
        "spread_label": "Spread",
        "team_sport": True,
    },
    # ---- Ice Hockey ----
    {
        "slug": "nhl",
        "name": "NHL",
        "pinnacle_league_id": 1456,
        "odds_api_key": "icehockey_nhl",
        "espn_scoreboard": "hockey/nhl",
        "ml_outcomes": 2,             # NHL ML is 2-way (regulation/OT/SO)
        "has_halves": True,
        "period_1_label": "1P",
        "spread_label": "Puck Line",
        "team_sport": True,
    },
    # ---- MMA ----
    {
        "slug": "ufc",
        "name": "UFC",
        "pinnacle_league_id": 1624,
        "odds_api_key": "mma_mixed_martial_arts",
        "espn_scoreboard": "mma/ufc",
        "ml_outcomes": 2,
        "has_halves": False,
        "team_sport": False,
    },
    # ---- Soccer ----
    {
        "slug": "epl",
        "name": "EPL",
        "pinnacle_league_id": 1980,
        "odds_api_key": "soccer_epl",
        "espn_scoreboard": "soccer/eng.1",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "laliga",
        "name": "La Liga",
        "pinnacle_league_id": 2196,
        "odds_api_key": "soccer_spain_la_liga",
        "espn_scoreboard": "soccer/esp.1",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "ligamx",
        "name": "Liga MX",
        "pinnacle_league_id": 2242,
        "odds_api_key": "soccer_mexico_ligamx",
        "espn_scoreboard": "soccer/mex.1",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "ucl",
        "name": "Champions Lg",
        "pinnacle_league_id": 2627,
        "odds_api_key": "soccer_uefa_champs_league",
        "espn_scoreboard": "soccer/uefa.champions",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "europa",
        "name": "Europa Lg",
        "pinnacle_league_id": 2630,
        "odds_api_key": "soccer_uefa_europa_league",
        "espn_scoreboard": "soccer/uefa.europa",
        "ml_outcomes": 3,
        "has_halves": True,
        "team_sport": True,
    },
    {
        "slug": "international",
        "name": "Int'l",
        # Aggregated: Friendlies + UEFA Nations League + CONCACAF Nations League
        # (Pinnacle guest API exposes each as its own league id; generic_odds
        # merges them under one "International" view).
        "pinnacle_league_id": [
            2117,     # International Friendlies
            200719,   # UEFA Nations League League A
            200721,   # UEFA Nations League League B
            200726,   # UEFA Nations League League C
            200727,   # UEFA Nations League League D
            200729,   # UEFA Nations League (playoffs / finals)
            205419,   # CONCACAF Nations League
        ],
        "odds_api_key": "soccer_uefa_nations_league",
        # ESPN hosts internationals under several league paths — list ALL
        # so the grader aggregates across friendlies, qualifiers, nations
        # leagues and World Cup. fetch_scores_espn walks every path.
        "espn_scoreboard": [
            "soccer/fifa.friendly",
            "soccer/fifa.friendly.w",
            "soccer/uefa.nations",
            "soccer/concacaf.nations.league",
            "soccer/fifa.worldq.uefa",
            "soccer/fifa.worldq.concacaf",
            "soccer/fifa.worldq.conmebol",
            "soccer/fifa.worldq.afc",
            "soccer/fifa.worldq.caf",
            "soccer/fifa.world",
        ],
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
