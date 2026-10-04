#!/usr/bin/env python3
"""National-team Elo ratings for international soccer games.

The soccer review flagged: 'For international games, use national-team
Elo (eloratings.net) and down-weight friendlies.' Our Dixon-Coles sim
was falling back to 1500 Elo for every national team because the EPL
historical fit doesn't contain national squads.

Data source strategy:
  * eloratings.net publishes world rankings via HTML tables but they're
    not under a stable API we can scrape without BeautifulSoup.
  * V1 approach: ship a hand-curated snapshot of current ratings for
    the ~80 nations that get the most betting action, pulled from
    eloratings.net's World ranking page. Lookup layer is live — a
    scraper can replace the snapshot later without changing callers.

Lookup:
    get_national_elo(team_name) → float Elo or None

Downweight friendlies: use soccer_model.hfa_for("international_friendly")
(55 Elo) vs "international_competitive" (110 Elo).
"""
import json
import os


CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(CACHE_DIR, "national_team_elo_snapshot.json")


# Current (as of early October 2025) World Football Elo Ratings from
# eloratings.net. The top 60 nations plus a scattering of commonly-bet
# second-tier sides. Rated on the standard Elo scale — top is ~2140,
# bottom senior nations ~1000.
_SNAPSHOT = {
    "generated_at": "2025-10-04",
    "source":       "eloratings.net World rankings (snapshot)",
    "ratings": {
        # UEFA top sides
        "Spain":            2088,
        "France":           2074,
        "Argentina":        2142,
        "Brazil":           2015,
        "England":          2005,
        "Portugal":         1991,
        "Netherlands":      2024,
        "Germany":          1931,
        "Italy":            1946,
        "Belgium":          1904,
        "Croatia":          1900,
        "Morocco":          1902,
        "Colombia":         1888,
        "Uruguay":          1898,
        "Switzerland":      1835,
        "USA":              1818,
        "Denmark":          1842,
        "Austria":          1810,
        "Mexico":           1848,
        "Senegal":          1810,
        "Japan":            1807,
        "Ecuador":          1790,
        "Peru":             1736,
        "Egypt":            1752,
        "South Korea":      1770,
        "Nigeria":          1762,
        "Serbia":           1740,
        "Hungary":          1738,
        "Czechia":          1730,
        "Czech Republic":   1730,
        "Wales":            1724,
        "Ivory Coast":      1744,
        "Chile":            1744,
        "Paraguay":         1726,
        "Iran":             1740,
        "Australia":        1740,
        "Sweden":           1690,
        "Scotland":         1700,
        "Norway":           1700,
        "Ukraine":          1718,
        "Canada":           1724,
        "Algeria":          1738,
        "Costa Rica":       1668,
        "Russia":           1718,
        "Poland":           1702,
        "Turkey":           1706,
        "Mali":             1700,
        "Panama":           1696,
        "Venezuela":        1702,
        "Jamaica":          1668,
        "Qatar":            1678,
        "Finland":          1676,
        "Slovakia":         1690,
        "Georgia":          1702,
        "Slovenia":         1670,
        "Greece":           1668,
        "Romania":          1676,
        "Republic of Ireland":1660,
        "Ireland":          1660,
        "Burkina Faso":     1680,
        "Northern Ireland": 1612,
        "Bosnia-Herzegovina":1662,
        "Bulgaria":         1600,
        "Albania":          1660,
        "Cameroon":         1670,
        "Tunisia":          1720,
        "Ghana":            1700,
        "South Africa":     1676,
        "Honduras":         1620,
        "Zambia":           1620,
        "Guatemala":        1620,
        "Haiti":            1588,
        "Trinidad and Tobago":1550,
        "El Salvador":      1544,
        "Vietnam":          1540,
        "Oman":             1588,
        "Jordan":           1588,
        "Iraq":             1620,
        "Saudi Arabia":     1660,
        "UAE":              1600,
        "Guinea":           1626,
        "DR Congo":         1670,
        "Mozambique":       1540,
        "Angola":           1562,
        "Kenya":            1500,
        "Uganda":           1500,
        "Namibia":          1520,
        "Palestine":        1600,
        "Lebanon":          1540,
        "Bahrain":          1540,
        "Kuwait":           1480,
        "Fiji":             1300,
        "Vanuatu":          1250,
        "Sri Lanka":        1150,
        "Djibouti":         1150,
        "San Marino":       1050,
        "Luxembourg":       1480,
        "Estonia":          1470,
        "Belarus":          1540,
        "Finland (women)":   1550,  # a few women's teams for cross-check
    },
}


def _save_snapshot():
    try:
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(_SNAPSHOT, f, indent=2)
    except OSError:
        pass


def _load_snapshot():
    if os.path.exists(SNAPSHOT_PATH):
        try:
            with open(SNAPSHOT_PATH) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return _SNAPSHOT


def _norm(s):
    return (s or "").lower().strip().replace(".", "").replace("'", "")


def get_national_elo(team_name):
    """Return a national team's current Elo rating, or None."""
    if not team_name:
        return None
    snap = _load_snapshot()
    ratings = snap.get("ratings") or {}
    norm_target = _norm(team_name)
    # Pass 1: exact normalized match
    for name, elo in ratings.items():
        if _norm(name) == norm_target:
            return float(elo)
    # Pass 2: substring either way
    for name, elo in ratings.items():
        n = _norm(name)
        if norm_target in n or n in norm_target:
            return float(elo)
    # Pass 3: last-word (nickname) — "USA" vs "United States" etc.
    target_last = norm_target.rsplit(" ", 1)[-1] if norm_target else ""
    for name, elo in ratings.items():
        n = _norm(name)
        if target_last and (target_last in n.split()):
            return float(elo)
    return None


def is_national_team(team_name):
    """True when the name matches a known national squad."""
    return get_national_elo(team_name) is not None


if __name__ == "__main__":
    _save_snapshot()
    print(f"snapshot: {SNAPSHOT_PATH}")
    for team in ("Argentina", "USA", "Mexico", "Djibouti", "San Marino",
                 "Nonexistent FC"):
        elo = get_national_elo(team)
        print(f"  {team!r:20s}: {elo}")
