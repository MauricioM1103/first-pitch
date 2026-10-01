#!/usr/bin/env python3
"""Betting-odds integration for First Pitch.

Sources (verified reachable, free unless noted):
  * Pinnacle guest API   — moneyline/spread/total/team-total per MLB game.
                           Pinnacle is the industry-sharpest book; devigged
                           Pinnacle prices approximate the true market
                           probability closer than any US-facing retail book.
  * Polymarket Gamma     — MLB futures (WS champion, awards) and some
                           single-game markets. Decentralized prediction
                           market — useful signal, limited game coverage.
  * The Odds API         — DK, FanDuel, BetMGM, Caesars main lines.
                           Requires paid key via ODDS_API_KEY env var.
                           ~$30/mo Starter tier = 20k requests, enough for
                           hourly polls over a season. Not called when key
                           is missing; adapter is scaffolded for later.

Math:
  * American ↔ Decimal ↔ Implied probability
  * Multiplicative two-sided devig (fair odds removal)
  * EV = p·d - 1  for a $1 unit bet at decimal odds d given model prob p
  * Kelly fraction = (bp - q) / b  where b = d - 1

Team-name matching: canonical statsapi full names ("Atlanta Braves"), case
and whitespace normalized. Pinnacle uses the same names; Polymarket titles
are parsed with team-term regexes.
"""
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
PINNACLE_LEAGUE_ID = 246  # MLB
POLYMARKET_BASE = "https://gamma-api.polymarket.com"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

_cache: dict = {}
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


# ============================================================================
# Odds math
# ============================================================================

def american_to_decimal(am):
    """American odds (e.g. -150, +170) → decimal odds (e.g. 1.667, 2.70)."""
    if am is None:
        return None
    am = float(am)
    if am >= 100:
        return 1 + am / 100.0
    if am <= -100:
        return 1 + 100.0 / abs(am)
    return None  # invalid American odds between -99 and +99


def decimal_to_american(d):
    if d is None or d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1) * 100))
    return int(round(-100 / (d - 1)))


def american_to_prob(am):
    """Implied probability (with the book's vig still in it)."""
    d = american_to_decimal(am)
    return 1.0 / d if d else None


def decimal_to_prob(d):
    if d is None or d <= 0:
        return None
    return 1.0 / d


def devig_two_sided(p_a, p_b):
    """Multiplicative devig: fair probs that sum to 1."""
    if p_a is None or p_b is None:
        return None, None
    total = p_a + p_b
    if total <= 0:
        return None, None
    return p_a / total, p_b / total


def ev_percent(model_prob, decimal_odds):
    """Return EV as a percentage of unit bet (e.g. +4.8 means +4.8% EV)."""
    if model_prob is None or decimal_odds is None:
        return None
    return (model_prob * decimal_odds - 1.0) * 100.0


def kelly_fraction(model_prob, decimal_odds, cap=0.25):
    """Full Kelly fraction, floor 0, cap at `cap` (default quarter Kelly)."""
    if model_prob is None or decimal_odds is None or decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    f = (b * model_prob - (1.0 - model_prob)) / b
    return max(0.0, min(cap, f))


# ============================================================================
# Team-name normalization
# ============================================================================

# Canonical MLB team names as used by statsapi; Pinnacle uses the same.
MLB_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles",
    "Boston Red Sox", "Chicago Cubs", "Chicago White Sox",
    "Cincinnati Reds", "Cleveland Guardians", "Colorado Rockies",
    "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins",
    "Milwaukee Brewers", "Minnesota Twins", "New York Mets",
    "New York Yankees", "Athletics", "Oakland Athletics",
    "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres",
    "San Francisco Giants", "Seattle Mariners", "St. Louis Cardinals",
    "Tampa Bay Rays", "Texas Rangers", "Toronto Blue Jays",
    "Washington Nationals",
]


def _normalize_team(name):
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    # Oakland Athletics vs Athletics canonicalization — treat as same team
    if s == "oakland athletics":
        s = "athletics"
    return s


def _team_key(team_name):
    return _normalize_team(team_name)


# ============================================================================
# Fetch primitives
# ============================================================================

def _fetch_json(url, headers=None, timeout=15):
    req = Request(url, headers=headers or {"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _cached(key, ttl, fetch_fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[1] < ttl:
        return hit[0]
    try:
        v = fetch_fn()
    except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
        v = None
    _cache[key] = (v, now)
    return v


# ============================================================================
# Pinnacle
# ============================================================================

def _fetch_pinnacle_matchups():
    return _fetch_json(f"{PINNACLE_BASE}/leagues/{PINNACLE_LEAGUE_ID}/matchups")


def _fetch_pinnacle_straight_markets():
    return _fetch_json(f"{PINNACLE_BASE}/leagues/{PINNACLE_LEAGUE_ID}/markets/straight")


def _fetch_pinnacle_related(matchup_id):
    return _fetch_json(f"{PINNACLE_BASE}/matchups/{matchup_id}/related")


def get_pinnacle_odds():
    """Return dict keyed by (away_team_norm, home_team_norm) with full odds payload.

    Each entry contains moneyline, run_line, total, team_totals, and limits.
    """
    return _cached(("pinnacle",), 300, _get_pinnacle_odds)


def _extract_total_line(key):
    """Parse '7.5' from 's;0;ou;7.5' / 's;1;ou;4.5'."""
    try:
        return float(key.split(";")[-1])
    except (ValueError, IndexError):
        return None


def _extract_spread_line(key):
    """Parse '1.5' from 's;0;s;1.5'. Pinnacle lists as magnitude; sign comes from
    participant via home_am vs away_am (favorite has negative price)."""
    try:
        return float(key.split(";")[-1])
    except (ValueError, IndexError):
        return None


def _best_main_total(candidates):
    """Pick the 'main' total line from a list of candidates: the one whose
    over/under prices are closest to -110 (least sided). Fallback: first."""
    if not candidates:
        return None
    def score(c):
        o, u = c.get("over_am"), c.get("under_am")
        if o is None or u is None:
            return 10000
        return abs(abs(o) - 110) + abs(abs(u) - 110)
    return min(candidates, key=score)


def _get_pinnacle_odds():
    matchups = _fetch_pinnacle_matchups() or []
    markets = _fetch_pinnacle_straight_markets() or []

    by_mu = {}
    for m in markets:
        by_mu.setdefault(m.get("matchupId"), []).append(m)

    out = {}
    for mu in matchups:
        if mu.get("type") != "matchup":
            continue
        parts = mu.get("participants") or []
        if len(parts) < 2:
            continue
        home = next((p for p in parts if p.get("alignment") == "home"), None)
        away = next((p for p in parts if p.get("alignment") == "away"), None)
        if not home or not away:
            away, home = parts[0], parts[1]

        home_name = home.get("name", "")
        away_name = away.get("name", "")
        home_id = home.get("id")
        away_id = away.get("id")
        mu_id = mu.get("id")
        mkts = by_mu.get(mu_id, [])

        entry = {
            "matchup_id": mu_id,
            "start_time": mu.get("startTime"),
            "is_live": mu.get("isLive"),
            "home_name": home_name, "away_name": away_name,
            # period 0 (full game) — key "moneyline" kept for backward-compat with schedule card
            "moneyline": None, "total": None, "run_line": None,
            "ml_limit": None, "total_limit": None,
            # period 1 (first 5 innings)
            "ml_f5": None, "total_f5": None,
            "ml_limit_f5": None, "total_limit_f5": None,
        }

        total_full_candidates = []
        total_f5_candidates = []

        for m in mkts:
            period = m.get("period")
            if period not in (0, 1):
                continue
            key = m.get("key", "")
            typ = m.get("type")
            prices = m.get("prices", []) or []
            limits = m.get("limits", []) or []
            max_limit = max([lim.get("amount", 0) for lim in limits]) if limits else None

            if typ == "moneyline":
                hp = next((p.get("price") for p in prices if p.get("participantId") == home_id), None)
                ap = next((p.get("price") for p in prices if p.get("participantId") == away_id), None)
                if period == 0:
                    entry["moneyline"] = {"home_am": hp, "away_am": ap}
                    entry["ml_limit"] = max_limit
                else:
                    entry["ml_f5"] = {"home_am": hp, "away_am": ap}
                    entry["ml_limit_f5"] = max_limit
            elif typ == "total":
                over = next((p.get("price") for p in prices if p.get("designation") == "over"), None)
                under = next((p.get("price") for p in prices if p.get("designation") == "under"), None)
                line = _extract_total_line(key)
                if line is None or over is None or under is None:
                    continue
                cand = {"line": line, "over_am": over, "under_am": under, "limit": max_limit}
                if period == 0:
                    total_full_candidates.append(cand)
                else:
                    total_f5_candidates.append(cand)
            elif typ == "spread" and period == 0:
                line = _extract_spread_line(key)
                if line != 1.5:
                    continue
                hp = next((p.get("price") for p in prices if p.get("participantId") == home_id), None)
                ap = next((p.get("price") for p in prices if p.get("participantId") == away_id), None)
                # home_am being negative means home is favored (-1.5 for home)
                entry["run_line"] = {
                    "line": line, "home_am": hp, "away_am": ap,
                    "limit": max_limit,
                }

        # Pick main total lines by proximity to -110 vig
        main_full = _best_main_total(total_full_candidates)
        if main_full:
            entry["total"] = {
                "line": main_full["line"],
                "over_am": main_full["over_am"],
                "under_am": main_full["under_am"],
            }
            entry["total_limit"] = main_full.get("limit")
        main_f5 = _best_main_total(total_f5_candidates)
        if main_f5:
            entry["total_f5"] = {
                "line": main_f5["line"],
                "over_am": main_f5["over_am"],
                "under_am": main_f5["under_am"],
            }
            entry["total_limit_f5"] = main_f5.get("limit")

        key = (_team_key(away_name), _team_key(home_name))
        out[key] = entry
    return out


# ============================================================================
# Polymarket (MLB-tagged events)
# ============================================================================

def get_polymarket_mlb():
    return _cached(("polymarket",), 1800, _get_polymarket_mlb)


def _get_polymarket_mlb():
    url = f"{POLYMARKET_BASE}/events?active=true&closed=false&limit=100&tag_slug=mlb&order=-volume"
    events = _fetch_json(url) or []
    results = {"futures": [], "single_game": []}
    for e in events:
        title = (e.get("title") or "").strip()
        slug = (e.get("slug") or "").strip()
        end_date = e.get("endDate", "")
        markets = e.get("markets") or []
        if not markets:
            continue
        # Classify as single-game vs. futures
        is_single_game = bool(re.search(r"\bvs\.?\b|game [0-9]+\s", title, re.I)) and \
                         not re.search(r"series|champion|division|mvp|cy young|rookie", title, re.I)
        pretty_markets = []
        for m in markets[:40]:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                prices = json.loads(m.get("outcomePrices") or "[]")
            except (json.JSONDecodeError, TypeError):
                outs, prices = [], []
            if not outs or len(outs) != len(prices):
                continue
            try:
                prices_f = [float(p) for p in prices]
            except (ValueError, TypeError):
                continue
            pretty_markets.append({
                "question": (m.get("question") or "").strip(),
                "outcomes": outs,
                "probs": prices_f,
                "volume": m.get("volume"),
                "liquidity": m.get("liquidity"),
            })
        event_payload = {
            "title": title, "slug": slug, "end_date": end_date,
            "markets": pretty_markets,
        }
        if is_single_game:
            results["single_game"].append(event_payload)
        else:
            results["futures"].append(event_payload)
    return results


# ============================================================================
# The Odds API — gated on ODDS_API_KEY env var
# ============================================================================

def odds_api_available():
    return bool(os.environ.get("ODDS_API_KEY"))


def get_odds_api_mlb():
    """Return odds from DK/FanDuel/BetMGM via The Odds API. Empty if no key."""
    if not odds_api_available():
        return {}
    return _cached(("oddsapi",), 600, _get_odds_api_mlb)


def _get_odds_api_mlb():
    key = os.environ["ODDS_API_KEY"]
    params = urlencode({
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    })
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds/?{params}"
    try:
        events = _fetch_json(url) or []
    except Exception:
        return {}
    out = {}
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        books = {}
        for bk in ev.get("bookmakers") or []:
            bname = bk.get("key")
            books[bname] = {}
            for mkt in bk.get("markets") or []:
                mkey = mkt.get("key")
                outcomes = mkt.get("outcomes") or []
                books[bname][mkey] = outcomes
        out[(_team_key(away), _team_key(home))] = {
            "home": home, "away": away, "commence_time": ev.get("commence_time"),
            "books": books,
        }
    return out


# ============================================================================
# Game-level odds assembly
# ============================================================================

def _add_two_sided_bets(bets, market, side_data, limit, decimal_only=False):
    """side_data: dict of side_key -> (label, american, model_prob, fair_prob, push_prob).

    Appends a bet entry per side (keeping negative-EV too; filter at display).
    """
    for side_key, (label, am, model_p, fair_p, push_p) in side_data.items():
        if am is None or model_p is None:
            continue
        dec = american_to_decimal(am)
        if not dec:
            continue
        push_p = push_p or 0.0
        # EV = p_win * d - (1 - p_push). For push_p=0 this reduces to p*d - 1.
        ev = model_p * dec - (1 - push_p)
        # Kelly: use conditional (no-push) prob when ties push
        p_eff = model_p if push_p == 0 else model_p / max(1e-9, 1 - push_p)
        kelly = kelly_fraction(p_eff, dec, cap=0.25)
        bets.append({
            "market": market,
            "side": side_key,
            "pick": label,
            "model_prob": model_p,
            "fair_prob": fair_p,
            "american": am,
            "decimal": dec,
            "ev_pct": ev * 100,
            "kelly_pct": kelly * 100,
            "push_prob": push_p,
            "limit": limit,
        })


def build_game_odds(away_team, home_team, model_prob_home=None, game_ctx=None):
    """Join every source and compute EV across all markets for the matchup.

    `game_ctx` carries the inputs needed by the totals/RL/F5 models:
        home_rpg, away_rpg, home_team_era, away_team_era,
        home_sp_era, home_sp_ip, away_sp_era, away_sp_ip
    """
    import mlb_model as M
    key = (_team_key(away_team), _team_key(home_team))

    pin = (get_pinnacle_odds() or {}).get(key)
    oapi = (get_odds_api_mlb() or {}).get(key)

    # ------- Pinnacle moneyline (full game) -------
    pin_home_fair = pin_away_fair = None
    pin_home_dec = pin_away_dec = None
    if pin and pin.get("moneyline"):
        ml = pin["moneyline"]
        p_home_raw = american_to_prob(ml.get("home_am"))
        p_away_raw = american_to_prob(ml.get("away_am"))
        pin_home_fair, pin_away_fair = devig_two_sided(p_home_raw, p_away_raw)
        pin_home_dec = american_to_decimal(ml.get("home_am"))
        pin_away_dec = american_to_decimal(ml.get("away_am"))

    ev_home_pin = ev_away_pin = None
    kelly_home_pin = kelly_away_pin = 0.0
    if model_prob_home is not None and pin_home_dec:
        ev_home_pin = ev_percent(model_prob_home, pin_home_dec)
        kelly_home_pin = kelly_fraction(model_prob_home, pin_home_dec)
    if model_prob_home is not None and pin_away_dec:
        model_prob_away = 1 - model_prob_home
        ev_away_pin = ev_percent(model_prob_away, pin_away_dec)
        kelly_away_pin = kelly_fraction(model_prob_away, pin_away_dec)

    # ------- The Odds API (best US book, full ML + spreads + totals) -------
    best_books = {"home": None, "away": None}
    best_decimal = {"home": None, "away": None}
    if oapi:
        for bname, markets in (oapi.get("books") or {}).items():
            h2h = markets.get("h2h") or []
            for outcome in h2h:
                side = "home" if outcome.get("name") == oapi.get("home") else "away" if outcome.get("name") == oapi.get("away") else None
                if not side:
                    continue
                price = outcome.get("price")
                if price and (best_decimal[side] is None or price > best_decimal[side]):
                    best_decimal[side] = price
                    best_books[side] = bname
    ev_home_book = ev_away_book = None
    if model_prob_home is not None and best_decimal["home"]:
        ev_home_book = ev_percent(model_prob_home, best_decimal["home"])
    if model_prob_home is not None and best_decimal["away"]:
        ev_away_book = ev_percent(1 - model_prob_home, best_decimal["away"])

    # ======= BETS LIST — all markets, each side =======
    bets = []

    # ---- Full ML ----
    if pin and pin.get("moneyline") and model_prob_home is not None:
        ml = pin["moneyline"]
        _add_two_sided_bets(bets, "ML", {
            "away": (away_team, ml.get("away_am"), 1 - model_prob_home, pin_away_fair, 0.0),
            "home": (home_team, ml.get("home_am"), model_prob_home, pin_home_fair, 0.0),
        }, limit=pin.get("ml_limit"))

    # ---- Full Total ----
    if pin and pin.get("total") and game_ctx:
        t = pin["total"]
        pred = M.predict_full_total(game_ctx, t["line"])
        if pred:
            # devig
            p_over_raw = american_to_prob(t.get("over_am"))
            p_under_raw = american_to_prob(t.get("under_am"))
            fair_o, fair_u = devig_two_sided(p_over_raw, p_under_raw)
            _add_two_sided_bets(bets, "Total", {
                "over":  (f"Over {t['line']}",  t.get("over_am"),  pred["p_over"],  fair_o, pred["p_push"]),
                "under": (f"Under {t['line']}", t.get("under_am"), pred["p_under"], fair_u, pred["p_push"]),
            }, limit=pin.get("total_limit"))

    # ---- Run Line (home -1.5 / away +1.5) ----
    if pin and pin.get("run_line") and game_ctx:
        rl = pin["run_line"]
        pred = M.predict_run_line(game_ctx, line=1.5)
        if pred:
            p_h_raw = american_to_prob(rl.get("home_am"))
            p_a_raw = american_to_prob(rl.get("away_am"))
            fair_h, fair_a = devig_two_sided(p_h_raw, p_a_raw)
            _add_two_sided_bets(bets, "Run Line", {
                "home": (f"{home_team} -1.5", rl.get("home_am"), pred["p_home_covers"], fair_h, 0.0),
                "away": (f"{away_team} +1.5", rl.get("away_am"), pred["p_away_covers"], fair_a, 0.0),
            }, limit=rl.get("limit"))

    # ---- F5 Moneyline (2-way with tie push) ----
    if pin and pin.get("ml_f5") and game_ctx:
        f5ml = pin["ml_f5"]
        pred = M.predict_f5_moneyline(game_ctx)
        if pred:
            p_h_raw = american_to_prob(f5ml.get("home_am"))
            p_a_raw = american_to_prob(f5ml.get("away_am"))
            fair_h, fair_a = devig_two_sided(p_h_raw, p_a_raw)
            _add_two_sided_bets(bets, "F5 ML", {
                "home": (f"{home_team} (F5)", f5ml.get("home_am"), pred["p_home"], fair_h, pred["p_tie"]),
                "away": (f"{away_team} (F5)", f5ml.get("away_am"), pred["p_away"], fair_a, pred["p_tie"]),
            }, limit=pin.get("ml_limit_f5"))

    # ---- F5 Total ----
    if pin and pin.get("total_f5") and game_ctx:
        t5 = pin["total_f5"]
        pred = M.predict_f5_total(game_ctx, t5["line"])
        if pred:
            p_o_raw = american_to_prob(t5.get("over_am"))
            p_u_raw = american_to_prob(t5.get("under_am"))
            fair_o, fair_u = devig_two_sided(p_o_raw, p_u_raw)
            _add_two_sided_bets(bets, "F5 Total", {
                "over":  (f"Over {t5['line']} (F5)",  t5.get("over_am"),  pred["p_over"],  fair_o, pred["p_push"]),
                "under": (f"Under {t5['line']} (F5)", t5.get("under_am"), pred["p_under"], fair_u, pred["p_push"]),
            }, limit=pin.get("total_limit_f5"))

    return {
        "pinnacle": pin,
        "pin_fair": {"home": pin_home_fair, "away": pin_away_fair} if pin_home_fair else None,
        "pin_decimal": {"home": pin_home_dec, "away": pin_away_dec},
        "ev_pinnacle": {"home": ev_home_pin, "away": ev_away_pin},
        "kelly_pinnacle": {"home": kelly_home_pin, "away": kelly_away_pin},
        "odds_api": oapi,
        "best_book": best_books,
        "best_decimal": best_decimal,
        "ev_book": {"home": ev_home_book, "away": ev_away_book},
        "odds_api_available": odds_api_available(),
        "bets": bets,
    }


# ============================================================================
# CLI quickcheck
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("Fetching Pinnacle MLB odds...")
    pin = get_pinnacle_odds()
    print(f"  games: {len(pin)}")
    for (ak, hk), g in list(pin.items())[:5]:
        ml = g.get("moneyline") or {}
        print(f"  {g['away_name']} @ {g['home_name']}  "
              f"ml away={ml.get('away_am')} home={ml.get('home_am')}  "
              f"limit=${g.get('ml_limit')}")

    print()
    print("Fetching Polymarket MLB events...")
    pm = get_polymarket_mlb()
    print(f"  futures: {len(pm.get('futures', []))}  single-game: {len(pm.get('single_game', []))}")

    print()
    print(f"Odds API key present: {odds_api_available()}")
    if odds_api_available():
        oa = get_odds_api_mlb()
        print(f"  games: {len(oa)}")
