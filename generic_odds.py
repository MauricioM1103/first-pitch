#!/usr/bin/env python3
"""Generic odds library for sports beyond MLB.

For each sport in sports.py, fetch Pinnacle lines (free, sharp) and compare
to US books via The Odds API (if ODDS_API_KEY is set). Positive-EV bets are
identified by comparing a book's price to Pinnacle's devigged "fair"
probability — a well-established model-free approach for sharp line-shopping.

Supports:
  * 2-way markets (NFL, UFC, soccer-when-no-draw)
  * 3-way markets (soccer: home/draw/away)
  * Full-game + 1st half markets (period 0 + period 1)
  * Main spread + total line selection via proximity to -110 vig

MLB has its own dedicated path (mlb_odds.py) with a model-driven EV.
"""
import json
import os
import time
from datetime import datetime
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mlb_odds import (
    american_to_decimal,
    decimal_to_american,
    american_to_prob,
    devig_two_sided,
    ev_percent,
    kelly_fraction,
    _team_key,
)

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

_cache: dict = {}


def _fetch_json(url, timeout=15):
    req = Request(url, headers={"User-Agent": _UA})
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
# Pinnacle fetch
# ============================================================================

def fetch_pinnacle_matchups(league_id):
    return _cached(("pin_mu", league_id), 300,
                   lambda: _fetch_json(f"{PINNACLE_BASE}/leagues/{league_id}/matchups"))


def fetch_pinnacle_markets(league_id):
    return _cached(("pin_mk", league_id), 300,
                   lambda: _fetch_json(f"{PINNACLE_BASE}/leagues/{league_id}/markets/straight"))


# ============================================================================
# Helpers for parsing Pinnacle market prices
# ============================================================================

def _price_designation(prices, designation):
    for p in prices:
        if p.get("designation") == designation:
            return p.get("price")
    return None


def _price_participant(prices, participant_id):
    for p in prices:
        if p.get("participantId") == participant_id:
            return p.get("price")
    return None


def _price_either(prices, designation, participant_id):
    v = _price_designation(prices, designation)
    if v is not None:
        return v
    return _price_participant(prices, participant_id)


def _max_limit(limits):
    return max([lim.get("amount", 0) for lim in limits]) if limits else None


def devig_three_way(p_h, p_d, p_a):
    """Multiplicative devig for 3-way (soccer home/draw/away) implied probs."""
    if None in (p_h, p_d, p_a):
        return None, None, None
    total = p_h + p_d + p_a
    if total <= 0:
        return None, None, None
    return p_h / total, p_d / total, p_a / total


def _score_main_line(c):
    """How close to -110 vig is this candidate (lower = main-line-ier)?"""
    pair = []
    for k in ("over_am", "under_am", "home_am", "away_am"):
        if k in c and c[k] is not None:
            pair.append(abs(abs(c[k]) - 110))
    return sum(pair) if pair else 10000


def _best_main(candidates):
    return min(candidates, key=_score_main_line) if candidates else None


# ============================================================================
# Parse one Pinnacle league into normalized games
# ============================================================================

def parse_pinnacle_games(league_id, ml_outcomes=2, has_halves=False):
    """Parse Pinnacle games for a league. league_id can be an int (single
    league) or a list (aggregate several leagues — e.g. Friendlies + Nations
    League under the "International" sport)."""
    league_ids = league_id if isinstance(league_id, (list, tuple)) else [league_id]
    matchups = []
    markets = []
    for lid in league_ids:
        try:
            matchups.extend(fetch_pinnacle_matchups(lid) or [])
            markets.extend(fetch_pinnacle_markets(lid) or [])
        except Exception:
            continue

    by_mu = {}
    for m in markets:
        by_mu.setdefault(m.get("matchupId"), []).append(m)

    # Prop-market variants Pinnacle returns alongside the real matchup
    # (e.g. "Team X (Corners)"). Skip them — we only want the main 90-min game.
    _PROP_MARKERS = ("(corners)", "(cards)", "(bookings)", "(fouls)",
                     "(offsides)", "(shots)", "(throw-ins)", "(saves)",
                     "(hits woodwork)")

    def _is_prop_participant(name):
        n = (name or "").lower()
        return any(m in n for m in _PROP_MARKERS)

    games = []
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
        if _is_prop_participant(home_name) or _is_prop_participant(away_name):
            continue
        home_id = home.get("id")
        away_id = away.get("id")
        mu_id = mu.get("id")

        entry = {
            "matchup_id": mu_id,
            "league_id": mu.get("league", {}).get("id") if isinstance(mu.get("league"), dict) else league_ids[0],
            "start_time": mu.get("startTime"),
            "is_live": mu.get("isLive"),
            "home_name": home_name,
            "away_name": away_name,
            # period 0
            "ml": None, "spread": None, "total": None,
            "ml_limit": None, "spread_limit": None, "total_limit": None,
            # period 1 (halves if has_halves)
            "ml_h1": None, "spread_h1": None, "total_h1": None,
            "ml_h1_limit": None,
        }

        tc0, sc0, tc1, sc1 = [], [], [], []

        for m in by_mu.get(mu_id, []):
            period = m.get("period")
            if period not in (0, 1):
                continue
            if period == 1 and not has_halves:
                continue
            typ = m.get("type")
            key = m.get("key", "")
            prices = m.get("prices") or []
            lim = _max_limit(m.get("limits") or [])

            if typ == "moneyline":
                h_am = _price_either(prices, "home", home_id)
                a_am = _price_either(prices, "away", away_id)
                d_am = _price_designation(prices, "draw") if ml_outcomes == 3 else None
                ml_data = {"home_am": h_am, "away_am": a_am, "draw_am": d_am}
                if period == 0:
                    entry["ml"] = ml_data
                    entry["ml_limit"] = lim
                else:
                    entry["ml_h1"] = ml_data
                    entry["ml_h1_limit"] = lim
            elif typ == "spread":
                try:
                    line_abs = abs(float(key.split(";")[-1]))
                except (ValueError, IndexError):
                    continue
                h_am = _price_either(prices, "home", home_id)
                a_am = _price_either(prices, "away", away_id)
                # home point (signed)
                h_pt = next((p.get("points") for p in prices if p.get("designation") == "home"), None)
                if h_pt is None:
                    h_pt = -line_abs if (h_am and h_am < a_am) else line_abs
                cand = {"line_home": h_pt, "home_am": h_am, "away_am": a_am, "limit": lim}
                (sc0 if period == 0 else sc1).append(cand)
            elif typ == "total":
                try:
                    line = float(key.split(";")[-1])
                except (ValueError, IndexError):
                    continue
                o_am = _price_designation(prices, "over")
                u_am = _price_designation(prices, "under")
                if o_am is None or u_am is None:
                    continue
                cand = {"line": line, "over_am": o_am, "under_am": u_am, "limit": lim}
                (tc0 if period == 0 else tc1).append(cand)

        if sc0:
            best = _best_main(sc0)
            entry["spread"] = {k: best[k] for k in ("line_home", "home_am", "away_am")}
            entry["spread_limit"] = best.get("limit")
        if sc1:
            best = _best_main(sc1)
            entry["spread_h1"] = {k: best[k] for k in ("line_home", "home_am", "away_am")}
        if tc0:
            best = _best_main(tc0)
            entry["total"] = {k: best[k] for k in ("line", "over_am", "under_am")}
            entry["total_limit"] = best.get("limit")
        if tc1:
            best = _best_main(tc1)
            entry["total_h1"] = {k: best[k] for k in ("line", "over_am", "under_am")}

        games.append(entry)

    games.sort(key=lambda g: g.get("start_time") or "")
    return games


# ============================================================================
# Odds API (any sport)
# ============================================================================

def odds_api_available():
    return bool(os.environ.get("ODDS_API_KEY"))


def fetch_odds_api(sport_key):
    if not odds_api_available() or not sport_key:
        return {}
    return _cached(("oapi", sport_key), 600, lambda: _fetch_odds_api(sport_key))


def _fetch_odds_api(sport_key):
    key = os.environ["ODDS_API_KEY"]
    # Soccer additionally supports btts (both teams to score). Odds API ignores
    # unsupported markets per sport, so including it is cheap.
    markets = "h2h,spreads,totals"
    if sport_key.startswith("soccer_"):
        markets += ",btts"
    params = urlencode({
        "apiKey": key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "decimal",
        "bookmakers": "draftkings,fanduel,betmgm,caesars",
    })
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds/?{params}"
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
                books[bname][mkt.get("key")] = mkt.get("outcomes") or []
        out[(_team_key(away), _team_key(home))] = {
            "home": home, "away": away, "commence_time": ev.get("commence_time"),
            "books": books,
        }
    return out


# ============================================================================
# Build sport games with EV computed vs Pinnacle devig
# ============================================================================

def _add_bet(bets, market, side, pick, fair_prob, pin_decimal, best_dec, best_book,
             limit=None, push_prob=0.0):
    """Add a bet entry per the best source price (prefer book over Pinnacle for EV)."""
    if fair_prob is None:
        return
    # Use best US book price if available (beats Pinnacle vig); fall back to Pinnacle
    use_dec = best_dec if (best_dec and best_dec > (pin_decimal or 0)) else pin_decimal
    use_source = best_book if (best_dec and best_dec > (pin_decimal or 0)) else "pinnacle"
    if not use_dec:
        return
    ev = fair_prob * use_dec - (1 - push_prob)
    kelly = kelly_fraction(fair_prob if push_prob == 0 else fair_prob / max(1e-9, 1 - push_prob),
                           use_dec, cap=0.25)
    bets.append({
        "market": market,
        "side": side,
        "pick": pick,
        "fair_prob": fair_prob,
        "pin_decimal": pin_decimal,
        "pin_american": decimal_to_american(pin_decimal) if pin_decimal else None,
        "book": use_source,
        "book_decimal": use_dec,
        "book_american": decimal_to_american(use_dec),
        "ev_pct": ev * 100,
        "kelly_pct": kelly * 100,
        "push_prob": push_prob,
        "limit": limit,
    })


def _best_book_price_for(side_key_fn, oapi_books, side_name_fn, side_key):
    """Walk books looking for best price on the named outcome.

    side_key_fn: callback given outcome -> True/False for the side we want
    """
    best_dec = None
    best_book = None
    for bname, markets_dict in (oapi_books or {}).items():
        for mkey, outcomes in markets_dict.items():
            for out in outcomes:
                if side_key_fn(out, mkey):
                    price = out.get("price")
                    if price and (best_dec is None or price > best_dec):
                        best_dec = price
                        best_book = bname
    return best_dec, best_book


def build_sport_games(sport, model_prob_fn=None):
    """For a non-MLB sport, return a list of game dicts with per-market odds
    and a bets list.

    model_prob_fn: optional callback (game_dict) -> dict with keys
        {home, away, draw} of win probabilities. When provided, ML "fair"
        probabilities come from the model instead of Pinnacle devig.
        Totals/spreads continue to use Pinnacle devig regardless.
    """
    games = parse_pinnacle_games(
        sport["pinnacle_league_id"],
        ml_outcomes=sport["ml_outcomes"],
        has_halves=sport.get("has_halves", False),
    )
    oapi = fetch_odds_api(sport.get("odds_api_key"))
    is_3way = sport["ml_outcomes"] == 3

    out = []
    for g in games:
        home_name = g["home_name"]
        away_name = g["away_name"]
        g_oapi = (oapi or {}).get((_team_key(away_name), _team_key(home_name)))
        books = (g_oapi or {}).get("books", {})

        bets = []
        home_fair = draw_fair = away_fair = None

        # ===== Full ML =====
        if g["ml"]:
            ml = g["ml"]
            h_dec = american_to_decimal(ml["home_am"])
            a_dec = american_to_decimal(ml["away_am"])
            d_dec = american_to_decimal(ml["draw_am"]) if ml.get("draw_am") else None

            if is_3way and d_dec:
                p_h = american_to_prob(ml["home_am"])
                p_d = american_to_prob(ml["draw_am"])
                p_a = american_to_prob(ml["away_am"])
                home_fair, draw_fair, away_fair = devig_three_way(p_h, p_d, p_a)
            else:
                p_h = american_to_prob(ml["home_am"])
                p_a = american_to_prob(ml["away_am"])
                home_fair, away_fair = devig_two_sided(p_h, p_a)

            # NOTE on model probabilities: we previously used the model's
            # win-prob to override Pinnacle devig as the "fair" input to EV.
            # In practice that produced misleading double-digit EV numbers on
            # longshot ML lines because small fair-prob shifts multiply by
            # large decimal odds. Pinnacle devig is a more reliable anchor
            # (it incorporates injuries/weather/sharp money). We still expose
            # the model via /sport/nfl/backtest and attach model_prob to each
            # game record so a dedicated "model view" could use it later.
            g["model_prob"] = model_prob_fn(g) if model_prob_fn else None

            # Odds API h2h outcome names come as team names
            def pick_name(side):
                return home_name if side == "home" else (away_name if side == "away" else "Draw")

            def side_match(outcome, mkey, team_name):
                return mkey == "h2h" and outcome.get("name") == team_name

            # Home ML
            best_h_dec, best_h_bk = _best_book_price_for(
                lambda o, mk: side_match(o, mk, home_name), books, pick_name, "home")
            _add_bet(bets, "ML", "home", home_name, home_fair, h_dec, best_h_dec, best_h_bk,
                     limit=g["ml_limit"])
            # Away ML
            best_a_dec, best_a_bk = _best_book_price_for(
                lambda o, mk: side_match(o, mk, away_name), books, pick_name, "away")
            _add_bet(bets, "ML", "away", away_name, away_fair, a_dec, best_a_dec, best_a_bk,
                     limit=g["ml_limit"])
            # Draw (3-way soccer)
            if is_3way and d_dec:
                best_d_dec, best_d_bk = _best_book_price_for(
                    lambda o, mk: mk == "h2h" and o.get("name") == "Draw", books, pick_name, "draw")
                _add_bet(bets, "ML", "draw", "Draw", draw_fair, d_dec, best_d_dec, best_d_bk,
                         limit=g["ml_limit"])

        # ===== Spread =====
        if g["spread"]:
            sp = g["spread"]
            h_dec = american_to_decimal(sp["home_am"])
            a_dec = american_to_decimal(sp["away_am"])
            if h_dec and a_dec:
                p_h = american_to_prob(sp["home_am"])
                p_a = american_to_prob(sp["away_am"])
                fair_h, fair_a = devig_two_sided(p_h, p_a)

                hpt = sp["line_home"]
                apt = -hpt if hpt is not None else None
                home_label = f"{home_name} {'' if (hpt or 0) < 0 else '+'}{hpt}"
                away_label = f"{away_name} {'' if (apt or 0) < 0 else '+'}{apt}"

                def spread_match(outcome, mkey, team_name, pt):
                    if mkey != "spreads" or outcome.get("name") != team_name:
                        return False
                    return abs((outcome.get("point") or 0) - (pt or 0)) < 0.01

                best_hd, best_hb = _best_book_price_for(
                    lambda o, mk: spread_match(o, mk, home_name, hpt), books, None, "home")
                _add_bet(bets, sport.get("spread_label", "Spread"), "home", home_label,
                         fair_h, h_dec, best_hd, best_hb, limit=g["spread_limit"])
                best_ad, best_ab = _best_book_price_for(
                    lambda o, mk: spread_match(o, mk, away_name, apt), books, None, "away")
                _add_bet(bets, sport.get("spread_label", "Spread"), "away", away_label,
                         fair_a, a_dec, best_ad, best_ab, limit=g["spread_limit"])

        # ===== Total =====
        if g["total"]:
            t = g["total"]
            o_dec = american_to_decimal(t["over_am"])
            u_dec = american_to_decimal(t["under_am"])
            if o_dec and u_dec:
                p_o = american_to_prob(t["over_am"])
                p_u = american_to_prob(t["under_am"])
                fair_o, fair_u = devig_two_sided(p_o, p_u)

                def total_match(outcome, mkey, side_name, line):
                    if mkey != "totals" or (outcome.get("name") or "").lower() != side_name:
                        return False
                    return abs((outcome.get("point") or 0) - (line or 0)) < 0.01

                best_od, best_ob = _best_book_price_for(
                    lambda o, mk: total_match(o, mk, "over", t["line"]), books, None, "over")
                _add_bet(bets, "Total", "over", f"Over {t['line']}",
                         fair_o, o_dec, best_od, best_ob, limit=g["total_limit"])
                best_ud, best_ub = _best_book_price_for(
                    lambda o, mk: total_match(o, mk, "under", t["line"]), books, None, "under")
                _add_bet(bets, "Total", "under", f"Under {t['line']}",
                         fair_u, u_dec, best_ud, best_ub, limit=g["total_limit"])

        # ===== 1H markets (NFL + soccer halves) =====
        if g.get("ml_h1"):
            ml = g["ml_h1"]
            h_dec = american_to_decimal(ml["home_am"])
            a_dec = american_to_decimal(ml["away_am"])
            d_dec = american_to_decimal(ml["draw_am"]) if ml.get("draw_am") else None
            if is_3way and d_dec:
                p_h = american_to_prob(ml["home_am"])
                p_d = american_to_prob(ml["draw_am"])
                p_a = american_to_prob(ml["away_am"])
                fh, fd, fa = devig_three_way(p_h, p_d, p_a)
            else:
                p_h = american_to_prob(ml["home_am"])
                p_a = american_to_prob(ml["away_am"])
                fh, fa = devig_two_sided(p_h, p_a)
                fd = None
            _add_bet(bets, "1H ML", "home", f"{home_name} (1H)", fh, h_dec, None, None,
                     limit=g["ml_h1_limit"])
            _add_bet(bets, "1H ML", "away", f"{away_name} (1H)", fa, a_dec, None, None,
                     limit=g["ml_h1_limit"])
            if fd:
                _add_bet(bets, "1H ML", "draw", "Draw (1H)", fd, d_dec, None, None,
                         limit=g["ml_h1_limit"])

        if g.get("spread_h1"):
            sp = g["spread_h1"]
            h_dec = american_to_decimal(sp["home_am"])
            a_dec = american_to_decimal(sp["away_am"])
            if h_dec and a_dec:
                p_h = american_to_prob(sp["home_am"])
                p_a = american_to_prob(sp["away_am"])
                fh, fa = devig_two_sided(p_h, p_a)
                hpt = sp["line_home"]; apt = -hpt if hpt is not None else None
                _add_bet(bets, "1H Spread", "home",
                         f"{home_name} {'' if (hpt or 0) < 0 else '+'}{hpt} (1H)",
                         fh, h_dec, None, None)
                _add_bet(bets, "1H Spread", "away",
                         f"{away_name} {'' if (apt or 0) < 0 else '+'}{apt} (1H)",
                         fa, a_dec, None, None)

        if g.get("total_h1"):
            t = g["total_h1"]
            o_dec = american_to_decimal(t["over_am"])
            u_dec = american_to_decimal(t["under_am"])
            if o_dec and u_dec:
                p_o = american_to_prob(t["over_am"])
                p_u = american_to_prob(t["under_am"])
                fo, fu = devig_two_sided(p_o, p_u)
                _add_bet(bets, "1H Total", "over", f"Over {t['line']} (1H)",
                         fo, o_dec, None, None)
                _add_bet(bets, "1H Total", "under", f"Under {t['line']} (1H)",
                         fu, u_dec, None, None)

        # ===== BTTS (soccer only) — Both Teams To Score =====
        # Only produced for soccer sports that pass btts through the Odds API.
        if sport.get("ml_outcomes") == 3 and books:
            btts_by_book = {}
            for bname, markets_dict in books.items():
                btts_outcomes = markets_dict.get("btts") or []
                for out_item in btts_outcomes:
                    name = (out_item.get("name") or "").strip().lower()
                    price = out_item.get("price")
                    if name in ("yes", "no") and price:
                        btts_by_book.setdefault(bname, {})[name] = price
            if btts_by_book:
                # Best book price per side
                best_yes_dec = best_no_dec = None
                best_yes_book = best_no_book = None
                for bname, prices in btts_by_book.items():
                    y = prices.get("yes")
                    n = prices.get("no")
                    if y and (best_yes_dec is None or y > best_yes_dec):
                        best_yes_dec, best_yes_book = y, bname
                    if n and (best_no_dec is None or n > best_no_dec):
                        best_no_dec, best_no_book = n, bname

                # Model probability via soccer model (fall back to market devig)
                btts_fair_yes = btts_fair_no = None
                if sport["slug"] in {"epl", "laliga", "ligamx"}:
                    try:
                        import soccer_model
                        btts_p = soccer_model.predict_btts(
                            home_name, away_name, sport["slug"]
                        )
                        btts_fair_yes = btts_p["yes"]
                        btts_fair_no = btts_p["no"]
                    except Exception:
                        pass
                if btts_fair_yes is None and best_yes_dec and best_no_dec:
                    # Market devig fallback
                    p_y_raw = 1.0 / best_yes_dec
                    p_n_raw = 1.0 / best_no_dec
                    s = p_y_raw + p_n_raw
                    btts_fair_yes = p_y_raw / s
                    btts_fair_no = p_n_raw / s

                if best_yes_dec and btts_fair_yes is not None:
                    _add_bet(bets, "BTTS", "yes", "BTTS Yes",
                             btts_fair_yes, None, best_yes_dec, best_yes_book)
                if best_no_dec and btts_fair_no is not None:
                    _add_bet(bets, "BTTS", "no", "BTTS No",
                             btts_fair_no, None, best_no_dec, best_no_book)

        out.append({**g, "bets": bets,
                    "fair": {"home": home_fair, "draw": draw_fair, "away": away_fair}})
    return out
