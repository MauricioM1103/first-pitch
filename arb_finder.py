#!/usr/bin/env python3
"""Arbitrage opportunity finder.

Scans Pinnacle + The Odds API books for any 2-way market where the best
available price on each side implies total probability < 1. When that
happens, splitting a bankroll between the two sides locks in a profit
regardless of outcome.

Covers:
  * Moneyline (every sport except 3-way soccer)
  * Total runs/goals/points (only when the two books have the same line)

Requires both sides to be at DIFFERENT books — you can't arb against
yourself. Soccer 3-way moneylines aren't covered (would need all three
prices lining up across up to three books).
"""
import mlb_odds
import generic_odds
import sports


def _two_way_arb(a_dec, a_book, b_dec, b_book, min_profit_pct=0.1):
    """Return arb payload if betting a_dec and b_dec locks in profit above threshold."""
    if a_dec is None or b_dec is None or a_dec <= 1.0 or b_dec <= 1.0:
        return None
    if a_book == b_book:
        return None  # need different books to arb
    implied = 1.0 / a_dec + 1.0 / b_dec
    if implied >= 1.0:
        return None
    profit_pct = (1.0 / implied - 1.0) * 100
    if profit_pct < min_profit_pct:
        return None
    return {
        "a_dec": a_dec, "a_book": a_book,
        "b_dec": b_dec, "b_book": b_book,
        "profit_pct": profit_pct,
        "a_stake_pct": (1.0 / a_dec) / implied,
        "b_stake_pct": (1.0 / b_dec) / implied,
    }


def _collect_moneyline_prices(pin_home_am, pin_away_am, oapi_books, oapi_home_name, oapi_away_name):
    """Return (home_candidates, away_candidates): list of (book, decimal)."""
    home = []
    away = []
    pin_h_dec = mlb_odds.american_to_decimal(pin_home_am)
    pin_a_dec = mlb_odds.american_to_decimal(pin_away_am)
    if pin_h_dec:
        home.append(("pinnacle", pin_h_dec))
    if pin_a_dec:
        away.append(("pinnacle", pin_a_dec))
    for bname, markets in (oapi_books or {}).items():
        for o in markets.get("h2h", []) or []:
            name = o.get("name")
            price = o.get("price")
            if price is None:
                continue
            if name == oapi_home_name:
                home.append((bname, price))
            elif name == oapi_away_name:
                away.append((bname, price))
    return home, away


def _collect_total_prices(pin_line, pin_over_am, pin_under_am, oapi_books):
    """Return (over_candidates, under_candidates) at the same line as Pinnacle."""
    over = []
    under = []
    if pin_line is None:
        return over, under
    pin_o_dec = mlb_odds.american_to_decimal(pin_over_am)
    pin_u_dec = mlb_odds.american_to_decimal(pin_under_am)
    if pin_o_dec:
        over.append(("pinnacle", pin_o_dec))
    if pin_u_dec:
        under.append(("pinnacle", pin_u_dec))
    for bname, markets in (oapi_books or {}).items():
        for o in markets.get("totals", []) or []:
            if abs((o.get("point") or 0) - pin_line) > 0.01:
                continue  # different line, can't arb
            side = (o.get("name") or "").lower()
            price = o.get("price")
            if price is None:
                continue
            if side == "over":
                over.append((bname, price))
            elif side == "under":
                under.append((bname, price))
    return over, under


def _best(lst):
    """Return (book, decimal) with the highest decimal odds."""
    if not lst:
        return (None, None)
    return max(lst, key=lambda x: x[1])


def _second_best_different_book(lst, exclude_book):
    """Fallback: next-best price at a different book than exclude_book."""
    filt = [x for x in lst if x[0] != exclude_book]
    if not filt:
        return (None, None)
    return max(filt, key=lambda x: x[1])


def _check_two_sided_arb(a_candidates, b_candidates):
    """Try best A × best B; if books collide, try next-best alternatives."""
    best_a = _best(a_candidates)
    best_b = _best(b_candidates)
    if not (best_a[0] and best_b[0]):
        return None
    if best_a[0] != best_b[0]:
        return _two_way_arb(best_a[1], best_a[0], best_b[1], best_b[0])
    # Books collide — try next-best on each side
    alt_a = _second_best_different_book(a_candidates, best_b[0])
    alt_b = _second_best_different_book(b_candidates, best_a[0])
    cands = []
    if alt_a[0]:
        cands.append((alt_a[1], alt_a[0], best_b[1], best_b[0]))
    if alt_b[0]:
        cands.append((best_a[1], best_a[0], alt_b[1], alt_b[0]))
    best = None
    for c in cands:
        arb = _two_way_arb(*c)
        if arb and (best is None or arb["profit_pct"] > best["profit_pct"]):
            best = arb
    return best


# ============================================================================
# MLB arb finder (uses mlb_odds dedicated structure)
# ============================================================================

def find_mlb_arbs():
    pin = mlb_odds.get_pinnacle_odds() or {}
    oapi = mlb_odds.get_odds_api_mlb() or {}
    arbs = []
    for (ak, hk), pin_game in pin.items():
        oapi_game = oapi.get((ak, hk)) or {}
        books = oapi_game.get("books") or {}
        home_name = pin_game.get("home_name", "")
        away_name = pin_game.get("away_name", "")

        # ---- Moneyline ----
        ml = pin_game.get("moneyline") or {}
        if ml.get("home_am") is not None and ml.get("away_am") is not None:
            home_prices, away_prices = _collect_moneyline_prices(
                ml["home_am"], ml["away_am"], books,
                oapi_game.get("home", home_name),
                oapi_game.get("away", away_name),
            )
            arb = _check_two_sided_arb(home_prices, away_prices)
            if arb:
                arbs.append({
                    "sport": "MLB", "game": f"{away_name} at {home_name}",
                    "market": "ML",
                    "pick_a": home_name, "pick_b": away_name,
                    "start_time": pin_game.get("start_time"),
                    **arb,
                })

        # ---- Total ----
        tot = pin_game.get("total") or {}
        if tot.get("line") is not None:
            over_prices, under_prices = _collect_total_prices(
                tot["line"], tot.get("over_am"), tot.get("under_am"), books,
            )
            arb = _check_two_sided_arb(over_prices, under_prices)
            if arb:
                arbs.append({
                    "sport": "MLB", "game": f"{away_name} at {home_name}",
                    "market": f"Total {tot['line']}",
                    "pick_a": f"Over {tot['line']}", "pick_b": f"Under {tot['line']}",
                    "start_time": pin_game.get("start_time"),
                    **arb,
                })
    return arbs


# ============================================================================
# Generic sport arb finder
# ============================================================================

def find_sport_arbs(sport):
    """Non-MLB sport. Soccer 3-way markets are skipped (would need all 3 sides)."""
    if sport.get("ml_outcomes") == 3:
        return []  # TODO: 3-way arb across up to 3 books
    games = generic_odds.parse_pinnacle_games(
        sport["pinnacle_league_id"],
        ml_outcomes=sport["ml_outcomes"],
        has_halves=False,  # only period-0 markets for arbs
    )
    oapi = generic_odds.fetch_odds_api(sport.get("odds_api_key"))
    arbs = []
    for g in games:
        home_name = g["home_name"]
        away_name = g["away_name"]
        g_oapi = (oapi or {}).get((generic_odds._team_key(away_name),
                                   generic_odds._team_key(home_name))) or {}
        books = g_oapi.get("books") or {}

        # Moneyline
        ml = g.get("ml") or {}
        if ml.get("home_am") is not None and ml.get("away_am") is not None:
            home_prices, away_prices = _collect_moneyline_prices(
                ml["home_am"], ml["away_am"], books,
                g_oapi.get("home", home_name),
                g_oapi.get("away", away_name),
            )
            arb = _check_two_sided_arb(home_prices, away_prices)
            if arb:
                arbs.append({
                    "sport": sport["name"],
                    "game": f"{away_name} at {home_name}",
                    "market": "ML",
                    "pick_a": home_name, "pick_b": away_name,
                    "start_time": g.get("start_time"),
                    **arb,
                })

        # Total
        tot = g.get("total") or {}
        if tot.get("line") is not None:
            over_prices, under_prices = _collect_total_prices(
                tot["line"], tot.get("over_am"), tot.get("under_am"), books,
            )
            arb = _check_two_sided_arb(over_prices, under_prices)
            if arb:
                arbs.append({
                    "sport": sport["name"],
                    "game": f"{away_name} at {home_name}",
                    "market": f"Total {tot['line']}",
                    "pick_a": f"Over {tot['line']}", "pick_b": f"Under {tot['line']}",
                    "start_time": g.get("start_time"),
                    **arb,
                })
    return arbs


def find_all_arbs():
    """Return every arbitrage opportunity across every sport, sorted by profit %."""
    arbs = []
    for sport in sports.SPORTS:
        try:
            if sport["slug"] == "mlb":
                arbs.extend(find_mlb_arbs())
            else:
                arbs.extend(find_sport_arbs(sport))
        except Exception as e:
            # One sport failing shouldn't nuke the whole scan
            print(f"[arb] {sport['slug']} failed: {e}", flush=True)
            continue
    arbs.sort(key=lambda x: -x["profit_pct"])
    return arbs
