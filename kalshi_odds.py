#!/usr/bin/env python3
"""Kalshi event-market cross-check for sports moneylines.

Kalshi lists game-outcome contracts (ticker series KXNFLGAME, KXMLBGAME,
KXNBAGAME, KXNHLGAME) that trade in cents 1-99. 65c yes-ask = the market
is pricing a 65% chance of yes. We treat the yes-ask as "fair prob" and
compare to Pinnacle's devigged number the same way we treat DraftKings.

The public /trade-api/v2/markets endpoint is readable without auth.
Sports game markets exist year-round but have very thin liquidity outside
the Super Bowl / playoffs — many open markets never trade, so yes_bid /
yes_ask / last_price come back None. This module exposes None gracefully
in that case and the UI simply doesn't render the chip. When liquidity
shows up, the same fetch starts returning real numbers and the chip
appears without any further code change.
"""
import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_UA   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
_TTL_S = 5 * 60   # 5 min cache; sports books don't move faster than that

# Kalshi series ticker prefixes by our sport slug. Each sport maps to
# the base per-game series (ML market). Soccer leagues share the UEFA
# aggregate series for UCL / Europa / international friendlies.
#
# Phase 22b: extended beyond US-sport MLs to cover our top-performing
# soccer leagues (EPL, LaLiga, Liga MX are each +15-19% ROI on 35+
# picks in the 90-day log) and UEFA competitions. All use the same
# yes_sub_title "home team wins" encoding so no per-sport parsing
# changes are needed.
_SERIES_PREFIX = {
    "nfl":    "KXNFLGAME",
    "mlb":    "KXMLBGAME",
    "nba":    "KXNBAGAME",
    "nhl":    "KXNHLGAME",
    "ncaaf":  "KXCFBGAME",
    # Soccer leagues (Phase 22b)
    "epl":    "KXEPLGAME",
    "laliga": "KXLALIGAGAME",
    "ligamx": "KXLIGAMXGAME",
    "ucl":    "KXUEFAGAME",          # UEFA covers Champions League matches
    "europa": "KXUEFAGAME",          # same aggregate
    "international": "KXUEFANLGAME", # UEFA Nations League approx
}

_cache = {}  # (sport_slug) -> (ts, [markets])


def _fetch_json(url, timeout=15):
    req = Request(url, headers={
        "User-Agent": _UA,
        "Accept": "application/json",
    })
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _series_markets(sport_slug):
    """Return the current list of Kalshi game markets for this sport, with
    caching. Empty list if the sport isn't in our map or the fetch fails
    — callers never see an exception."""
    pfx = _SERIES_PREFIX.get(sport_slug)
    if not pfx:
        return []
    now = time.time()
    hit = _cache.get(sport_slug)
    if hit and (now - hit[0]) < _TTL_S:
        return hit[1]
    try:
        d = _fetch_json(
            f"{_BASE}/markets?limit=200&status=open&series_ticker={pfx}",
            timeout=20,
        )
        markets = d.get("markets") or []
    except (URLError, ValueError, OSError, TimeoutError, ConnectionError):
        markets = []
    _cache[sport_slug] = (now, markets)
    return markets


def _norm_team(name):
    """Lowercase + strip punctuation to compare 'Vegas' with 'Vegas Golden Knights'."""
    if not name:
        return ""
    s = (name or "").lower()
    for ch in ",.-'":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def moneyline_yes_prob(sport_slug, home_team, away_team):
    """Return Kalshi's implied Yes probability (0..1) for the HOME team to
    win the game, or None when no Kalshi market exists / no trading.

    Kalshi encodes one side per market (title "Philadelphia wins"), so we
    find the market whose yes_sub_title matches our home team and read its
    yes_ask price (cents 1..99 → probability 0.01..0.99). yes_ask over
    yes_bid is intentional: we want the market-maker's offer price, i.e.
    "what Kalshi says Yes is worth right now". When the book is empty we
    fall back to last_price; if that's also None we return None.
    """
    pfx = _SERIES_PREFIX.get(sport_slug)
    if not pfx:
        return None
    markets = _series_markets(sport_slug)
    if not markets:
        return None
    home_norm = _norm_team(home_team)
    away_norm = _norm_team(away_team)
    for m in markets:
        # Yes side encodes a single team as the winner. We want the market
        # whose yes_sub_title is our HOME team AND whose no_sub_title
        # mentions our away team (or whose ticker contains both abbrs).
        yes_team = _norm_team(m.get("yes_sub_title"))
        no_team  = _norm_team(m.get("no_sub_title"))
        if not yes_team:
            continue
        # Loose match: Kalshi often uses the city only ("Vegas") while we
        # may have "Vegas Golden Knights". Match on substring either way.
        home_matches_yes = (
            yes_team == home_norm
            or yes_team in home_norm
            or home_norm in yes_team
        )
        if not home_matches_yes:
            continue
        # Prefer yes_ask (offer) → fall back to yes_bid → last traded.
        for key in ("yes_ask", "yes_bid", "last_price"):
            v = m.get(key)
            if v is None:
                continue
            try:
                cents = float(v)
            except (TypeError, ValueError):
                continue
            if cents <= 0 or cents >= 100:
                continue
            return cents / 100.0
        return None
    return None


def moneyline_fair_probs(sport_slug, games):
    """Bulk variant for picks board. `games` is iterable of (home, away)
    tuples; returns {(home, away): prob_home or None}. Thin wrapper over
    the single-market lookup, kept for API parity with DraftKings."""
    out = {}
    for home, away in games:
        try:
            p = moneyline_yes_prob(sport_slug, home, away)
        except Exception:
            p = None
        if p is not None:
            out[(home, away)] = p
    return out


# ---------------------------------------------------------------------------
# Signal-upgrade helpers (Phase 21): orderbook depth + trade freshness +
# exchange status, so the Kalshi quote only influences Signal when the
# market is thick, fresh, and the exchange is actually open.
# ---------------------------------------------------------------------------

_STATUS_CACHE = (0.0, None)       # (ts, status dict or None)
_STATUS_TTL_S = 10 * 60
_ORDERBOOK_CACHE = {}             # ticker -> (ts, orderbook dict)
_ORDERBOOK_TTL_S = 60
_TRADES_CACHE = {}                # ticker -> (ts, [trade dicts])
_TRADES_TTL_S = 60


def exchange_status():
    """Return Kalshi exchange status dict (exchange_active, trading_active,
    intra_exchange_transfers_active) or None on failure. Cached 10 min.
    Short timeout — if Kalshi is slow the whole board build should NOT
    wait on it."""
    global _STATUS_CACHE
    ts, cached = _STATUS_CACHE
    now = time.time()
    if cached is not None and (now - ts) < _STATUS_TTL_S:
        return cached
    try:
        d = _fetch_json(f"{_BASE}/exchange/status", timeout=3)
    except (URLError, ValueError, OSError, TimeoutError, ConnectionError):
        d = None
    _STATUS_CACHE = (now, d)
    return d


def exchange_is_trading():
    """True iff Kalshi exchange is up AND accepting trades right now.
    Falls back to True on fetch failure — don't disable Kalshi signal
    confirmation just because the status probe timed out."""
    s = exchange_status()
    if s is None:
        return True
    return bool(s.get("exchange_active")) and bool(s.get("trading_active"))


def market_orderbook(ticker):
    """Return the top-of-book snapshot for a Kalshi market ticker, or None.
    Shape: {'yes': [[price_cents, size], ...], 'no': [[price_cents, size], ...]}.
    Cached 60s per ticker — orderbook changes faster than series listings
    but we don't need second-by-second refresh for a signal confirmation."""
    if not ticker:
        return None
    hit = _ORDERBOOK_CACHE.get(ticker)
    now = time.time()
    if hit and (now - hit[0]) < _ORDERBOOK_TTL_S:
        return hit[1]
    try:
        d = _fetch_json(
            f"{_BASE}/markets/{ticker}/orderbook?depth=5",
            timeout=3,
        )
        ob = (d or {}).get("orderbook") or None
    except (URLError, ValueError, OSError, TimeoutError, ConnectionError):
        ob = None
    _ORDERBOOK_CACHE[ticker] = (now, ob)
    if len(_ORDERBOOK_CACHE) > 128:
        for k in sorted(_ORDERBOOK_CACHE.items(), key=lambda kv: kv[1][0])[:64]:
            _ORDERBOOK_CACHE.pop(k[0], None)
    return ob


def market_last_trade_ts(ticker):
    """Return the UNIX timestamp of the most-recent trade on this market,
    or None. Trades endpoint returns newest-first; we only need the first.
    Cached 60s."""
    if not ticker:
        return None
    hit = _TRADES_CACHE.get(ticker)
    now = time.time()
    if hit and (now - hit[0]) < _TRADES_TTL_S:
        return hit[1]
    last_ts = None
    try:
        d = _fetch_json(
            f"{_BASE}/markets/trades?ticker={ticker}&limit=1",
            timeout=3,
        )
        trades = (d or {}).get("trades") or []
        if trades:
            created = trades[0].get("created_time")
            if isinstance(created, str):
                # ISO 8601 -> unix ts
                from datetime import datetime as _dt
                try:
                    last_ts = _dt.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except (TypeError, ValueError):
                    last_ts = None
            elif isinstance(created, (int, float)):
                last_ts = float(created)
    except (URLError, ValueError, OSError, TimeoutError, ConnectionError):
        last_ts = None
    _TRADES_CACHE[ticker] = (now, last_ts)
    if len(_TRADES_CACHE) > 128:
        for k in sorted(_TRADES_CACHE.items(), key=lambda kv: kv[1][0])[:64]:
            _TRADES_CACHE.pop(k[0], None)
    return last_ts


# Thresholds for "trustworthy Kalshi quote". All three must pass for a
# Kalshi price to influence the Signal composite — a quote on an empty
# book or a market that hasn't traded in days is noise, not signal.
KALSHI_MIN_BOOK_DEPTH_USD = 500.0   # top-5 cumulative size × price in dollars
KALSHI_MAX_TRADE_AGE_S    = 6 * 3600   # 6 hours
KALSHI_MAX_PROB_GAP       = 0.05       # within 5pp of our fair_prob

# Circuit breaker: if Kalshi starts timing out, stop issuing new calls
# for a while so the main board doesn't accumulate 10+ second lag per
# render. One failure flips the breaker open for 60s.
_CB_OPEN_UNTIL = 0.0
_CB_COOLDOWN_S = 60


def _cb_open():
    return time.time() < _CB_OPEN_UNTIL


def _cb_trip():
    global _CB_OPEN_UNTIL
    _CB_OPEN_UNTIL = time.time() + _CB_COOLDOWN_S


def _market_for_matchup(sport_slug, home_team):
    """Return the Kalshi market dict matching this matchup, or None.
    Shared helper for moneyline_yes_prob + kalshi_confirms_pick so both
    traverse the same series listing exactly once."""
    markets = _series_markets(sport_slug)
    if not markets:
        return None
    home_norm = _norm_team(home_team)
    for m in markets:
        yes_team = _norm_team(m.get("yes_sub_title"))
        if not yes_team:
            continue
        if (yes_team == home_norm
                or yes_team in home_norm
                or home_norm in yes_team):
            return m
    return None


def _orderbook_depth_usd(orderbook, side):
    """Sum top-of-book size × price for one side of the book, in dollars.
    Each Kalshi contract settles at $1.00 so size × (price/100) = dollars
    locked up on that side."""
    if not orderbook:
        return 0.0
    levels = orderbook.get(side) or []
    total = 0.0
    for lvl in levels[:5]:
        try:
            price, size = float(lvl[0]), float(lvl[1])
        except (TypeError, ValueError, IndexError):
            continue
        total += size * (price / 100.0)
    return total


def kalshi_confirms_pick(sport_slug, home_team, is_home_pick, our_fair_prob):
    """True iff Kalshi independently confirms our pick with a trustworthy
    quote. Confirmation requires ALL of:
      * exchange is trading
      * a Kalshi market exists for this matchup
      * live quote (yes_ask / yes_bid / last_price) exists
      * Kalshi implied prob is within 5pp of our fair_prob (same direction)
      * orderbook depth >= $500 on OUR side (meaningful quote)
      * last trade within 6h (fresh quote, not stale)

    Routed to the +1 "cross-book agreement" slot in the Signal composite.
    Returns False (not None) when confirmation fails so the Signal math
    stays simple — the chip in the UI still shows the raw divergence.
    """
    if _cb_open():
        return False
    try:
        if not exchange_is_trading():
            return False
        if our_fair_prob is None or sport_slug not in _SERIES_PREFIX:
            return False
        m = _market_for_matchup(sport_slug, home_team)
        if not m:
            return False
        ticker = m.get("ticker")
        if not ticker:
            return False
        # Kalshi encodes "home wins" as the YES side. If our pick is HOME,
        # Kalshi YES must agree. If our pick is AWAY, Kalshi NO must agree.
        yes_cents = None
        for key in ("yes_ask", "yes_bid", "last_price"):
            v = m.get(key)
            if v is None:
                continue
            try:
                c = float(v)
            except (TypeError, ValueError):
                continue
            if 0 < c < 100:
                yes_cents = c
                break
        if yes_cents is None:
            return False
        kalshi_home_prob = yes_cents / 100.0
        kalshi_our_prob = kalshi_home_prob if is_home_pick else (1.0 - kalshi_home_prob)
        if abs(kalshi_our_prob - our_fair_prob) > KALSHI_MAX_PROB_GAP:
            return False
        # Depth gate — our side of the book
        ob = market_orderbook(ticker)
        depth = _orderbook_depth_usd(ob, "yes" if is_home_pick else "no")
        if depth < KALSHI_MIN_BOOK_DEPTH_USD:
            return False
        # Liveness gate
        last_ts = market_last_trade_ts(ticker)
        if last_ts is None:
            return False
        if (time.time() - last_ts) > KALSHI_MAX_TRADE_AGE_S:
            return False
        return True
    except Exception:
        _cb_trip()
        return False


# ---------------------------------------------------------------------------
# Settlement + live results (Phase 22). Kalshi markets settle to "yes" or
# "no" once a game ends — we use that as a fallback result source for ML
# picks that our ESPN grader couldn't match (e.g. accented team names
# that fall through all the fuzzy-match passes). For LIVE games the
# yes_ask price is the market's current in-play win estimate, which we
# surface alongside the ● LIVE badge on the main board.
# ---------------------------------------------------------------------------

# NOTE: once a Kalshi market is settled, its result never changes. We
# use the series-listing endpoint (status=settled) rather than per-ticker
# polling so one call covers every game of the day.
_SETTLED_CACHE = {}   # sport_slug -> (ts, [markets])
_SETTLED_TTL_S = 10 * 60   # settled markets never change — long cache


def _settled_markets(sport_slug):
    """Return Kalshi SETTLED game markets for this sport, cached. Settled
    markets have status="settled" and result="yes"|"no"."""
    pfx = _SERIES_PREFIX.get(sport_slug)
    if not pfx:
        return []
    hit = _SETTLED_CACHE.get(sport_slug)
    now = time.time()
    if hit and (now - hit[0]) < _SETTLED_TTL_S:
        return hit[1]
    try:
        d = _fetch_json(
            f"{_BASE}/markets?limit=200&status=settled&series_ticker={pfx}",
            timeout=20,
        )
        markets = d.get("markets") or []
    except (URLError, ValueError, OSError, TimeoutError, ConnectionError):
        markets = []
    _SETTLED_CACHE[sport_slug] = (now, markets)
    return markets


def ml_result(sport_slug, home_team, away_team):
    """Return 'home_win' / 'away_win' / None based on Kalshi's settled
    market for this matchup. Kalshi encodes 'home team wins' as YES,
    so result == 'yes' → home win; result == 'no' → away win."""
    if _cb_open() or sport_slug not in _SERIES_PREFIX:
        return None
    markets = _settled_markets(sport_slug)
    if not markets:
        return None
    home_norm = _norm_team(home_team)
    for m in markets:
        yes_team = _norm_team(m.get("yes_sub_title"))
        if not yes_team:
            continue
        if not (yes_team == home_norm
                or yes_team in home_norm
                or home_norm in yes_team):
            continue
        result = (m.get("result") or "").lower()
        if result == "yes":
            return "home_win"
        if result == "no":
            return "away_win"
        return None
    return None


def live_in_play_prob(sport_slug, home_team):
    """Return the market's CURRENT in-play yes_ask (0..1) for this
    matchup's open market, or None if the game isn't live or there's no
    quote. Same lookup path as moneyline_yes_prob — the market stays
    open while the game is in progress and the price updates with each
    scoring event. Caller displays this as 'Kalshi live XX%' on the
    pick card next to the LIVE badge."""
    if _cb_open():
        return None
    return moneyline_yes_prob(sport_slug, home_team, None)


if __name__ == "__main__":
    # CLI smoke
    for sport in ("nfl", "mlb", "nhl", "nba"):
        mk = _series_markets(sport)
        traded = sum(1 for m in mk if m.get("yes_ask") or m.get("yes_bid"))
        settled = _settled_markets(sport)
        print(f"{sport}: {len(mk)} open markets ({traded} live-quote), {len(settled)} settled")
