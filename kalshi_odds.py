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

# Kalshi series ticker prefixes by our sport slug.
_SERIES_PREFIX = {
    "nfl":   "KXNFLGAME",
    "mlb":   "KXMLBGAME",
    "nba":   "KXNBAGAME",
    "nhl":   "KXNHLGAME",
    "ncaaf": "KXCFBGAME",
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


if __name__ == "__main__":
    # CLI smoke
    for sport in ("nfl", "mlb", "nhl", "nba"):
        mk = _series_markets(sport)
        traded = sum(1 for m in mk if m.get("yes_ask") or m.get("yes_bid"))
        print(f"{sport}: {len(mk)} markets listed, {traded} with a live quote")
