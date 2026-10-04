#!/usr/bin/env python3
"""Load the user's Polymarket activity export and surface it as a picks signal.

Picks are filtered + ranked by the model-market consensus layer, but the
user also has a REAL betting history that shows which categories they
consistently win and lose on. This module uses that history as a
feedback signal so daily picks that match a historically-winning
pattern (e.g. "soccer underdog ML", "NCAAF total under") get a visible
badge, and picks that match a chronically-losing pattern get flagged.

Input CSV: ~/OneDrive/Desktop/DocumentsBets/polymarket_activities.csv
  (user's Polymarket export; one row per activity, hundreds of columns)

What we extract per resolved trade:
  * sport_group:  football | basketball | baseball | hockey | soccer | other
  * market_class: moneyline | spread | total | period | prop | first_half
  * side:         yes_long | no_short
  * pnl:          realized profit/loss in USD
  * title / question / event_slug for display

What we expose:
  * load_history(path=None) → list of normalized trade dicts
  * category_stats()        → {(sport, market_class): {w, l, n, pnl, roi}}
  * score_pick(pick)        → {match_cat, w, l, roi, badge} or None
  * summary()               → overall record + top winning / losing patterns
"""
import csv
import json
import os
from collections import defaultdict


# Where we look for the Polymarket CSV, in priority order. The project-local
# path lets the file ship to Render via git; the user's local export is the
# fallback for development on the laptop.
_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "polymarket", "activities.csv"),
    os.path.expanduser("~/OneDrive/Desktop/DocumentsBets/polymarket_activities.csv"),
]


def _resolve_path():
    for p in _SEARCH_PATHS:
        if os.path.exists(p):
            return p
    return _SEARCH_PATHS[0]


DEFAULT_PATH = _resolve_path()

# In-process cache so repeated picks renders don't re-parse the 240-row CSV
_HISTORY_CACHE = None


def _classify_sport(sports_market_type, event_slug):
    s = (sports_market_type or "").lower()
    e = (event_slug or "").lower()
    if "football" in s or e.startswith("cfb-") or e.startswith("nfl-") or "cfb" in e:
        if "cfb" in e or "ncaa" in s:
            return "ncaaf"
        return "nfl"
    if "basketball" in s or e.startswith("nba-"):
        return "nba"
    if "baseball" in s or "mlb" in e:
        return "mlb"
    if "hockey" in s or "nhl" in e:
        return "nhl"
    if "soccer" in s or any(k in e for k in ("soc-", "epl-", "laliga-", "intl-",
                                             "mex-", "ucl-", "europa-")):
        return "soccer"
    return "other"


def _classify_market(sports_market_type, sports_market_type_v2):
    s = (sports_market_type or "").lower()
    v2 = (sports_market_type_v2 or "").lower()
    if "total" in s or "total" in v2:
        return "total"
    if "spread" in s or "spread" in v2 or "handicap" in s:
        return "spread"
    if "first_period" in s or "first_period" in v2 or "1p" in s:
        return "period"
    if "first_half" in s or "halftime" in s or "half_time" in s or "1h" in s:
        return "first_half"
    if "prop" in v2 or "prop" in s:
        return "prop"
    if "moneyline" in s or "winner" in s or "match_winner" in v2 or "ml" in s:
        return "moneyline"
    return "other"


def _classify_side(resolution_side):
    """POSITION_RESOLUTION_SIDE_LONG = bought Yes / Over / favorite;
    POSITION_RESOLUTION_SIDE_SHORT = bought No / Under / underdog."""
    s = (resolution_side or "").upper()
    if "LONG" in s:
        return "yes_long"
    if "SHORT" in s:
        return "no_short"
    return "unknown"


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _parse_row(r):
    """Pull the fields we need from one position-resolution CSV row.
    Returns None if the row isn't a resolved trade with realized PnL."""
    title = (r.get("positionResolution.market.title")
             or r.get("positionResolution.afterPosition.marketMetadata.title")
             or "")
    if not title:
        return None
    pnl = _to_float(r.get("positionResolution.afterPosition.realized.value"))
    sports_market_type = r.get("positionResolution.market.sportsMarketType") or ""
    sports_market_type_v2 = r.get("positionResolution.market.sportsMarketTypeV2") or ""
    event_slug = r.get("positionResolution.afterPosition.marketMetadata.eventSlug") or ""
    side = r.get("positionResolution.side") or ""
    question = r.get("positionResolution.market.question") or ""
    line = _to_float(r.get("positionResolution.market.line"))
    return {
        "title":          title,
        "question":       question,
        "line":           line,
        "sport":          _classify_sport(sports_market_type, event_slug),
        "market_class":   _classify_market(sports_market_type, sports_market_type_v2),
        "side":           _classify_side(side),
        "pnl":            pnl,
        "event_slug":     event_slug,
        "won":            pnl > 0,
        "lost":           pnl < 0,
        "settled_at":     r.get("positionResolution.updateTime", ""),
    }


def load_history(path=None):
    """Parse the Polymarket CSV into normalized trade dicts. Cached per process."""
    global _HISTORY_CACHE
    if _HISTORY_CACHE is not None:
        return _HISTORY_CACHE
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        _HISTORY_CACHE = []
        return _HISTORY_CACHE
    out = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tr = _parse_row(row)
                if tr and (tr["won"] or tr["lost"]):
                    out.append(tr)
    except Exception:
        out = []
    _HISTORY_CACHE = out
    return out


def category_stats(trades=None):
    """Bucket trades by (sport, market_class, side) and compute win rate + ROI."""
    trades = trades if trades is not None else load_history()
    buckets = defaultdict(lambda: {"w": 0, "l": 0, "n": 0, "pnl": 0.0})
    for t in trades:
        for key in (
            (t["sport"], t["market_class"], t["side"]),
            (t["sport"], t["market_class"], "any"),
            (t["sport"], "any", "any"),
        ):
            b = buckets[key]
            b["n"] += 1
            if t["won"]:  b["w"] += 1
            if t["lost"]: b["l"] += 1
            b["pnl"] += t["pnl"]
    out = {}
    for k, b in buckets.items():
        settled = b["w"] + b["l"]
        pct = (b["w"] / settled * 100) if settled else 0.0
        roi = (b["pnl"] / settled * 100) if settled else 0.0
        out[k] = {"wins": b["w"], "losses": b["l"], "settled": settled,
                  "pnl": b["pnl"], "win_pct": pct, "roi_pct": roi}
    return out


def _classify_pick(pick):
    """Map a daily pick to (sport, market_class, side) for signal lookup."""
    sport_map = {
        "mlb": "mlb", "nfl": "nfl", "ncaaf": "ncaaf", "nhl": "nhl",
        "ufc": "other",
    }
    slug = pick.get("sport_slug", "")
    sport = sport_map.get(slug) or (
        "soccer" if slug in ("epl", "laliga", "ligamx", "ucl", "europa",
                              "international") else "other"
    )
    market = (pick.get("market") or "").lower()
    if "total" in market:  mc = "total"
    elif "spread" in market or "run line" in market or "puck line" in market: mc = "spread"
    elif "1p" in market:   mc = "period"
    elif "1h" in market:   mc = "first_half"
    elif "btts" in market or "win-to-nil" in market: mc = "prop"
    else:                   mc = "moneyline"

    text = (pick.get("pick") or "").lower()
    if "over" in text:   side = "yes_long"
    elif "under" in text: side = "no_short"
    elif "yes" in text:  side = "yes_long"
    elif "no" in text:   side = "no_short"
    else:                 side = "any"
    return sport, mc, side


def score_pick(pick, min_samples=3):
    """Return the historical Polymarket signal for a pick, or None.

    Tries most-specific key first (sport × market × side), falls back
    through sport × market, then sport × any. Returns the first bucket
    with at least `min_samples` settled trades.
    """
    stats = category_stats()
    sport, market, side = _classify_pick(pick)
    for key in (
        (sport, market, side),
        (sport, market, "any"),
        (sport, "any", "any"),
    ):
        bucket = stats.get(key)
        if bucket and bucket["settled"] >= min_samples:
            tone = ("good" if bucket["roi_pct"] >= 5 and bucket["win_pct"] >= 55
                    else "bad" if bucket["roi_pct"] <= -10 or bucket["win_pct"] <= 35
                    else "info")
            return {
                "match_key":  "/".join(str(k) for k in key),
                "wins":       bucket["wins"],
                "losses":     bucket["losses"],
                "win_pct":    bucket["win_pct"],
                "roi_pct":    bucket["roi_pct"],
                "pnl":        bucket["pnl"],
                "settled":    bucket["settled"],
                "tone":       tone,
                "label":      _signal_label(key, bucket, tone),
            }
    return None


def _signal_label(key, bucket, tone):
    sport, market, side = key
    if tone == "good":
        prefix = "Pattern match"
    elif tone == "bad":
        prefix = "Pattern warning"
    else:
        prefix = "Pattern"
    sport_disp = sport.upper() if sport != "other" else "sport"
    side_disp = "" if side == "any" else f" / {side.replace('_', ' ')}"
    # roi_pct is PnL per trade in cents-of-stake units — label it accurately
    return (f"{prefix}: your Polymarket history on {sport_disp} {market}{side_disp} "
            f"is {bucket['wins']}-{bucket['losses']} "
            f"({bucket['win_pct']:.0f}% hit, ${bucket['pnl']:+.0f} total)")


def summary():
    """Overall record + top-5 winning / losing (sport, market, side) buckets."""
    trades = load_history()
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                "top_winning": [], "top_losing": []}
    stats = category_stats(trades)
    # Only show specific (sport, market, side) buckets with ≥ 3 samples
    specific = {k: v for k, v in stats.items()
                if k[1] != "any" and k[2] != "any" and v["settled"] >= 3}
    top_winning = sorted(specific.items(), key=lambda kv: -kv[1]["pnl"])[:5]
    top_losing  = sorted(specific.items(), key=lambda kv:  kv[1]["pnl"])[:5]
    wins = sum(1 for t in trades if t["won"])
    losses = sum(1 for t in trades if t["lost"])
    pnl = sum(t["pnl"] for t in trades)
    return {
        "total_trades": len(trades),
        "wins":   wins,
        "losses": losses,
        "win_pct": (wins / (wins + losses) * 100) if (wins + losses) else 0,
        "pnl":    pnl,
        "roi_pct": (pnl / (wins + losses) * 100) if (wins + losses) else 0,
        "top_winning": [{"key": "/".join(str(x) for x in k), **v} for k, v in top_winning],
        "top_losing":  [{"key": "/".join(str(x) for x in k), **v} for k, v in top_losing],
    }


if __name__ == "__main__":
    s = summary()
    print(f"Total trades: {s['total_trades']}  record {s['wins']}-{s['losses']}  "
          f"PnL ${s['pnl']:+.2f} ({s['roi_pct']:+.1f}% ROI)")
    print("Top winning patterns:")
    for row in s["top_winning"]:
        print(f"  +${row['pnl']:6.2f}  {row['wins']}-{row['losses']} "
              f"({row['win_pct']:.0f}%)  {row['key']}")
    print("Top losing patterns:")
    for row in s["top_losing"]:
        print(f"  ${row['pnl']:+7.2f}  {row['wins']}-{row['losses']} "
              f"({row['win_pct']:.0f}%)  {row['key']}")
