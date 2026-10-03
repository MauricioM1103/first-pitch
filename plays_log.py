#!/usr/bin/env python3
"""Daily picks snapshot + results grading for the /logged dashboard.

Each day's picks (as produced by picks.collect_picks) are saved to
`logs/picks_YYYY-MM-DD.json` the first time anyone views the picks page
that day. The /logged page then grades those picks against final scores
pulled from The Odds API (`/sports/<key>/scores`) and surfaces W/L
records for last 7 / 30 / 90 days, split into "strong" vs the rest.

Graded picks are persisted to `logs/graded_YYYY-MM-DD.json` so we only
pay the API cost once per day per sport, and so games older than the
Odds API's 3-day `daysFrom` window keep their final grade.

Pure stdlib + urllib — fits the Flask/Render deployment.
"""
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def picks_file(date_str):
    return os.path.join(LOG_DIR, f"picks_{date_str}.json")


def graded_file(date_str):
    return os.path.join(LOG_DIR, f"graded_{date_str}.json")


def save_daily_picks(date_str, picks):
    """Snapshot the day's picks. Idempotent per date — won't overwrite an
    existing file (we want the snapshot locked in from the first view, so
    later edits to the model don't retroactively rewrite history).
    """
    ensure_log_dir()
    path = picks_file(date_str)
    if os.path.exists(path):
        return False
    slim = []
    for p in picks:
        slim.append({
            "id":          p.get("id"),
            "sport":       p.get("sport"),
            "sport_slug":  p.get("sport_slug"),
            "home_team":   p.get("home_team"),
            "away_team":   p.get("away_team"),
            "market":      p.get("market"),
            "pick":        p.get("pick"),
            "decimal":     p.get("decimal"),
            "american":    p.get("american"),
            "book":        p.get("book"),
            "fair_prob":   p.get("fair_prob"),
            "ev_pct":      p.get("ev_pct"),
            "strong":      bool(p.get("strong")),
            "start_time":  p.get("start_time"),
            "snapshotted_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })
    try:
        with open(path, "w") as f:
            json.dump({"date": date_str, "picks": slim}, f)
    except OSError:
        return False
    return True


def all_logged_dates(limit=120):
    """Sorted list of YYYY-MM-DD dates we have snapshots for (newest first)."""
    ensure_log_dir()
    files = [f for f in os.listdir(LOG_DIR)
             if f.startswith("picks_") and f.endswith(".json")]
    dates = sorted((f[6:-5] for f in files), reverse=True)
    return dates[:limit]


def read_picks(date_str):
    path = picks_file(date_str)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_graded(date_str):
    path = graded_file(date_str)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_graded(date_str, graded):
    try:
        with open(graded_file(date_str), "w") as f:
            json.dump(graded, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Odds API results fetch (one call per sport key, cached per process)
# ---------------------------------------------------------------------------

_RESULTS_CACHE = {}      # sport_key -> (events, ts)
_RESULTS_TTL_S = 30 * 60


def _fetch_json(url, timeout=15):
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_scores(sport_key, days_from=3):
    """Pull final scores from The Odds API. Returns list of event dicts."""
    if not sport_key:
        return []
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    now = time.time()
    hit = _RESULTS_CACHE.get(sport_key)
    if hit and now - hit[1] < _RESULTS_TTL_S:
        return hit[0]
    params = urlencode({"apiKey": key, "daysFrom": days_from})
    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores/?{params}"
    try:
        events = _fetch_json(url) or []
    except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
        events = []
    _RESULTS_CACHE[sport_key] = (events, now)
    return events


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _norm(s):
    return (s or "").lower().strip().replace(".", "").replace("-", " ")


def find_result_event(pick, events):
    """Match a pick to a scored event (same home/away teams)."""
    home = _norm(pick.get("home_team"))
    away = _norm(pick.get("away_team"))
    for ev in events:
        if not ev.get("completed"):
            continue
        eh = _norm(ev.get("home_team"))
        ea = _norm(ev.get("away_team"))
        if (eh == home and ea == away) or (home and home in eh) or (home and eh in home):
            # Try fuzzy match if exact fails
            if eh == home and ea == away:
                return ev
    # Fallback: fuzzy — any event where both team-name substrings line up
    for ev in events:
        if not ev.get("completed"):
            continue
        eh = _norm(ev.get("home_team"))
        ea = _norm(ev.get("away_team"))
        if home and away and ((home in eh or eh in home) and (away in ea or ea in away)):
            return ev
    return None


def _extract_scores(event):
    """Return (home_score, away_score) or (None, None) if not complete."""
    scores = event.get("scores") or []
    if not scores:
        return None, None
    by_name = {}
    for s in scores:
        try:
            by_name[_norm(s.get("name"))] = float(s.get("score", 0))
        except (TypeError, ValueError):
            pass
    home = by_name.get(_norm(event.get("home_team")))
    away = by_name.get(_norm(event.get("away_team")))
    return home, away


def grade_pick(pick, event):
    """Return 'W' | 'L' | 'P' | 'pending' for the pick given the result event."""
    if not event:
        return "pending"
    home, away = _extract_scores(event)
    if home is None or away is None:
        return "pending"

    market = pick.get("market", "") or ""
    text = pick.get("pick", "") or ""
    home_name = _norm(pick.get("home_team"))
    away_name = _norm(pick.get("away_team"))
    pick_norm = _norm(text)

    # Moneyline (incl. 1H / 1P ML and soccer draws)
    if "ML" in market:
        if "draw" in pick_norm:
            return "W" if home == away else "L"
        if home_name and (pick_norm == home_name or home_name in pick_norm):
            return "W" if home > away else ("P" if home == away else "L")
        if away_name and (pick_norm == away_name or away_name in pick_norm):
            return "W" if away > home else ("P" if home == away else "L")

    # Totals
    if "Total" in market:
        m = re.search(r"(over|under)\s*([\d.]+)", pick_norm)
        if m:
            side = m.group(1)
            try:
                line = float(m.group(2))
            except ValueError:
                return "pending"
            tot = home + away
            if abs(tot - line) < 1e-6:
                return "P"
            if side == "over":
                return "W" if tot > line else "L"
            return "W" if tot < line else "L"

    # BTTS
    if market == "BTTS":
        if "yes" in pick_norm:
            return "W" if (home > 0 and away > 0) else "L"
        if "no" in pick_norm:
            return "W" if (home == 0 or away == 0) else "L"

    # Spread / Run Line / Puck Line — pick text looks like
    #   "Dallas Cowboys -3.5" or "Yankees +1.5"
    if market in ("Spread", "Run Line", "Puck Line", "1H Spread", "1P Spread"):
        m = re.search(r"([+-][\d.]+)", text)
        if not m:
            return "pending"
        try:
            line = float(m.group(1))
        except ValueError:
            return "pending"
        if home_name and home_name in pick_norm:
            margin = (home - away) + line
        elif away_name and away_name in pick_norm:
            margin = (away - home) + line
        else:
            return "pending"
        if abs(margin) < 1e-6:
            return "P"
        return "W" if margin > 0 else "L"

    return "pending"


def profit_at_1u(result, decimal):
    """Units profit at a 1-unit stake."""
    try:
        d = float(decimal or 0)
    except (TypeError, ValueError):
        d = 0
    if result == "W":
        return d - 1.0 if d > 1 else 0.0
    if result == "L":
        return -1.0
    return 0.0  # push or pending


# ---------------------------------------------------------------------------
# Grading pipeline
# ---------------------------------------------------------------------------

def grade_date(date_str, sport_key_by_slug, force=False):
    """Grade one date — write graded_{date}.json. Skip if already written
    AND all picks are settled (unless force=True)."""
    existing = read_graded(date_str)
    if existing and not force:
        all_settled = all(p.get("result") in ("W", "L", "P") for p in existing.get("picks", []))
        if all_settled:
            return existing

    raw = read_picks(date_str)
    if not raw:
        return existing

    # Only call sports that this date actually has picks in
    need_sports = {p["sport_slug"] for p in raw["picks"]}
    events_by_sport = {}
    for slug in need_sports:
        key = sport_key_by_slug.get(slug)
        if key:
            events_by_sport[slug] = fetch_scores(key)

    graded_picks = []
    for p in raw["picks"]:
        events = events_by_sport.get(p["sport_slug"], [])
        ev = find_result_event(p, events)
        prior = None
        if existing:
            prior = next((g for g in existing.get("picks", []) if g.get("id") == p.get("id")), None)
        if prior and prior.get("result") in ("W", "L", "P"):
            # Keep prior settled result (Odds API window drops after 3 days)
            graded_picks.append(prior)
            continue
        res = grade_pick(p, ev)
        home_s, away_s = _extract_scores(ev) if ev else (None, None)
        graded_picks.append({
            **p,
            "result":     res,
            "profit_u":   profit_at_1u(res, p.get("decimal")),
            "home_score": home_s,
            "away_score": away_s,
            "graded_at":  datetime.utcnow().isoformat(timespec="seconds") + "Z" if res in ("W","L","P") else None,
        })

    graded = {"date": date_str, "picks": graded_picks}
    write_graded(date_str, graded)
    return graded


def grade_all(limit_dates=90):
    """Grade every logged date up to `limit_dates` back. Returns dict by date."""
    import sports as _sports
    sport_key_by_slug = {s["slug"]: s.get("odds_api_key") for s in _sports.SPORTS}
    out = {}
    for ds in all_logged_dates(limit_dates):
        g = grade_date(ds, sport_key_by_slug)
        if g:
            out[ds] = g["picks"]
    return out


# ---------------------------------------------------------------------------
# Record aggregation
# ---------------------------------------------------------------------------

def record_in_window(graded_by_date, days_back, strong_only=False, today=None):
    today = today or date.today()
    cutoff = today - timedelta(days=days_back - 1)
    w = l = p = pending = 0
    units = 0.0
    for ds, picks in graded_by_date.items():
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff or d > today:
            continue
        for pk in picks:
            if strong_only and not pk.get("strong"):
                continue
            r = pk.get("result")
            if r == "W":
                w += 1
            elif r == "L":
                l += 1
            elif r == "P":
                p += 1
            else:
                pending += 1
            units += pk.get("profit_u", 0) or 0
    settled = w + l
    pct = (w / settled * 100) if settled else 0.0
    roi = (units / settled * 100) if settled else 0.0
    return {
        "wins": w, "losses": l, "pushes": p, "pending": pending,
        "settled": settled, "win_pct": pct, "units": units, "roi_pct": roi,
    }


def summary(graded_by_date, today=None):
    """Return dict of 7D / 30D / 90D records for all + strong-only."""
    out = {}
    for label, days in (("7d", 7), ("30d", 30), ("90d", 90)):
        out[f"all_{label}"]    = record_in_window(graded_by_date, days, False, today)
        out[f"strong_{label}"] = record_in_window(graded_by_date, days, True,  today)
    return out


def breakdown_by_key(graded_by_date, key_fn, days_back=90, today=None):
    """Group settled picks by a key (sport, market, etc) and compute per-group record.

    Returns list of {key, wins, losses, pushes, settled, win_pct, units, roi_pct}
    sorted by settled desc.
    """
    today = today or date.today()
    cutoff = today - timedelta(days=days_back - 1)
    buckets = {}
    for ds, picks in graded_by_date.items():
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff or d > today:
            continue
        for pk in picks:
            k = key_fn(pk) or "(unknown)"
            b = buckets.setdefault(k, {"wins": 0, "losses": 0, "pushes": 0, "units": 0.0})
            r = pk.get("result")
            if r == "W":   b["wins"] += 1
            elif r == "L": b["losses"] += 1
            elif r == "P": b["pushes"] += 1
            b["units"] += pk.get("profit_u", 0) or 0
    rows = []
    for k, b in buckets.items():
        settled = b["wins"] + b["losses"]
        pct = (b["wins"] / settled * 100) if settled else 0.0
        roi = (b["units"] / settled * 100) if settled else 0.0
        rows.append({
            "key": k, "wins": b["wins"], "losses": b["losses"],
            "pushes": b["pushes"], "settled": settled,
            "win_pct": pct, "units": b["units"], "roi_pct": roi,
        })
    rows.sort(key=lambda r: (-r["settled"], -r["units"]))
    return rows
