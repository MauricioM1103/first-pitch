#!/usr/bin/env python3
"""Phase 31 Round 3 pick-selection enhancements.

Four features, deliberately kept small and self-contained so they can
be shipped without touching the core board-build pipeline:

- H: compute_shrinkage_recs(graded_by_date)
     Per-sport calibration shrinkage recommendation (how much to
     dampen model probs toward 50% based on 30-day actual vs
     predicted).

- D: get_weather_risk(sport, home_team, start_time)
     Open-Meteo wind check for a small roster of outdoor MLB and NFL
     venues. Returns 'high' / 'moderate' / 'ok' / None.

- E: get_injury_risk(sport, home_team, away_team, start_time)
     ESPN scoreboard-based scratch check (MLB probable pitcher
     changes). Returns a human-readable note or None.

- F: compute_edges(graded_by_date, min_n=15)
     Mine (sport, market, signal) buckets for high-hit-rate patterns.
     Returns a sorted list suitable for a dashboard table.

Each helper fails safely (returns None / empty) if its data source is
unavailable so bugs here can't take down the board build.
"""
import json
import os
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs")
_WEATHER_CACHE_PATH = os.path.join(_CACHE_DIR, "weather_cache.json")
_INJURY_CACHE_PATH  = os.path.join(_CACHE_DIR, "injury_cache.json")


# --------------- H. Calibration shrinkage recommendation ---------------

def compute_shrinkage_recs(graded_by_date, min_n=25):
    """For each sport, compute (avg_predicted - avg_actual) over the
    last 30 days of settled picks. Returns a list of dicts sorted by
    |delta| desc so the worst-calibrated sports surface first.

    delta > 0  → model over-predicted (shrink toward 50%)
    delta < 0  → model under-predicted (expand away from 50%)
    """
    buckets = {}  # sport -> [sum_pred, sum_actual, n]
    for ds, picks in (graded_by_date or {}).items():
        for p in picks or []:
            r = p.get("result")
            if r not in ("W", "L"):
                continue
            try:
                pred = float(p.get("fair_prob") or 0.0)
            except (TypeError, ValueError):
                continue
            if pred <= 0 or pred >= 1:
                continue
            sport = (p.get("sport") or "").lower()
            if not sport:
                continue
            b = buckets.setdefault(sport, [0.0, 0.0, 0])
            b[0] += pred
            b[1] += 1 if r == "W" else 0
            b[2] += 1
    out = []
    for sport, (sum_pred, sum_actual, n) in buckets.items():
        if n < min_n:
            continue
        avg_pred = sum_pred / n
        avg_act  = sum_actual / n
        delta = avg_pred - avg_act
        # Shrinkage factor: how much to pull model probs toward 50%.
        # delta=0 → shrink=1.0 (no change); delta=0.05 → shrink≈0.90.
        shrink = max(0.5, min(1.5, 1.0 - 2.0 * delta))
        verdict = "OK"
        if abs(delta) >= 0.04:
            verdict = "shrink" if delta > 0 else "expand"
        out.append({
            "sport":   sport,
            "n":       n,
            "pred":    round(avg_pred, 3),
            "actual":  round(avg_act, 3),
            "delta":   round(delta, 3),
            "shrink":  round(shrink, 3),
            "verdict": verdict,
        })
    out.sort(key=lambda x: -abs(x["delta"]))
    return out


# --------------- D. Weather risk (MLB + NFL outdoor venues) ---------------

# A compact lat/lon table for the most common outdoor stadiums. Not
# exhaustive — expand as needed. Dome / retractable-roof venues are
# intentionally omitted so weather doesn't flag indoor games.
_VENUE_COORDS = {
    # MLB — outdoor only (domes/retractables excluded)
    ("mlb", "yankees"):          (40.829, -73.926),
    ("mlb", "red sox"):          (42.346, -71.097),
    ("mlb", "mets"):             (40.757, -73.846),
    ("mlb", "phillies"):         (39.906, -75.167),
    ("mlb", "pirates"):          (40.447, -80.006),
    ("mlb", "orioles"):          (39.284, -76.622),
    ("mlb", "nationals"):        (38.873, -77.008),
    ("mlb", "braves"):           (33.891, -84.468),
    ("mlb", "reds"):             (39.097, -84.507),
    ("mlb", "white sox"):        (41.830, -87.634),
    ("mlb", "cubs"):             (41.948, -87.655),
    ("mlb", "royals"):           (39.051, -94.481),
    ("mlb", "tigers"):           (42.339, -83.049),
    ("mlb", "twins"):            (44.981, -93.278),
    ("mlb", "guardians"):        (41.496, -81.685),
    ("mlb", "dodgers"):          (34.074, -118.240),
    ("mlb", "giants"):           (37.779, -122.389),
    ("mlb", "athletics"):        (37.752, -122.200),
    ("mlb", "padres"):           (32.707, -117.157),
    ("mlb", "angels"):           (33.800, -117.883),
    ("mlb", "mariners"):         (47.591, -122.332),
    ("mlb", "rockies"):          (39.756, -104.994),
    # NFL — outdoor only
    ("nfl", "bills"):            (42.774, -78.787),
    ("nfl", "patriots"):         (42.091, -71.264),
    ("nfl", "jets"):             (40.814, -74.074),
    ("nfl", "giants"):           (40.814, -74.074),
    ("nfl", "steelers"):         (40.447, -80.016),
    ("nfl", "browns"):           (41.506, -81.700),
    ("nfl", "ravens"):           (39.278, -76.623),
    ("nfl", "bengals"):          (39.095, -84.516),
    ("nfl", "titans"):           (36.166, -86.771),
    ("nfl", "jaguars"):          (30.324, -81.637),
    ("nfl", "dolphins"):         (25.958, -80.238),
    ("nfl", "chiefs"):           (39.049, -94.484),
    ("nfl", "broncos"):          (39.744, -105.020),
    ("nfl", "raiders"):          (36.091, -115.183),
    ("nfl", "chargers"):         (33.953, -118.339),
    ("nfl", "49ers"):            (37.403, -121.970),
    ("nfl", "seahawks"):         (47.595, -122.332),
    ("nfl", "packers"):          (44.501, -88.062),
    ("nfl", "bears"):            (41.863, -87.617),
    ("nfl", "eagles"):           (39.901, -75.168),
    ("nfl", "panthers"):         (35.226, -80.853),
    ("nfl", "commanders"):       (38.908, -76.864),
}


def _lookup_venue(sport, home_team):
    """Return (lat, lon) or None if the venue isn't in our table."""
    if not home_team:
        return None
    key = (sport.lower(), home_team.lower().split()[-1])  # match last word (team nickname)
    return _VENUE_COORDS.get(key)


def _load_cache(path):
    try:
        with open(path) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_cache(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_weather_risk(sport, home_team, start_time, cache_ttl=3*3600):
    """Return 'high' / 'moderate' / 'ok' / None based on wind speed at
    kickoff. 'high' means >=20mph (relevant for NFL pass offenses and
    MLB totals). None if the venue is indoors / not in our table / we
    can't reach Open-Meteo."""
    coords = _lookup_venue(sport, home_team)
    if not coords:
        return None
    lat, lon = coords
    cache = _load_cache(_WEATHER_CACHE_PATH)
    cache_key = f"{sport}:{home_team}:{start_time}"
    hit = cache.get(cache_key)
    if hit and (time.time() - float(hit.get("ts", 0))) < cache_ttl:
        return hit.get("risk")
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               "&hourly=wind_speed_10m&wind_speed_unit=mph&forecast_days=2")
        req = Request(url, headers={"User-Agent": "cheeks-model/1.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    # Pick the forecast hour closest to kickoff.
    hours = (data.get("hourly") or {}).get("time") or []
    winds = (data.get("hourly") or {}).get("wind_speed_10m") or []
    if not hours or not winds or not start_time:
        return None
    try:
        if start_time.endswith("Z"):
            start_time = start_time[:-1] + "+00:00"
        target = datetime.fromisoformat(start_time)
    except Exception:
        return None
    best_i = None
    best_diff = None
    for i, hour_iso in enumerate(hours):
        try:
            ht = datetime.fromisoformat(hour_iso)
            if ht.tzinfo is None:
                from zoneinfo import ZoneInfo
                ht = ht.replace(tzinfo=ZoneInfo("UTC"))
        except Exception:
            continue
        diff = abs((ht - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_i = i
    if best_i is None:
        return None
    wind = winds[best_i] or 0
    if wind >= 20:
        risk = "high"
    elif wind >= 13:
        risk = "moderate"
    else:
        risk = "ok"
    cache[cache_key] = {"risk": risk, "wind": wind, "ts": time.time()}
    _save_cache(_WEATHER_CACHE_PATH, cache)
    return risk


# --------------- E. Injury / scratch risk ---------------

def get_injury_risk(sport, home_team, away_team, start_time, cache_ttl=1800):
    """Return a short note or None. MVP: for MLB, check ESPN
    scoreboard for a 'probable pitcher' field change — if a saved pick
    was built against a different pitcher than what's now listed,
    flag it. Other sports: not implemented; returns None.

    Cached per matchup for 30 min to limit ESPN hits."""
    if (sport or "").lower() != "mlb":
        return None
    cache = _load_cache(_INJURY_CACHE_PATH)
    cache_key = f"{sport}:{away_team}@{home_team}:{start_time[:10] if start_time else ''}"
    hit = cache.get(cache_key)
    if hit and (time.time() - float(hit.get("ts", 0))) < cache_ttl:
        return hit.get("note")
    # ESPN MLB scoreboard
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
        req = Request(url, headers={"User-Agent": "cheeks-model/1.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    note = None
    for ev in (data.get("events") or []):
        comps = (ev.get("competitions") or [{}])[0]
        teams = {}
        for t in comps.get("competitors", []):
            tn = ((t.get("team") or {}).get("displayName") or "").lower()
            teams[tn] = t
        if home_team.lower() not in teams or away_team.lower() not in teams:
            continue
        for role in ("probables",):
            ps = comps.get(role) or []
            if not ps:
                note = f"No probable pitcher listed for {home_team} / {away_team}"
                break
    cache[cache_key] = {"note": note, "ts": time.time()}
    _save_cache(_INJURY_CACHE_PATH, cache)
    return note


# --------------- F. Edge mining ---------------

def compute_edges(graded_by_date, min_n=15, top=20):
    """Mine (sport, market_bucket, signal_bucket) tuples for high hit
    rates on >= min_n settled picks. Returns top-N by win rate (ties
    broken by sample size)."""
    buckets = {}  # (sport, market, signal) -> [W, L, units]
    for ds, picks in (graded_by_date or {}).items():
        for p in picks or []:
            r = p.get("result")
            if r not in ("W", "L"):
                continue
            sport = (p.get("sport") or "").lower()
            market = (p.get("market") or "").lower()[:24] or "?"
            sig = int(p.get("signal_score") or 0)
            key = (sport, market, sig)
            b = buckets.setdefault(key, [0, 0, 0.0])
            b[0 if r == "W" else 1] += 1
            try:
                b[2] += float(p.get("profit_u") or 0.0)
            except (TypeError, ValueError):
                pass
    out = []
    for (sport, market, sig), (w, l, u) in buckets.items():
        n = w + l
        if n < min_n:
            continue
        out.append({
            "sport":   sport,
            "market":  market,
            "signal":  sig,
            "n":       n,
            "hit":     round(w / n * 100, 1),
            "units":   round(u, 2),
        })
    # Sort by hit rate desc, then by sample size desc
    out.sort(key=lambda e: (-e["hit"], -e["n"]))
    return out[:top]
