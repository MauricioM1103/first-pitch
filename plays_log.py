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

import log_persist

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
REMOTE_LOGS_PATH = log_persist.DEFAULT_LOGS_PATH
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


def save_daily_picks(date_str, picks, overwrite_today=True):
    """Snapshot the day's picks.

    For TODAY's date we allow overwrite — the board updates throughout the
    day (consensus probs refresh, new games get added) and we want the most
    recent view on disk. For PAST dates the first snapshot locks in so
    history never changes.

    Also mirrors to the GitHub repo via log_persist when GITHUB_TOKEN is
    set — critical on Render free tier, which wipes `logs/` on every
    spin-down (so without the mirror, nothing survives from day to day).
    """
    ensure_log_dir()
    path = picks_file(date_str)
    try:
        from zoneinfo import ZoneInfo
        is_today = (date_str == datetime.now(ZoneInfo("America/Chicago")).date().isoformat())
    except Exception:
        is_today = True  # safer to allow overwrite than to silently skip

    if os.path.exists(path) and not (overwrite_today and is_today):
        return False

    # Preserve first-save snapshots per pick ID so CLV (closing-line value)
    # can be computed later. On OVERWRITE, we lock in the ORIGINAL
    # pinnacle_prob / decimal from the first time we saved this pick; later
    # saves within the day only update the "latest" fields.
    existing_first = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                prev = json.load(f)
            for prev_p in prev.get("picks", []):
                pid = prev_p.get("id")
                if pid:
                    existing_first[pid] = {
                        "first_snapshot_at":  prev_p.get("first_snapshot_at") or prev_p.get("snapshotted_at"),
                        "first_pinnacle_prob": prev_p.get("first_pinnacle_prob")
                                                 if prev_p.get("first_pinnacle_prob") is not None
                                                 else prev_p.get("pinnacle_prob"),
                        "first_decimal":      prev_p.get("first_decimal") or prev_p.get("decimal"),
                        "first_fair_prob":    prev_p.get("first_fair_prob")
                                                 if prev_p.get("first_fair_prob") is not None
                                                 else prev_p.get("fair_prob"),
                    }
        except (OSError, json.JSONDecodeError):
            pass

    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    slim = []
    for p in picks:
        pid = p.get("id")
        first = existing_first.get(pid, {})
        slim.append({
            "id":            pid,
            "sport":         p.get("sport"),
            "sport_slug":    p.get("sport_slug"),
            "home_team":     p.get("home_team"),
            "away_team":     p.get("away_team"),
            "market":        p.get("market"),
            "pick":          p.get("pick"),
            "decimal":       p.get("decimal"),
            "american":      p.get("american"),
            "book":          p.get("book"),
            "fair_prob":     p.get("fair_prob"),
            "pricing_prob":  p.get("pricing_prob"),
            "mc_prob":       p.get("mc_prob"),
            "consensus_prob": p.get("consensus_prob"),
            "pinnacle_prob": p.get("pinnacle_prob"),
            "ev_pct":        p.get("ev_pct"),
            "strong":        bool(p.get("strong")),
            "model_only":    bool(p.get("model_only")),
            "source":        p.get("source") or ("book_dk" if p.get("book") == "draftkings"
                                                   else "model" if p.get("model_only")
                                                   else "ev"),
            "start_time":    p.get("start_time"),
            "snapshotted_at": now_iso,
            # CLV capture — first-save snapshot locked in across the day
            "first_snapshot_at":  first.get("first_snapshot_at") or now_iso,
            "first_pinnacle_prob": first.get("first_pinnacle_prob")
                                     if first.get("first_pinnacle_prob") is not None
                                     else p.get("pinnacle_prob"),
            "first_decimal":      first.get("first_decimal") or p.get("decimal"),
            "first_fair_prob":    first.get("first_fair_prob")
                                     if first.get("first_fair_prob") is not None
                                     else p.get("fair_prob"),
        })
    payload = {"date": date_str, "picks": slim}
    text = json.dumps(payload)
    try:
        with open(path, "w") as f:
            f.write(text)
    except OSError:
        return False
    # Mirror to GitHub so the snapshot survives the next Render restart.
    try:
        log_persist.write_file(
            f"{REMOTE_LOGS_PATH}/picks_{date_str}.json", text,
            message=f"log: picks for {date_str} ({len(slim)} picks)",
        )
    except Exception:
        pass
    return True


def all_logged_dates(limit=120):
    """Sorted list of YYYY-MM-DD dates we have snapshots for (newest first).

    Merges local disk and the GitHub-backed store so we see every date we've
    ever logged, even after a Render restart wiped the local copy.
    """
    ensure_log_dir()
    local = {f[6:-5] for f in os.listdir(LOG_DIR)
             if f.startswith("picks_") and f.endswith(".json")}
    remote = set()
    try:
        for entry in log_persist.list_dir(REMOTE_LOGS_PATH):
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            if name.startswith("picks_") and name.endswith(".json"):
                remote.add(name[6:-5])
    except Exception:
        pass
    dates = sorted(local | remote, reverse=True)
    return dates[:limit]


def _hydrate_from_remote(remote_path, local_path):
    """Download a file from GitHub into local cache (so subsequent reads
    hit the fast path). Returns the text or None."""
    try:
        text, _ = log_persist.read_file(remote_path)
    except Exception:
        text = None
    if not text:
        return None
    try:
        ensure_log_dir()
        with open(local_path, "w") as f:
            f.write(text)
    except OSError:
        pass
    return text


def read_picks(date_str):
    path = picks_file(date_str)
    text = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            text = None
    if text is None:
        text = _hydrate_from_remote(f"{REMOTE_LOGS_PATH}/picks_{date_str}.json", path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_graded(date_str):
    path = graded_file(date_str)
    text = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            text = None
    if text is None:
        text = _hydrate_from_remote(f"{REMOTE_LOGS_PATH}/graded_{date_str}.json", path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def write_graded(date_str, graded):
    text = json.dumps(graded)
    try:
        with open(graded_file(date_str), "w") as f:
            f.write(text)
    except OSError:
        pass
    # Mirror the graded file to GitHub so results persist across restarts.
    try:
        log_persist.write_file(
            f"{REMOTE_LOGS_PATH}/graded_{date_str}.json", text,
            message=f"log: grade {date_str}",
        )
    except Exception:
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
    """Pull final scores from The Odds API. Returns list of event dicts.
    Returns [] when ODDS_API_KEY isn't set — the grader then uses sport-
    specific fallbacks (fetch_scores_mlb for MLB)."""
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


def fetch_scores_mlb(date_str):
    """Fallback MLB grader via statsapi.mlb.com (public, no key required).

    Returns the same shape as the Odds API scores endpoint so grade_pick
    can grade MLB without ODDS_API_KEY. We only need completed games with
    final scores.
    """
    now = time.time()
    cache_key = ("mlb_statsapi", date_str)
    hit = _RESULTS_CACHE.get(cache_key)
    if hit and now - hit[1] < _RESULTS_TTL_S:
        return hit[0]
    # statsapi schedule endpoint — includes linescore for finished games
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={date_str}"
           f"&hydrate=linescore,team")
    try:
        data = _fetch_json(url) or {}
    except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
        _RESULTS_CACHE[cache_key] = ([], now)
        return []
    events = []
    for day in (data.get("dates") or []):
        for g in day.get("games") or []:
            status = (g.get("status") or {}).get("abstractGameState") or ""
            if status != "Final":
                continue
            teams = g.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            home_score = (teams.get("home") or {}).get("score")
            away_score = (teams.get("away") or {}).get("score")
            if home_score is None or away_score is None:
                continue
            # Inning-by-inning scoring for F5 / NRFI grading. Each entry is
            # {"num": int, "home": runs, "away": runs}. statsapi exposes
            # these under linescore.innings when the game has started.
            innings = []
            linescore = g.get("linescore") or {}
            for inn in (linescore.get("innings") or []):
                try:
                    h = inn.get("home", {}).get("runs")
                    a = inn.get("away", {}).get("runs")
                    innings.append({
                        "num":  int(inn.get("num") or 0),
                        "home": int(h) if h is not None else None,
                        "away": int(a) if a is not None else None,
                    })
                except (TypeError, ValueError):
                    continue
            events.append({
                "id":            str(g.get("gamePk")),
                "home_team":     home.get("name") or home.get("teamName") or "",
                "away_team":     away.get("name") or away.get("teamName") or "",
                "home_team_short": home.get("teamName") or home.get("shortName") or "",
                "away_team_short": away.get("teamName") or away.get("shortName") or "",
                "completed":     True,
                "scores": [
                    {"name": home.get("name") or home.get("teamName"), "score": str(home_score)},
                    {"name": away.get("name") or away.get("teamName"), "score": str(away_score)},
                ],
                "innings":       innings,
                "commence_time": g.get("gameDate"),
            })
    _RESULTS_CACHE[cache_key] = (events, now)
    return events


def grader_status():
    """Diagnostic for /logged: which graders are available right now?"""
    return {
        "odds_api_enabled": bool(os.environ.get("ODDS_API_KEY")),
        "mlb_statsapi":     True,  # always available, no key needed
    }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _norm(s):
    return (s or "").lower().strip().replace(".", "").replace("-", " ")


def find_result_event(pick, events):
    """Match a pick to a scored event — handles short vs full team names
    (e.g. pick carries "White Sox", event has "Chicago White Sox"). Also
    checks statsapi's short-name field when present.
    """
    home = _norm(pick.get("home_team"))
    away = _norm(pick.get("away_team"))
    if not home or not away:
        return None

    def _candidate_names(ev, side):
        full  = _norm(ev.get(f"{side}_team"))
        short = _norm(ev.get(f"{side}_team_short"))
        return [n for n in (full, short) if n]

    # Pass 1: exact or substring match on either direction
    for ev in events:
        if not ev.get("completed"):
            continue
        home_cands = _candidate_names(ev, "home")
        away_cands = _candidate_names(ev, "away")
        home_hit = any(home == c or home in c or c in home for c in home_cands)
        away_hit = any(away == c or away in c or c in away for c in away_cands)
        if home_hit and away_hit:
            return ev

    # Pass 2: last-word match (team mascot) — "white sox" ↔ "white sox",
    # "guardians" ↔ "guardians" even if full names differ
    pick_home_last = home.rsplit(" ", 1)[-1] if " " in home else home
    pick_away_last = away.rsplit(" ", 1)[-1] if " " in away else away
    for ev in events:
        if not ev.get("completed"):
            continue
        for ec in _candidate_names(ev, "home"):
            if pick_home_last in ec:
                for ec2 in _candidate_names(ev, "away"):
                    if pick_away_last in ec2:
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


def _f5_scores(event):
    """Sum home + away runs through the FIRST 5 INNINGS of an MLB game.
    Returns (home_f5, away_f5) or (None, None) if we don't have the data
    or the game didn't make it through 5 innings."""
    innings = event.get("innings") or []
    if not innings:
        return None, None
    h = a = 0
    seen = 0
    for inn in innings:
        if (inn.get("num") or 0) > 5:
            continue
        if inn.get("home") is None and inn.get("away") is None:
            continue
        h += int(inn.get("home") or 0)
        a += int(inn.get("away") or 0)
        seen += 1
    if seen < 5:
        return None, None
    return h, a


def _first_inning_scores(event):
    """Home + away runs in the TOP + BOTTOM of the 1st inning. Returns
    (home_1, away_1) or (None, None) if the inning hasn't played yet."""
    for inn in (event.get("innings") or []):
        if (inn.get("num") or 0) == 1:
            if inn.get("home") is None and inn.get("away") is None:
                return None, None
            return int(inn.get("home") or 0), int(inn.get("away") or 0)
    return None, None


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
    # text_lc keeps the decimal point intact — _norm strips dots, which
    # silently broke Over/Under line parsing on all .5 totals (e.g.
    # "Over 9.5" became "over 95" → line = 95). Any downstream regex that
    # pulls a line out of the pick text must use `text_lc`, not pick_norm.
    text_lc = text.lower()
    is_f5 = ("f5" in market.lower()) or ("first 5" in market.lower())
    is_1h = (("1h" in market.lower() or "first half" in market.lower())
             and "ml" not in market.lower())  # 1H Total specifically

    # ---- NRFI / YRFI (MLB, first-inning runs) ----
    if "NRFI" in market.upper() or "YRFI" in market.upper():
        h1, a1 = _first_inning_scores(event)
        if h1 is None:
            return "pending"
        no_runs_1st = (h1 == 0 and a1 == 0)
        wants_nrfi = "nrfi" in pick_norm or "no runs" in pick_norm
        if wants_nrfi:
            return "W" if no_runs_1st else "L"
        return "L" if no_runs_1st else "W"

    # ---- 1H / P1 Total (NFL / NCAAF / NHL) ----
    # Our schedule-fetch doesn't carry halftime / period-1 scores for these
    # sports (The Odds API only returns final). Mark pending — tracked but
    # not graded until we add a scoreboard source that gives period data.
    if is_1h and ("total" in market.lower()):
        return "pending"

    # ---- F5 ML / F5 Total ----
    if is_f5:
        hf, af = _f5_scores(event)
        if hf is None:
            return "pending"
        if "total" in market.lower():
            m = re.search(r"(over|under)\s*([\d.]+)", text_lc)
            if not m:
                return "pending"
            side = m.group(1)
            try:
                line = float(m.group(2))
            except ValueError:
                return "pending"
            tot = hf + af
            if abs(tot - line) < 1e-6:
                return "P"
            if side == "over":
                return "W" if tot > line else "L"
            return "W" if tot < line else "L"
        # F5 ML — tie after 5 is a push on most books
        if home_name and (pick_norm == home_name or home_name in pick_norm):
            return "W" if hf > af else ("P" if hf == af else "L")
        if away_name and (pick_norm == away_name or away_name in pick_norm):
            return "W" if af > hf else ("P" if hf == af else "L")
        return "pending"

    # ---- Team Total ----
    if "team total" in market.lower():
        m = re.search(r"(over|under)\s*([\d.]+)", text_lc)
        if not m:
            return "pending"
        side = m.group(1)
        try:
            line = float(m.group(2))
        except ValueError:
            return "pending"
        if home_name and home_name in pick_norm:
            team_total = home
        elif away_name and away_name in pick_norm:
            team_total = away
        else:
            return "pending"
        if abs(team_total - line) < 1e-6:
            return "P"
        if side == "over":
            return "W" if team_total > line else "L"
        return "W" if team_total < line else "L"

    # ---- Win to Nil (soccer) — team wins AND opponent scores 0 ----
    if "win to nil" in market.lower() or "clean sheet" in market.lower():
        if home_name and home_name in pick_norm:
            return "W" if (home > away and away == 0) else "L"
        if away_name and away_name in pick_norm:
            return "W" if (away > home and home == 0) else "L"
        return "pending"

    # ---- Double Chance (soccer) — side wins OR draws ----
    if "double chance" in market.lower() or "1x" in pick_norm or "x2" in pick_norm:
        # "home or draw" (1X) or "away or draw" (X2)
        if home_name and home_name in pick_norm:
            return "W" if home >= away else "L"
        if away_name and away_name in pick_norm:
            return "W" if away >= home else "L"
        return "pending"

    # ---- Moneyline (incl. 1H / 1P ML and soccer draws) ----
    if "ML" in market:
        if "draw" in pick_norm:
            return "W" if home == away else "L"
        if home_name and (pick_norm == home_name or home_name in pick_norm):
            return "W" if home > away else ("P" if home == away else "L")
        if away_name and (pick_norm == away_name or away_name in pick_norm):
            return "W" if away > home else ("P" if home == away else "L")

    # ---- Totals (full-game) ----
    if "Total" in market:
        m = re.search(r"(over|under)\s*([\d.]+)", text_lc)
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

    # ---- BTTS ----
    if market == "BTTS":
        if "yes" in pick_norm:
            return "W" if (home > 0 and away > 0) else "L"
        if "no" in pick_norm:
            return "W" if (home == 0 or away == 0) else "L"

    # ---- Spread / Run Line / Puck Line ----
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
        events = fetch_scores(key) if key else []
        # MLB fallback: statsapi.mlb.com works without any API key, so even
        # if ODDS_API_KEY isn't configured on this host, MLB picks still
        # grade. Merge both sources — MLB first (richer team-name coverage).
        if slug == "mlb":
            mlb_events = fetch_scores_mlb(date_str)
            events = mlb_events + events
        events_by_sport[slug] = events

    graded_picks = []
    for p in raw["picks"]:
        # DraftKings-imported bets arrive pre-graded in the CSV (Won / Lost /
        # Push / Open) — we never try to re-grade them against our event feed
        # because DK has markets we don't ingest (props, parlays, player
        # points, etc). Pass them through, keeping the user's bookkeeping.
        if (p.get("source") == "book_dk") and p.get("result") in ("W", "L", "P"):
            graded_picks.append({
                **p,
                "profit_u": p.get("profit_u", profit_at_1u(p.get("result"), p.get("decimal"))),
                "graded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })
            continue

        events = events_by_sport.get(p["sport_slug"], [])
        ev = find_result_event(p, events)
        prior = None
        if existing:
            prior = next((g for g in existing.get("picks", []) if g.get("id") == p.get("id")), None)
        if prior and prior.get("result") in ("W", "L", "P"):
            # Keep prior settled result (Odds API window drops after 3 days)
            graded_picks.append(prior)
            continue
        # Prior was pending — fall through and try to grade again. If we still
        # can't find the game, pending stays pending.
        res = grade_pick(p, ev)
        home_s, away_s = _extract_scores(ev) if ev else (None, None)
        # CLV: how much did Pinnacle's devigged line move from our FIRST
        # snapshot (opening-ish) to the latest save (closing-ish)? Positive
        # pp = market moved toward our side after we picked (sharp signal).
        first_pin = p.get("first_pinnacle_prob")
        close_pin = p.get("pinnacle_prob")
        clv_pp = None
        clv_pct = None  # EV at close using the price we actually got
        if first_pin is not None and close_pin is not None:
            try:
                clv_pp = (float(close_pin) - float(first_pin)) * 100.0
            except (TypeError, ValueError):
                clv_pp = None
        dec = p.get("decimal") or p.get("first_decimal")
        if close_pin is not None and dec:
            try:
                clv_pct = (float(close_pin) * float(dec) - 1.0) * 100.0
            except (TypeError, ValueError):
                clv_pct = None
        graded_picks.append({
            **p,
            "result":     res,
            "profit_u":   profit_at_1u(res, p.get("decimal")),
            "home_score": home_s,
            "away_score": away_s,
            "clv_pp":     clv_pp,
            "clv_ev_pct": clv_pct,
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

def record_in_window(graded_by_date, days_back, strong_only=False, today=None,
                     source=None):
    """Aggregate W/L/P/pending + units + ROI for picks that match `source`.
    source=None → all picks (back-compat).
    source="ev"    → only EV-verified consensus picks.
    source="model" → only model-only picks that scored 4/5 or 5/5.
    source="book_dk" → only picks imported from the user's DK history."""
    today = today or date.today()
    cutoff = today - timedelta(days=days_back - 1)
    w = l = p = pending = 0
    units = 0.0
    clv_pps = []
    clv_evs = []
    clv_positive = 0
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
            if source is not None:
                pk_source = pk.get("source") or ("model" if pk.get("model_only") else "ev")
                if pk_source != source:
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
            # CLV aggregation — only count settled picks (W/L/P) so pending
            # picks don't bias the signal with half-moved lines.
            if r in ("W", "L", "P"):
                cpp = pk.get("clv_pp")
                if cpp is not None:
                    try:
                        v = float(cpp)
                        clv_pps.append(v)
                        if v > 0:
                            clv_positive += 1
                    except (TypeError, ValueError):
                        pass
                cev = pk.get("clv_ev_pct")
                if cev is not None:
                    try:
                        clv_evs.append(float(cev))
                    except (TypeError, ValueError):
                        pass
    settled = w + l
    pct = (w / settled * 100) if settled else 0.0
    roi = (units / settled * 100) if settled else 0.0
    clv_n = len(clv_pps)
    clv_avg_pp = (sum(clv_pps) / clv_n) if clv_n else 0.0
    clv_pos_pct = (clv_positive / clv_n * 100) if clv_n else 0.0
    clv_avg_ev = (sum(clv_evs) / len(clv_evs)) if clv_evs else 0.0
    return {
        "wins": w, "losses": l, "pushes": p, "pending": pending,
        "settled": settled, "win_pct": pct, "units": units, "roi_pct": roi,
        "clv_n": clv_n, "clv_avg_pp": clv_avg_pp,
        "clv_pos_pct": clv_pos_pct, "clv_avg_ev_pct": clv_avg_ev,
    }


def summary(graded_by_date, today=None):
    """Return dict of 7D / 30D / 90D records sliced by subset:
      all_*     → every settled pick (back-compat)
      strong_*  → the ★ strong EV picks
      ev_*      → only consensus/EV-verified picks (source == "ev")
      model_*   → only 4/5-5/5 model-only picks (source == "model")
      dk_*      → only user's imported DraftKings bets (source == "book_dk")
    """
    out = {}
    for label, days in (("7d", 7), ("30d", 30), ("90d", 90)):
        out[f"all_{label}"]    = record_in_window(graded_by_date, days, False, today)
        out[f"strong_{label}"] = record_in_window(graded_by_date, days, True,  today)
        out[f"ev_{label}"]     = record_in_window(graded_by_date, days, False, today, source="ev")
        out[f"model_{label}"]  = record_in_window(graded_by_date, days, False, today, source="model")
        out[f"dk_{label}"]     = record_in_window(graded_by_date, days, False, today, source="book_dk")
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
