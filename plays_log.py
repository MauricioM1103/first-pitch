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
ESPN_SCOREBOARD_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Daily anchor times (Central). Each save that happens on-or-after one of
# these buckets locks a frozen EV + signal + filter judgment on every pick
# that doesn't already have that bucket recorded. The 'curfilter' record
# then uses the EARLIEST fired anchor's frozen judgment per pick instead
# of drifting with live price. Picks the user looks at near these times
# see the same filter membership that gets counted later.
ANCHOR_TIMES_CT = ("00:00", "03:00", "11:00", "16:00")

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


def _now_central():
    """Return current time in America/Chicago. Falls back to UTC on
    systems without zoneinfo (should not happen on Python 3.9+)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        return datetime.utcnow()


def _anchors_due(now_ct):
    """List of ANCHOR_TIMES_CT buckets whose clock time has passed today.
    E.g. at 11:35 CT returns ['00:00', '03:00', '11:00']."""
    out = []
    for bucket in ANCHOR_TIMES_CT:
        hh, mm = bucket.split(":")
        bucket_dt = now_ct.replace(hour=int(hh), minute=int(mm),
                                    second=0, microsecond=0)
        if now_ct >= bucket_dt:
            out.append(bucket)
    return out


def _anchor_snapshot_for_pick(p, now_iso):
    """Snapshot the fields the current-filter judgment depends on, at
    this moment, for later recordkeeping."""
    return {
        "ev_pct":            p.get("ev_pct"),
        "fair_prob":         p.get("fair_prob"),
        "line_move_pp":      p.get("line_move_pp"),
        "signal_score":      _backtest_signal_score(p),
        "passes_curfilter":  _passes_current_filter_live(p),
        "snapshotted_at":    now_iso,
    }


def anchor_status_today():
    """Public: what anchor buckets exist, which have fired already, and
    which comes next in CT. The /logged UI renders this so the user can
    see when picks get frozen for recordkeeping."""
    now_ct = _now_central()
    due = _anchors_due(now_ct)
    next_bucket = None
    next_in_min = None
    for bucket in ANCHOR_TIMES_CT:
        if bucket in due:
            continue
        hh, mm = bucket.split(":")
        bucket_dt = now_ct.replace(hour=int(hh), minute=int(mm),
                                    second=0, microsecond=0)
        delta_min = int((bucket_dt - now_ct).total_seconds() // 60)
        if delta_min >= 0:
            next_bucket = bucket
            next_in_min = delta_min
            break
    if next_bucket is None:
        # All anchors fired today — next one is tomorrow's 00:00
        tomorrow = now_ct.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        next_bucket = "00:00"
        next_in_min = int((tomorrow - now_ct).total_seconds() // 60)
    # Count picks whose first anchor passed the filter — i.e. how many
    # picks would be in the day's current-filter record given anchors
    # already fired.
    anchored_passed = anchored_total = 0
    try:
        picks = (read_picks(now_ct.date().isoformat()) or {}).get("picks") or []
        for pk in picks:
            anchors = pk.get("anchors") or {}
            if not anchors:
                continue
            anchored_total += 1
            first_key = sorted(anchors.keys())[0]
            if (anchors.get(first_key) or {}).get("passes_curfilter"):
                anchored_passed += 1
    except Exception:
        pass
    return {
        "buckets":         list(ANCHOR_TIMES_CT),
        "fired_today":     due,
        "next_bucket":     next_bucket,
        "next_in_min":     next_in_min,
        "anchored_total":  anchored_total,
        "anchored_passed": anchored_passed,
    }


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

    # MERGE with existing picks instead of replacing. The main picks pipeline
    # drops started games via _is_started, so after kickoff the picks list
    # shrinks — if we overwrote blindly, those picks would disappear from the
    # log and nothing graded them. Pick IDs are stable, so new saves update
    # existing entries by id and leave untouched entries in place.
    prev_picks_by_id = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                prev_full = json.load(f)
            for prev_p in prev_full.get("picks", []):
                pid = prev_p.get("id")
                if pid:
                    prev_picks_by_id[pid] = prev_p
        except (OSError, json.JSONDecodeError):
            pass

    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    now_ct = _now_central()
    # Only anchor on the date whose log we're writing — a save for a past
    # date (backfill) shouldn't record anchors against today's clock.
    try:
        _is_todays_log = (date_str == now_ct.date().isoformat())
    except Exception:
        _is_todays_log = True
    due_buckets = _anchors_due(now_ct) if _is_todays_log else []
    merged_by_id = dict(prev_picks_by_id)   # start with everything we had

    # Cross-day line-move backfill: build the 3-day earliest-pinnacle
    # map once, keyed by (sport, home, away, market, pick). Save paths
    # from collect_picks don't attach line_move_pp (that only happens on
    # the board-rows overlay), so without this fallback cross-day moves
    # are invisible and the 2+ icon board filter + Signal composite both
    # lose a point they should earn.
    try:
        _cross_day_hist = line_movement_history(date_str, days_back=3)
    except Exception:
        _cross_day_hist = {}
    for p in picks:
        pid = p.get("id")
        first = existing_first.get(pid, {})
        prev_anchors = (prev_picks_by_id.get(pid) or {}).get("anchors") or {}
        anchors = dict(prev_anchors)
        for bucket in due_buckets:
            if bucket not in anchors:
                anchors[bucket] = _anchor_snapshot_for_pick(p, now_iso)
        # Fallback: every pick gets at least one frozen anchor from its
        # first save, even if no 12/3/11/16 CT bucket fires before the
        # game starts. Keyed "zz_first_save" so lexical sort still prefers
        # any clock-time bucket.
        if not anchors:
            anchors["zz_first_save"] = _anchor_snapshot_for_pick(p, now_iso)
        merged_by_id[pid] = {
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
            # line_move_pp is a Signal composite component (+1 if > 0).
            # Persisting it so the backtest signal score + current-filter
            # judgment on read-back matches what the live board computed
            # at save time. Prior to this, read-back picks always lost
            # the line-move bonus and the curfilter record shrank.
            #
            # If the input pick didn't carry a line_move_pp (common for
            # picks saved via the collect_picks EV path, which doesn't
            # attach the line-history overlay), reconstruct it here from
            # first_pinnacle_prob (earliest observation across the save
            # history for THIS pick) vs current pinnacle_prob. Without
            # this fallback, cross-day line movement was invisible to the
            # 2+ icon board filter and to the Signal composite.
            "line_move_pp":  (
                p.get("line_move_pp")
                if p.get("line_move_pp") is not None
                else (
                    (lambda f, n: (float(n) - float(f))
                                   if (f is not None and n is not None)
                                   else None)(
                        (first.get("first_pinnacle_prob")
                         if first.get("first_pinnacle_prob") is not None
                         else (p.get("first_pinnacle_prob")
                               if p.get("first_pinnacle_prob") is not None
                               else (_cross_day_hist.get(_match_by_market_key(p))
                                     or {}).get("earliest_pinnacle_prob"))),
                        p.get("pinnacle_prob"),
                    )
                )
            ),
            "strong":        bool(p.get("strong")),
            "model_only":    bool(p.get("model_only")),
            "source":        p.get("source") or ("book_dk" if p.get("book") == "draftkings"
                                                   else "model" if p.get("model_only")
                                                   else "ev"),
            "start_time":    p.get("start_time"),
            "snapshotted_at": now_iso,
            # Updated every save while the pick is still in the live
            # picks arg. Compared to start_time at record time to
            # determine whether this pick stayed on until kickoff.
            "last_live_at":  now_iso,
            # CLV capture — first-save snapshot locked in across the day
            "first_snapshot_at":  first.get("first_snapshot_at") or now_iso,
            "first_pinnacle_prob": first.get("first_pinnacle_prob")
                                     if first.get("first_pinnacle_prob") is not None
                                     else p.get("pinnacle_prob"),
            "first_decimal":      first.get("first_decimal") or p.get("decimal"),
            "first_fair_prob":    first.get("first_fair_prob")
                                     if first.get("first_fair_prob") is not None
                                     else p.get("fair_prob"),
            # Anchor snapshots: frozen EV + signal + filter judgment at
            # each daily checkpoint (12 AM / 3 AM / 11 AM / 4 PM CT).
            "anchors":            anchors,
            # AI breakdown fields (shipped 2026-10-24 v3). Preserved so
            # the Discord moderator's resend + the /logged tab's AI
            # record view can rebuild the matchup/factors/risks body
            # without re-querying Claude. Lists stay as lists; strings
            # as strings.
            "ai_verdict":         p.get("ai_verdict"),
            "ai_reason":          p.get("ai_reason"),
            "ai_summary":         p.get("ai_summary"),
            "ai_matchup":         p.get("ai_matchup") or [],
            "ai_factors":         p.get("ai_factors") or [],
            "ai_risks":           p.get("ai_risks") or [],
            # Signal composite — persisted so historical reads see the
            # same score the live board rendered at save time. Fallback
            # to the reconstructed score when the input didn't carry one
            # (collect_picks EV path doesn't attach signal_score). Use
            # the pick dict with the just-computed line_move_pp fallback
            # so the reconstruction sees the real cross-day move.
            "signal_score": (
                p.get("signal_score")
                if p.get("signal_score") is not None
                else _backtest_signal_score({
                    **p,
                    "line_move_pp": (
                        p.get("line_move_pp")
                        if p.get("line_move_pp") is not None
                        else (
                            (lambda f, n: (float(n) - float(f))
                                           if (f is not None and n is not None)
                                           else None)(
                                (first.get("first_pinnacle_prob")
                                 if first.get("first_pinnacle_prob") is not None
                                 else (p.get("first_pinnacle_prob")
                                       if p.get("first_pinnacle_prob") is not None
                                       else (_cross_day_hist.get(_match_by_market_key(p))
                                             or {}).get("earliest_pinnacle_prob"))),
                                p.get("pinnacle_prob"),
                            )
                        )
                    ),
                })
            ),
            "whale":              bool(p.get("whale")),
        }
    # Carry anchors forward for picks that disappeared from the live
    # feed (post-kickoff). The merged_by_id already includes them from
    # prev_picks_by_id, but if a new anchor just fired we also want to
    # capture it against the stored state (not update though — anchors
    # only lock on save when the pick is still live).
    slim = list(merged_by_id.values())
    # Dedup by (sport_slug, home, away, market, pick) tuple. Different
    # emission paths (EV path vs model-fill vs AI vs 1H Goals) sometimes
    # produce the SAME effective pick with DIFFERENT ids — e.g.,
    # EV path saves 'Arsenal ML' via collect_picks, and the AI analysis
    # route saves the same 'Arsenal ML' via /ai-analysis — different
    # ids, same bet. The record counted them twice. Keep the entry
    # with the OLDEST first_snapshot_at (preserves CLV baseline), prefer
    # entries with a book price (decimal/american) over those without.
    def _match_key(pk):
        def _n(x): return (x or "").strip().lower()
        return (_n(pk.get("sport_slug")), _n(pk.get("home_team")),
                _n(pk.get("away_team")), _n(pk.get("market")),
                _n(pk.get("pick")))
    def _entry_rank(pk):
        """Smaller is better. Prefer: has graded result > has book price
        > oldest first_snapshot_at. Picks we already know the outcome of
        beat duplicates that are still 'pending'."""
        has_result = pk.get("result") in ("W", "L", "P")
        has_price  = pk.get("decimal") is not None or pk.get("american") is not None
        snapshot   = pk.get("first_snapshot_at") or pk.get("snapshotted_at") or "9999"
        return (0 if has_result else 1,
                0 if has_price else 1,
                snapshot)
    deduped = {}
    for pk in slim:
        k = _match_key(pk)
        prev = deduped.get(k)
        if prev is None or _entry_rank(pk) < _entry_rank(prev):
            deduped[k] = pk
    slim = list(deduped.values())
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
    _invalidate_caches_on_change()
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


def _match_by_market_key(p):
    """Stable cross-day key for a pick: (sport, home, away, market, pick_label).

    Pick IDs include a random-hash suffix for model-fill picks, so same
    game + same market + same side across days will NOT share an id. Match
    via the market+pick-label tuple instead; game-level identity comes from
    the sport/home/away triple."""
    def _norm(x):
        return (x or "").strip().lower()
    return (
        _norm(p.get("sport_slug") or p.get("sport")),
        _norm(p.get("home_team")),
        _norm(p.get("away_team")),
        _norm(p.get("market")),
        _norm(p.get("pick")),
    )


_LINE_HISTORY_CACHE = {}  # (date_str, days_back) -> (ts, dict)
_LINE_HISTORY_TTL_S = 60   # pick logs change at most once per picks-pipeline run


def line_movement_history(current_date_str, days_back=3):
    """Return {match_key: {"earliest_pinnacle_prob": pct, "earliest_date": "YYYY-MM-DD",
                           "earliest_decimal": dec, "earliest_fair_prob": pct}}
    for every pick we've logged across the last `days_back` days, scanned
    newest-first. The earliest observation across the window is what we
    anchor the "line moved by X pp" column against.

    Why multiple days: a game listed on 2026-10-14 might have first been
    quoted on 2026-10-12 (two days of advance pricing). The intraday
    first_pinnacle_prob only captures movement within the current day,
    missing the 48h of movement BEFORE today's first snapshot.
    """
    try:
        today = datetime.strptime(current_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return {}
    # Hit cache first — reading 4 log files + loose JSON parse adds up on
    # every render. 60s TTL is well below the pick-log write cadence.
    cache_key = (current_date_str, days_back)
    hit = _LINE_HISTORY_CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _LINE_HISTORY_TTL_S:
        return hit[1]
    scan_dates = []
    for i in range(days_back + 1):
        scan_dates.append((today - timedelta(days=i)).isoformat())
    # Walk oldest-first so later-date overwrites don't clobber earlier
    # observations. For each (sport, home, away, market, pick), the first
    # time we see it wins — that's the opening line.
    scan_dates.sort()
    history = {}
    for ds in scan_dates:
        data = read_picks(ds)
        if not data:
            continue
        for p in data.get("picks", []):
            key = _match_by_market_key(p)
            if key in history:
                continue
            # Prefer the day's first_* snapshot. Fall back to live fields
            # if the first_ fields are missing on an older log (early rows
            # before we wired up first_* capture).
            open_pin = p.get("first_pinnacle_prob")
            if open_pin is None:
                open_pin = p.get("pinnacle_prob")
            open_fair = p.get("first_fair_prob")
            if open_fair is None:
                open_fair = p.get("fair_prob")
            open_dec = p.get("first_decimal") or p.get("decimal")
            # Skip entries with no pricing anchor — they can't contribute
            # to a delta.
            if open_pin is None and open_fair is None:
                continue
            history[key] = {
                "earliest_pinnacle_prob": open_pin,
                "earliest_fair_prob":     open_fair,
                "earliest_decimal":       open_dec,
                "earliest_date":          ds,
                "earliest_snapshot_at":   p.get("first_snapshot_at") or p.get("snapshotted_at"),
            }
    _LINE_HISTORY_CACHE[cache_key] = (time.time(), history)
    # Keep cache bounded; only 2-3 keys realistic
    if len(_LINE_HISTORY_CACHE) > 10:
        oldest = sorted(_LINE_HISTORY_CACHE.items(), key=lambda kv: kv[1][0])[:5]
        for k, _ in oldest:
            _LINE_HISTORY_CACHE.pop(k, None)
    return history


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
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    # Dedup on read — historical graded files may carry doubles from
    # before the save-path dedup landed. Keep best-ranked entry per
    # (sport, home, away, market, pick) tuple so record math is honest.
    picks = data.get("picks") or []
    if picks:
        def _n(x): return (x or "").strip().lower()
        def _key(pk):
            return (_n(pk.get("sport_slug")), _n(pk.get("home_team")),
                    _n(pk.get("away_team")), _n(pk.get("market")),
                    _n(pk.get("pick")))
        def _rank(pk):
            has_result = pk.get("result") in ("W", "L", "P")
            has_price  = pk.get("decimal") is not None or pk.get("american") is not None
            snapshot   = pk.get("first_snapshot_at") or pk.get("snapshotted_at") or "9999"
            return (0 if has_result else 1,
                    0 if has_price else 1,
                    snapshot)
        best = {}
        for pk in picks:
            k = _key(pk)
            prev = best.get(k)
            if prev is None or _rank(pk) < _rank(prev):
                best[k] = pk
        data["picks"] = list(best.values())
    return data


def write_graded(date_str, graded):
    text = json.dumps(graded)
    # No-op diff: skip the disk write + GitHub push if nothing changed.
    # Matters because grade_date(rescore=True) runs on every main-board
    # visit for today's date — without this check we'd commit to the
    # data branch dozens of times a day with identical content.
    path = graded_file(date_str)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                prev_text = f.read()
            if prev_text == text:
                return
        except OSError:
            pass
    try:
        with open(path, "w") as f:
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


# Last error per sport key so grader_status can surface it on /logged.
# Kept in-process; resets on worker restart (same lifetime as _RESULTS_CACHE).
_RESULTS_LAST_ERROR = {}
_ESPN_CACHE = {}
_ESPN_TTL_S = 10 * 60   # scores refresh every 10 min — ESPN doesn't rate-limit at this rate


def _parse_espn_events(data):
    """Convert an ESPN scoreboard JSON payload into the event dict shape
    the rest of the grader (find_result_event / _extract_scores / grade_pick)
    already understands. Picks up period/half scores too so NFL 1H Total,
    NHL P1 Total, F5 ML/Total etc. have the raw data they need."""
    events_out = []
    for ev in (data or {}).get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (comp.get("status") or {}).get("type") or {}
        completed = bool(status.get("completed")) or (status.get("state") == "post")
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_team = (home.get("team") or {}).get("displayName") or ""
        away_team = (away.get("team") or {}).get("displayName") or ""
        home_short = (home.get("team") or {}).get("shortDisplayName") or (home.get("team") or {}).get("name") or ""
        away_short = (away.get("team") or {}).get("shortDisplayName") or (away.get("team") or {}).get("name") or ""
        home_score = home.get("score")
        away_score = away.get("score")
        # Period-level scores (quarters / periods / innings) for finer grading.
        # ESPN exposes them as a list under competitor.linescores, one entry
        # per period in order. We align them into num/home/away triples so
        # grade_pick's F5/NRFI/1H helpers can consume them like statsapi.
        home_ls = home.get("linescores") or []
        away_ls = away.get("linescores") or []
        innings = []
        for i, pair in enumerate(zip(home_ls, away_ls), start=1):
            h_ls, a_ls = pair
            try:
                innings.append({
                    "num":  i,
                    "home": int(float(h_ls.get("value"))) if h_ls.get("value") is not None else None,
                    "away": int(float(a_ls.get("value"))) if a_ls.get("value") is not None else None,
                })
            except (TypeError, ValueError):
                continue
        events_out.append({
            "id":                str(ev.get("id") or ""),
            "home_team":         home_team,
            "away_team":         away_team,
            "home_team_short":   home_short,
            "away_team_short":   away_short,
            "completed":         completed,
            "scores": [
                {"name": home_team, "score": str(home_score) if home_score is not None else "0"},
                {"name": away_team, "score": str(away_score) if away_score is not None else "0"},
            ],
            "innings":           innings,
            "commence_time":     comp.get("date"),
        })
    return events_out


def fetch_scores_espn(paths, days_back=3, ref_date=None):
    """Fetch final scores from ESPN's public scoreboard JSON. `paths` is a
    sport-path string ("football/nfl") or a list of them (for aggregated
    sports like international soccer).

    No API key, no rate-limit meter, no monthly cap. Caches in-process for
    10 min. Walks `days_back` calendar days backward from `ref_date`
    (default: today), hitting `?dates=YYYYMMDD` per day. Pass `ref_date`
    when backfilling grades for older pending picks whose game date falls
    outside the default today-anchored window.
    """
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    ref_date = ref_date or date.today()
    cache_key = ("espn", tuple(paths), days_back, ref_date.isoformat())
    now = time.time()
    hit = _ESPN_CACHE.get(cache_key)
    if hit:
        events, saved_at = hit
        ttl = _ESPN_TTL_S if events else 60
        if now - saved_at < ttl:
            return events
    all_events = []
    seen_ids = set()
    cutoff_dates = [(ref_date - timedelta(days=d)).strftime("%Y%m%d")
                    for d in range(days_back + 1)]
    for path in paths:
        for d in cutoff_dates:
            url = f"{ESPN_SCOREBOARD_BASE}/{path}/scoreboard?dates={d}"
            try:
                data = _fetch_json(url)
            except Exception as e:
                _RESULTS_LAST_ERROR[path] = f"{type(e).__name__}: {e}"
                continue
            for ev in _parse_espn_events(data):
                if ev["id"] and ev["id"] not in seen_ids:
                    seen_ids.add(ev["id"])
                    all_events.append(ev)
            _RESULTS_LAST_ERROR.pop(path, None)
    # Soccer scoreboard omits half-time linescores. For completed soccer
    # events with empty innings, pull the per-event summary (one call per
    # game, cached in process) to populate the first-half scores so 1H
    # Total picks grade cleanly instead of staying pending forever.
    _is_soccer = any("soccer" in p for p in paths)
    if _is_soccer:
        for ev in all_events:
            if ev.get("innings"):
                continue
            if not ev.get("completed"):
                continue
            # Pick the first soccer path that matches this event's league
            # (we only have one path when it's a specific league; for
            # international aggregates we try each until one answers).
            for path in paths:
                if "soccer" not in path:
                    continue
                h1_h, h1_a = _fetch_espn_half_scores(path, ev["id"])
                if h1_h is not None and h1_a is not None:
                    ev["innings"] = [{"num": 1, "home": h1_h, "away": h1_a}]
                    break
    _ESPN_CACHE[cache_key] = (all_events, now)
    # Bounded LRU: cap at 64 entries, drop oldest half when exceeded.
    # Over a long-running process the key space (sport × days_back ×
    # ref_date) can accumulate 300+ entries across a 30-day grading
    # window. Keep it bounded so the cache doesn't become a leak.
    if len(_ESPN_CACHE) > 64:
        for k in sorted(_ESPN_CACHE.items(), key=lambda kv: kv[1][1])[:32]:
            _ESPN_CACHE.pop(k[0], None)
    return all_events


_ESPN_HALF_CACHE = {}  # (path, event_id) -> (home_1h, away_1h, ts)
_ESPN_HALF_TTL_S = 24 * 60 * 60   # final scores never change

def _fetch_espn_half_scores(path, event_id):
    """Fetch first-half scores from ESPN's event summary endpoint.
    Returns (home_1h, away_1h) or (None, None) on failure. Soccer-only;
    the ESPN scoreboard omits period-level data for soccer but the
    summary endpoint carries it via competitions[0].competitors[].linescores.
    Cached 24h — final scores are immutable."""
    if not event_id:
        return None, None
    key = (path, event_id)
    hit = _ESPN_HALF_CACHE.get(key)
    now = time.time()
    if hit and (now - hit[2]) < _ESPN_HALF_TTL_S:
        return hit[0], hit[1]
    url = f"{ESPN_SCOREBOARD_BASE}/{path}/summary?event={event_id}"
    try:
        data = _fetch_json(url, timeout=10)
    except Exception:
        return None, None
    try:
        header = (data or {}).get("header") or {}
        comps  = header.get("competitions") or []
        if not comps:
            return None, None
        comp = comps[0]
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None, None
        h_ls = home.get("linescores") or []
        a_ls = away.get("linescores") or []
        # Soccer periods: linescores[0] is first-half, linescores[1] is second-half.
        if not h_ls or not a_ls:
            return None, None
        try:
            h1_h = int(float(h_ls[0].get("value", 0)))
            h1_a = int(float(a_ls[0].get("value", 0)))
        except (TypeError, ValueError):
            return None, None
        _ESPN_HALF_CACHE[key] = (h1_h, h1_a, now)
        # Bounded: 128 entries is 128 completed soccer games — plenty
        # headroom over a season but prevents unbounded growth.
        if len(_ESPN_HALF_CACHE) > 128:
            for k in sorted(_ESPN_HALF_CACHE.items(), key=lambda kv: kv[1][2])[:64]:
                _ESPN_HALF_CACHE.pop(k[0], None)
        return h1_h, h1_a
    except Exception:
        return None, None


def fetch_scores(sport_key, days_from=3):
    """Pull final scores from The Odds API. Returns list of event dicts.
    Returns [] when ODDS_API_KEY isn't set — the grader then uses sport-
    specific fallbacks (fetch_scores_mlb for MLB).

    Empty responses cache for only 2 minutes (vs 30 min for non-empty) so
    a transient API hiccup during the first grader pass doesn't lock us
    out of grading for half an hour. Also records the last HTTP error per
    sport so /logged can show WHY a sport is sitting at zero events.
    """
    if not sport_key:
        return []
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        _RESULTS_LAST_ERROR[sport_key] = "ODDS_API_KEY not set"
        return []
    now = time.time()
    hit = _RESULTS_CACHE.get(sport_key)
    if hit:
        events, saved_at = hit
        ttl = _RESULTS_TTL_S if events else 120
        if now - saved_at < ttl:
            return events
    params = urlencode({"apiKey": key, "daysFrom": days_from})
    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores/?{params}"
    try:
        events = _fetch_json(url) or []
        _RESULTS_LAST_ERROR.pop(sport_key, None)
    except URLError as e:
        reason = getattr(e, "reason", str(e))
        code = getattr(e, "code", None)
        _RESULTS_LAST_ERROR[sport_key] = f"HTTP {code}: {reason}" if code else str(reason)
        events = []
    except (ValueError, TimeoutError, ConnectionError, OSError) as e:
        _RESULTS_LAST_ERROR[sport_key] = f"{type(e).__name__}: {e}"
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
    """Diagnostic for /logged: which graders are available right now AND how
    many events each one is currently returning. Previously only reported
    whether the ODDS_API_KEY env var was set — which hid the common failure
    mode where the key is set but the API returns empty for a given sport
    (match not in the fetched league, API coverage gap, rate-limited, etc).

    `espn_enabled` is True when any configured sport has an ESPN scoreboard
    path — ESPN covers every sport in sports.py as of the ESPN switch, so
    the user's "ODDS_API_KEY not set" warning is now obsolete even when the
    key is missing. The UI banner uses this to say "ESPN primary" instead
    of warning that NFL/NCAAF/NHL picks will stay pending."""
    import sports as _sports
    out = {
        "odds_api_enabled": bool(os.environ.get("ODDS_API_KEY")),
        "mlb_statsapi":     True,
        "espn_enabled":     any(sp.get("espn_scoreboard") for sp in _sports.SPORTS),
        "per_sport":        {},
    }
    # Count events currently cached per sport key (reflects last fetch).
    # Also show mlb statsapi event count for today.
    for sp in _sports.SPORTS:
        slug = sp["slug"]
        key = sp.get("odds_api_key")
        espn = sp.get("espn_scoreboard")
        n_events = 0
        source = "none"
        last_err = None
        if slug == "mlb":
            try:
                today_str = date.today().isoformat()
                n_events = len(fetch_scores_mlb(today_str))
                source = "statsapi"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
        if espn:
            try:
                espn_events = len(fetch_scores_espn(espn))
                n_events = max(n_events, espn_events)
                if source == "none":
                    source = "espn"
                else:
                    source = f"{source}+espn"
                # Surface any per-path ESPN error
                paths = espn if isinstance(espn, list) else [espn]
                for p in paths:
                    if p in _RESULTS_LAST_ERROR:
                        last_err = _RESULTS_LAST_ERROR[p]
                        break
            except Exception as e:
                last_err = last_err or f"{type(e).__name__}: {e}"
        # Only surface an Odds API error when ESPN isn't covering this
        # sport (i.e. no espn_scoreboard configured). ESPN is the primary
        # grader for every sport in sports.py, so "ODDS_API_KEY not set"
        # is noise, not a real failure, whenever ESPN exists for the sport.
        if n_events == 0 and key and not last_err and not espn:
            last_err = _RESULTS_LAST_ERROR.get(key)
        out["per_sport"][slug] = {
            "source":     source,
            "events":     n_events,
            "api_key":    key or "(none)",
            "espn_path":  espn or "(none)",
            "last_error": last_err,
        }
    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _norm(s):
    """Normalize a team name for fuzzy matching. Strips accents (ASCII
    fold) so ESPN's "Atlético de San Luis" matches our pick's "Atletico
    San Luis". Lowercases, trims punctuation, collapses whitespace."""
    import unicodedata as _u
    s = (s or "")
    # NFD split lets us drop combining marks (accents) with a filter.
    s = _u.normalize("NFD", s)
    s = "".join(ch for ch in s if not _u.combining(ch))
    s = s.lower().strip().replace(".", "").replace("-", " ")
    # Collapse multiple spaces to one — tokens line up for overlap check.
    s = " ".join(s.split())
    return s


def _name_tokens(s):
    """Normalized name split into significant tokens — drops stop words
    like 'de', 'la', 'fc', 'cf', 'ac' that bloat Spanish/Italian/Latam
    club names without disambiguating. Lets us token-overlap-match
    'Atletico San Luis' against 'Atletico de San Luis' cleanly."""
    stop = {"de", "la", "el", "los", "las", "fc", "cf", "sc", "ac", "cd",
            "cdf", "club", "deportivo"}
    toks = _norm(s).split()
    toks = [t for t in toks if t and t not in stop]
    # Expand common NCAAF / mascot abbreviations so our Pinnacle-style
    # "Central Florida" picks match ESPN's "UCF Knights" and vice versa.
    expanded = set(toks)
    for t in toks:
        for alias in _TEAM_ALIAS.get(t, ()):
            expanded.update(alias.split())
    return list(expanded)


# Bidirectional abbreviation table. Each key maps to every token form the
# same team might appear under across Pinnacle vs ESPN. Lowercase, accent-
# stripped already (because this is applied after _norm).
_TEAM_ALIAS = {
    # NCAAF
    "ucf":       ("central", "florida"),
    "central":   ("ucf",),
    "usc":       ("southern", "california"),
    "lsu":       ("louisiana", "state"),
    "ucla":      ("california", "los", "angeles"),
    "byu":       ("brigham", "young"),
    "smu":       ("southern", "methodist"),
    "tcu":       ("texas", "christian"),
    "unlv":      ("nevada", "las", "vegas"),
    "fiu":       ("florida", "international"),
    "fau":       ("florida", "atlantic"),
    "ulm":       ("louisiana", "monroe"),
    # Common ESPN mascot-only -> city expansions
    "cowboys":   ("oklahoma", "state", "dallas"),  # NCAAF OKstate OR NFL Dallas
    "knights":   ("ucf", "central", "florida"),
    # Ole Miss <-> Mississippi (ESPN uses "Ole Miss Rebels", Pinnacle uses "Mississippi")
    "ole":       ("mississippi",),
    "miss":      ("mississippi",),
    "mississippi": ("ole", "miss"),
    # San Jose State / San Jose (Pinnacle often "San Jose State", ESPN too but accents vary)
    "jose":      ("san",),
    # Hawaii Rainbow Warriors <-> just "Hawaii"
    "rainbow":   ("hawaii",),
    "warriors":  ("hawaii",),
    # Common Liga MX / La Liga doesn't need more — stop-word strip + token
    # overlap covers them. Keep this table NCAAF-centric.
}


def _event_date_close_to_pick(ev, pick, hours_tolerance=12):
    """True iff the event's commence_time is within `hours_tolerance` of
    the pick's start_time. Guards against matching a pick to the SAME
    TEAMS' different game — e.g. MLB divisional series where the same
    two teams play 3+ games in a week (Mon/Tue/Wed), each ~24h apart.
    With a 36h window those back-to-back-day games used to collide
    and today's not-yet-played pick showed yesterday's W/L. 12h is
    tight enough to reject day-apart games while still allowing
    doubleheaders (both games <= 6h apart). Returns True if either
    side has no timestamp (fall through to legacy name-only match)."""
    pst = pick.get("start_time")
    est = ev.get("commence_time")
    if not pst or not est:
        return True
    try:
        pdt = datetime.fromisoformat(pst.replace("Z", "+00:00"))
        edt = datetime.fromisoformat(est.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return True
    delta_h = abs((pdt - edt).total_seconds()) / 3600.0
    return delta_h <= hours_tolerance


def find_result_event(pick, events):
    """Match a pick to a scored event — handles short vs full team names
    (e.g. pick carries "White Sox", event has "Chicago White Sox"). Also
    checks statsapi's short-name field when present.

    Prefers an event whose commence_time is within 36h of the pick's start
    time — this is the fix for "results posted for games that have not
    started": without a date filter, a pick for an upcoming game between
    two teams who just played last week would match the past result.
    """
    home = _norm(pick.get("home_team"))
    away = _norm(pick.get("away_team"))
    if not home or not away:
        return None

    def _candidate_names(ev, side):
        full  = _norm(ev.get(f"{side}_team"))
        short = _norm(ev.get(f"{side}_team_short"))
        return [n for n in (full, short) if n]

    # Pass 1a: date-aligned exact/substring match (preferred)
    for ev in events:
        if not ev.get("completed"):
            continue
        if not _event_date_close_to_pick(ev, pick):
            continue
        home_cands = _candidate_names(ev, "home")
        away_cands = _candidate_names(ev, "away")
        home_hit = any(home == c or home in c or c in home for c in home_cands)
        away_hit = any(away == c or away in c or c in away for c in away_cands)
        if home_hit and away_hit:
            return ev

    # Pass 1b: date-aligned mascot-only match (last-word of each team name)
    pick_home_last = home.rsplit(" ", 1)[-1] if " " in home else home
    pick_away_last = away.rsplit(" ", 1)[-1] if " " in away else away
    for ev in events:
        if not ev.get("completed"):
            continue
        if not _event_date_close_to_pick(ev, pick):
            continue
        for ec in _candidate_names(ev, "home"):
            if pick_home_last in ec:
                for ec2 in _candidate_names(ev, "away"):
                    if pick_away_last in ec2:
                        return ev

    # Pass 1c: date-aligned token-overlap match. Covers soccer clubs where
    # ESPN includes stopwords (Atlético **de** San Luis) and our pick does
    # not (Atletico San Luis), and short-vs-full variants (Juárez vs FC
    # Juárez). Requires at least one significant token to overlap on each
    # side so random noise doesn't match.
    pick_home_toks = set(_name_tokens(pick.get("home_team")))
    pick_away_toks = set(_name_tokens(pick.get("away_team")))
    for ev in events:
        if not ev.get("completed"):
            continue
        if not _event_date_close_to_pick(ev, pick):
            continue
        ev_home_toks = set(_name_tokens(ev.get("home_team"))) | set(_name_tokens(ev.get("home_team_short")))
        ev_away_toks = set(_name_tokens(ev.get("away_team"))) | set(_name_tokens(ev.get("away_team_short")))
        # Try both home/away orientations — Pinnacle and ESPN disagree on
        # which side is "home" for a chunk of La Liga / Liga MX fixtures.
        # For moneyline / spread / total grading we only need the correct
        # team-to-score mapping, which grade_pick already handles.
        if pick_home_toks & ev_home_toks and pick_away_toks & ev_away_toks:
            return ev
        if pick_home_toks & ev_away_toks and pick_away_toks & ev_home_toks:
            return ev

    # Pass 2: legacy name-only match. ONLY used for picks without a
    # start_time (can't do date-aware matching anyway) — picks that DO
    # have a start_time but didn't find a date-aligned match stay pending
    # rather than being matched against a different week's same-teams game.
    if pick.get("start_time"):
        return None
    for ev in events:
        if not ev.get("completed"):
            continue
        home_cands = _candidate_names(ev, "home")
        away_cands = _candidate_names(ev, "away")
        home_hit = any(home == c or home in c or c in home for c in home_cands)
        away_hit = any(away == c or away in c or c in away for c in away_cands)
        if home_hit and away_hit:
            return ev
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


# ESPN's scoreboard endpoint omits halftime splits for soccer — all
# we get is the full-time score. For 1H Total picks on soccer, we
# have to hit the per-event summary endpoint which DOES include
# linescores: [first_half, second_half] per competitor.
_ESPN_SUMMARY_CACHE = {}      # (sport_slug, event_id) -> ((h_h1, a_h1), ts)
_ESPN_SUMMARY_TTL_S = 60 * 60 # 1 hr — final linescores never change

_ESPN_SOCCER_SUMMARY_PATH = {
    "epl":           "soccer/eng.1",
    "laliga":        "soccer/esp.1",
    "ligamx":        "soccer/mex.1",
    "ucl":           "soccer/uefa.champions",
    "europa":        "soccer/uefa.europa",
    "international": "soccer/fifa.worldq.conmebol",   # fallback best-guess
}


def fetch_soccer_halftime(sport_slug, event_id):
    """Return (home_h1, away_h1) for a completed soccer match, or
    (None, None) if ESPN didn't return halftime data. Caches 1 hour.

    Needed because fetch_scores_espn's scoreboard pass doesn't include
    linescores for soccer (fullish score only), so grade_pick's 1H Total
    branch has no first-half data to compute from without a second
    HTTP call per event."""
    if not event_id:
        return (None, None)
    slug = (sport_slug or "").lower()
    path = _ESPN_SOCCER_SUMMARY_PATH.get(slug)
    if not path:
        return (None, None)
    key = (slug, str(event_id))
    now = time.time()
    hit = _ESPN_SUMMARY_CACHE.get(key)
    if hit and (now - hit[1]) < _ESPN_SUMMARY_TTL_S:
        return hit[0]
    url = (f"https://site.api.espn.com/apis/site/v2/sports/{path}"
           f"/summary?event={event_id}")
    try:
        data = _fetch_json(url) or {}
    except (URLError, ValueError, TimeoutError, ConnectionError, OSError):
        _ESPN_SUMMARY_CACHE[key] = ((None, None), now)
        return (None, None)
    sc = (data.get("header") or {}).get("competitions") or []
    if not sc:
        _ESPN_SUMMARY_CACHE[key] = ((None, None), now)
        return (None, None)
    h1 = {"home": None, "away": None}
    for c in sc[0].get("competitors", []):
        ls = c.get("linescores") or []
        if not ls:
            continue
        try:
            # linescores[0] = first half, linescores[1] = second half
            h1_score = int(float(ls[0].get("displayValue")))
        except (TypeError, ValueError, IndexError):
            continue
        side = "home" if c.get("homeAway") == "home" else "away"
        h1[side] = h1_score
    result = (h1["home"], h1["away"])
    _ESPN_SUMMARY_CACHE[key] = (result, now)
    return result


def grade_pick(pick, event):
    """Return 'W' | 'L' | 'P' | 'pending' for the pick given the result event.

    POLICY (shipped 2026-10-23): any would-be push (exact-margin spread,
    exact total, F5 moneyline draw, team total hit the line, etc.) is
    reported as 'L' here, not 'P'. This matches the pick-emission
    policy that only puts .5-fractional lines on the board, so the
    only way a pick CAN push now is on a stale historical pick that
    was logged before the .5-only rule, or on an F5 ML draw. In both
    cases the user wants the record to reflect "you did not win" rather
    than a push refund, so we grade it as a loss.

    Internal pushes (exact ties) still go through the same branches
    below for determinism; we just rewrite 'P' -> 'L' at the exit."""
    def _no_push(r):
        return "L" if r == "P" else r
    if not event:
        return "pending"
    # Belt-and-suspenders: find_result_event already filters to
    # completed events, but any future caller that bypasses it would
    # otherwise grade full-game markets off a live scoreline. If the
    # event itself isn't marked completed, bail out — unless it's a
    # sub-period market (1H / F5 / NRFI / YRFI / 1P) which legitimately
    # settles before the final whistle.
    market = pick.get("market", "") or ""
    _market_lc = market.lower()
    _is_subperiod = (
        "1h" in _market_lc or "first half" in _market_lc
        or "f5" in _market_lc or "first 5" in _market_lc
        or "nrfi" in _market_lc or "yrfi" in _market_lc
        or "1p" in _market_lc or "p1" in _market_lc
    )
    if not event.get("completed") and not _is_subperiod:
        return "pending"
    home, away = _extract_scores(event)
    if home is None or away is None:
        return "pending"

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

    # ---- 1H / P1 Total (NFL / NCAAF / NHL / soccer) ----
    # ESPN linescores are parsed into event["innings"] = [{num, home, away}].
    # For NFL/NCAAF, "1H Total" = Q1 + Q2 for both teams.
    # For NHL, "P1 Total" = period 1 goals (both teams).
    # For soccer, "1H Total" = half 1 goals (both teams).
    if is_1h and ("total" in market.lower()):
        innings_ls = event.get("innings") or []
        slug = (pick.get("sport_slug") or "").lower()
        _SOCCER_SLUGS = ("epl", "laliga", "ligamx", "ucl", "europa", "international")
        if slug in ("nfl", "ncaaf"):
            if not innings_ls:
                return "pending"
            # 4 quarters — need the first two
            q1 = next((inn for inn in innings_ls if inn.get("num") == 1), None)
            q2 = next((inn for inn in innings_ls if inn.get("num") == 2), None)
            if q1 is None or q2 is None:
                return "pending"
            h_1h = (q1.get("home") or 0) + (q2.get("home") or 0)
            a_1h = (q1.get("away") or 0) + (q2.get("away") or 0)
        elif slug in _SOCCER_SLUGS:
            # ESPN's soccer scoreboard omits halftime splits, so
            # innings_ls is typically empty for soccer. Fetch the
            # per-event summary which does carry linescores.
            # (Cached 1h — final linescores are immutable.)
            h_h1, a_h1 = fetch_soccer_halftime(slug, event.get("id"))
            if h_h1 is None or a_h1 is None:
                return "pending"
            h_1h, a_1h = h_h1, a_h1
        else:
            if not innings_ls:
                return "pending"
            # NHL period 1 — first linescore entry
            p1 = next((inn for inn in innings_ls if inn.get("num") == 1), None)
            if p1 is None or p1.get("home") is None:
                return "pending"
            h_1h = p1.get("home") or 0
            a_1h = p1.get("away") or 0
        tot_1h = h_1h + a_1h
        m = re.search(r"(over|under)\s*([\d.]+)", text_lc)
        if not m:
            return "pending"
        side = m.group(1)
        try:
            line = float(m.group(2))
        except ValueError:
            return "pending"
        if abs(tot_1h - line) < 1e-6:
            return "L"  # policy: pushes count as losses (see grade_pick docstring)
        if side == "over":
            return "W" if tot_1h > line else "L"
        return "W" if tot_1h < line else "L"

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
                return "L"  # policy: pushes count as losses (see grade_pick docstring)
            if side == "over":
                return "W" if tot > line else "L"
            return "W" if tot < line else "L"
        # F5 ML — tie after 5 is a push on most books
        if home_name and (pick_norm == home_name or home_name in pick_norm):
            return "W" if hf > af else "L"  # push → L per policy
        if away_name and (pick_norm == away_name or away_name in pick_norm):
            return "W" if af > hf else "L"  # push → L per policy
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
            return "L"  # policy: pushes count as losses (see grade_pick docstring)
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

    # ---- Draw No Bet (soccer) — graded as "team or draw" (Double Chance
    # 1X / X2) per user's betting style: they bet the market as "No on
    # [other team]", so a draw counts as a Win, not a stake refund.
    # "Shakhtar Donetsk (DNB)" wins on Shakhtar win OR draw; "AEK Athens
    # (DNB)" wins on AEK win OR draw. ----
    if "draw no bet" in market.lower() or market == "DNB":
        # Strip the (DNB) marker so team-name substring match works.
        tag_stripped = re.sub(r"\s*\(dnb\)\s*", "", pick_norm)
        if home_name and (tag_stripped == home_name or home_name in tag_stripped):
            return "W" if home >= away else "L"
        if away_name and (tag_stripped == away_name or away_name in tag_stripped):
            return "W" if away >= home else "L"
        return "pending"

    # ---- Moneyline (incl. 1H / 1P ML and soccer draws) ----
    if "ML" in market:
        if "draw" in pick_norm:
            return "W" if home == away else "L"
        if home_name and (pick_norm == home_name or home_name in pick_norm):
            return "W" if home > away else "L"  # push → L per policy
        if away_name and (pick_norm == away_name or away_name in pick_norm):
            return "W" if away > home else "L"  # push → L per policy

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
                return "L"  # policy: pushes count as losses (see grade_pick docstring)
            if side == "over":
                return "W" if tot > line else "L"
            return "W" if tot < line else "L"

    # ---- BTTS ----
    if market == "BTTS":
        if "yes" in pick_norm:
            return "W" if (home > 0 and away > 0) else "L"
        if "no" in pick_norm:
            return "W" if (home == 0 or away == 0) else "L"

    # ---- Correct Score ("Correct Score X-Y") ----
    # Pick is a single exact scoreline; wins only when both final-score
    # components match.
    if "correct score" in market.lower() or market == "CS":
        m = re.search(r"(\d+)\s*[-:]\s*(\d+)", text)
        if not m:
            return "pending"
        try:
            pick_h = int(m.group(1))
            pick_a = int(m.group(2))
        except ValueError:
            return "pending"
        return "W" if (home == pick_h and away == pick_a) else "L"

    # ---- BTTS + Total combo (BTTS & Over/Under 2.5) ----
    # Pick text is "BTTS Yes & Over 2.5" (or Under / No variants).
    # Wins only when BOTH the BTTS side AND the total side resolve the
    # bettor's way. Pick label contains both halves — parse them.
    if ("btts & total" in market.lower() or "btts & " in market.lower()
            or market == "BTTS & Total"):
        wants_yes = "yes" in pick_norm
        wants_over = "over" in pick_norm
        m = re.search(r"(over|under)\s*([\d.]+)", text_lc)
        if not m:
            return "pending"
        try:
            line = float(m.group(2))
        except ValueError:
            return "pending"
        btts_yes = (home > 0 and away > 0)
        tot = home + away
        total_over = tot > line
        want_btts_hit = (btts_yes == wants_yes)
        want_total_hit = (total_over == wants_over)
        if abs(tot - line) < 1e-6:
            return "L"  # policy: pushes count as losses (see grade_pick docstring)
        return "W" if (want_btts_hit and want_total_hit) else "L"

    # ---- Spread / Run Line / Puck Line (incl. Alt Spread, 1H/1P Spread) ----
    # Substring match catches "Alt Spread", "Alt Run Line", "1H Spread",
    # "1P Spread", etc. — the previous exact-list check missed Alt Spread
    # and left all those picks pending indefinitely.
    _mlo = market.lower()
    if ("spread" in _mlo or "run line" in _mlo or "puck line" in _mlo
            or "runline" in _mlo or "puckline" in _mlo):
        # Extract the signed line that follows the TEAM name in the pick
        # text. The previous regex grabbed the FIRST signed number which,
        # for alt-line labels like "Giants +7.0 (ALT · +3.0 pts easier
        # vs main Giants +4.0)", would still work (first number = the
        # alt line itself) — but it's safer to look for a +/-number that
        # immediately follows a team name. Fall back to first-signed.
        line = None
        if home_name:
            mh = re.search(r"(?:" + re.escape(home_name) + r")\s*([+-][\d.]+)",
                           text_lc)
            if mh:
                try: line = float(mh.group(1))
                except ValueError: pass
        if line is None and away_name:
            ma = re.search(r"(?:" + re.escape(away_name) + r")\s*([+-][\d.]+)",
                           text_lc)
            if ma:
                try: line = float(ma.group(1))
                except ValueError: pass
        if line is None:
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
        # For a 1H / P1 spread, use first-period margin instead of final.
        if "1h" in _mlo or "1p" in _mlo:
            innings_ls = event.get("innings") or []
            slug = (pick.get("sport_slug") or "").lower()
            if slug in ("nfl", "ncaaf"):
                q1 = next((inn for inn in innings_ls if inn.get("num") == 1), None)
                q2 = next((inn for inn in innings_ls if inn.get("num") == 2), None)
                if q1 and q2:
                    h_1h = (q1.get("home") or 0) + (q2.get("home") or 0)
                    a_1h = (q1.get("away") or 0) + (q2.get("away") or 0)
                    if home_name and home_name in pick_norm:
                        margin = (h_1h - a_1h) + line
                    elif away_name and away_name in pick_norm:
                        margin = (a_1h - h_1h) + line
            else:
                p1 = next((inn for inn in innings_ls if inn.get("num") == 1), None)
                if p1 and p1.get("home") is not None:
                    h1 = p1.get("home") or 0
                    a1 = p1.get("away") or 0
                    if home_name and home_name in pick_norm:
                        margin = (h1 - a1) + line
                    elif away_name and away_name in pick_norm:
                        margin = (a1 - h1) + line
        if abs(margin) < 1e-6:
            return "L"  # policy: pushes count as losses (see grade_pick docstring)
        return "W" if margin > 0 else "L"

    return "pending"


def profit_at_1u(result, decimal, fair_prob=None):
    """Units profit at a 1-unit stake.

    When we have a book `decimal`, use it directly. When the pick is a
    model-only market we don't fetch prices for (NRFI, Win to Nil, soccer
    BTTS, Team Total, etc.), fall back to the model's fair probability to
    derive breakeven odds (dec = 1 / fair_prob). That gives a zero-EV
    reference line — any W/L streak on these picks shows up as a non-zero
    units curve so /logged ROI reflects conviction accuracy, not just
    "we didn't attach a price".
    """
    try:
        d = float(decimal or 0)
    except (TypeError, ValueError):
        d = 0
    if d <= 1 and fair_prob is not None:
        try:
            fp = float(fair_prob)
            if 0.01 < fp < 0.99:
                d = 1.0 / fp   # breakeven implied decimal
        except (TypeError, ValueError):
            pass
    if result == "W":
        return d - 1.0 if d > 1 else 0.0
    if result == "L":
        return -1.0
    return 0.0  # push or pending


# ---------------------------------------------------------------------------
# Grading pipeline
# ---------------------------------------------------------------------------

def grade_date(date_str, sport_key_by_slug, force=False, rescore=False):
    """Grade one date — write graded_{date}.json.

    `force=True`  bypasses the "all settled → skip" early-exit so the
                  grader runs again.
    `rescore=True` ALSO ignores prior per-pick results so each pick is
                  re-evaluated against the current grader (e.g. after
                  a rule change). Implies force.
    """
    if rescore:
        force = True
    existing = read_graded(date_str)
    if existing and not force:
        all_settled = all(p.get("result") in ("W", "L", "P") for p in existing.get("picks", []))
        if all_settled:
            return existing

    raw = read_picks(date_str)
    if not raw:
        return existing

    # Only call sports that this date actually has picks in. ESPN is now
    # the primary scores source (free, unlimited, period-level data for
    # every sport); Odds API only runs when ESPN returns nothing AND an
    # Odds API key is configured. MLB keeps statsapi as its first source
    # because statsapi has richer inning data than ESPN does for baseball.
    # Anchor the ESPN query at the date we are grading so a backfill run
    # for picks that are 5+ days old reaches the right scoreboard day.
    try:
        ref_d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        ref_d = date.today()
    import sports as _sports
    sport_entry_by_slug = {s["slug"]: s for s in _sports.SPORTS}
    need_sports = {p["sport_slug"] for p in raw["picks"]}
    events_by_sport = {}
    for slug in need_sports:
        events = []
        sp = sport_entry_by_slug.get(slug) or {}
        if slug == "mlb":
            events.extend(fetch_scores_mlb(date_str))
        espn_path = sp.get("espn_scoreboard")
        if espn_path:
            events.extend(fetch_scores_espn(espn_path, ref_date=ref_d))
        if not events:
            key = sport_key_by_slug.get(slug)
            if key:
                events = fetch_scores(key)
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
                "profit_u": p.get("profit_u", profit_at_1u(p.get("result"), p.get("decimal"),
                                                            fair_prob=p.get("fair_prob"))),
                "graded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })
            continue

        events = events_by_sport.get(p["sport_slug"], [])
        ev = find_result_event(p, events)
        prior = None
        if existing:
            prior = next((g for g in existing.get("picks", []) if g.get("id") == p.get("id")), None)
        if prior and prior.get("result") in ("W", "L", "P") and not rescore:
            # Keep prior settled result (Odds API window drops after 3 days)
            graded_picks.append(prior)
            continue
        # Prior was pending — fall through and try to grade again. If we still
        # can't find the game, pending stays pending.
        res = grade_pick(p, ev)
        home_s, away_s = _extract_scores(ev) if ev else (None, None)
        # Phase 22: Kalshi settlement fallback for ML picks. When ESPN
        # can't find/match the game but Kalshi has it listed and
        # settled, use Kalshi's result. Only fires for ML markets on
        # Kalshi-covered sports (NFL/NCAAF/NHL/MLB) — Kalshi doesn't
        # list spreads, totals, or soccer/UFC/Liga MX.
        if res == "pending" and "ML" in (p.get("market") or ""):
            try:
                import kalshi_odds as _ko
                k_result = _ko.ml_result(
                    p.get("sport_slug"), p.get("home_team"), p.get("away_team"))
                if k_result in ("home_win", "away_win"):
                    pick_text = (p.get("pick") or "").lower()
                    home_lc = (p.get("home_team") or "").lower()
                    away_lc = (p.get("away_team") or "").lower()
                    picked_home = bool(home_lc and (pick_text == home_lc or home_lc in pick_text))
                    picked_away = bool(away_lc and (pick_text == away_lc or away_lc in pick_text))
                    if picked_home:
                        res = "W" if k_result == "home_win" else "L"
                    elif picked_away:
                        res = "W" if k_result == "away_win" else "L"
            except Exception:
                pass
        # Phase 25: Kalshi settlement fallback for DRAW NO BET picks.
        # DNB-on-A ≡ NO-on-B's ML. We look at Kalshi's opposite-team
        # ML market: if opponent settled YES (opponent won), our DNB
        # loses; if opponent settled NO (draw or we won), our DNB wins.
        # Matches plays_log's Phase 20 team-or-draw grading convention.
        _mkt_lc = (p.get("market") or "").lower()
        if res == "pending" and ("draw no bet" in _mkt_lc or p.get("market") == "DNB"):
            try:
                import kalshi_odds as _ko
                pick_lc = (p.get("pick") or "").lower()
                home_lc = (p.get("home_team") or "").lower()
                away_lc = (p.get("away_team") or "").lower()
                picked_home = bool(home_lc and home_lc in pick_lc)
                picked_away = bool(away_lc and away_lc in pick_lc)
                if picked_home or picked_away:
                    dnb_res = _ko.dnb_result(
                        p.get("sport_slug"),
                        p.get("home_team"), p.get("away_team"),
                        picked_home=picked_home,
                    )
                    if dnb_res in ("W", "L"):
                        res = dnb_res
            except Exception:
                pass
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
            "profit_u":   profit_at_1u(res, p.get("decimal"),
                                        fair_prob=p.get("fair_prob")),
            "home_score": home_s,
            "away_score": away_s,
            "clv_pp":     clv_pp,
            "clv_ev_pct": clv_pct,
            "graded_at":  datetime.utcnow().isoformat(timespec="seconds") + "Z" if res in ("W","L","P") else None,
        })

    graded = {"date": date_str, "picks": graded_picks}
    write_graded(date_str, graded)
    _invalidate_caches_on_change()
    return graded


def _invalidate_caches_on_change():
    """Shared post-write hook so grade_all / downstream consumers see
    fresh data immediately rather than up to 60s later."""
    try:
        invalidate_grade_all_cache()
    except Exception:
        pass


def grade_pending_backfill(days_back=14, grace_hours=4, force_rescan=False):
    """Walk logged dates in the last `days_back` days and re-grade any
    date that still has pending picks whose games finished (start_time +
    `grace_hours` ago). This catches picks that were pending when first
    graded — common on dates when the grader ran before the game ended,
    or when a sport's scoreboard query returned empty on the first pass.

    Returns {"dates_scanned": int, "resettled": int, "still_pending": int}.
    `force_rescan=True` ignores the "all settled" early-exit in grade_date.
    """
    import sports as _sports
    sport_key_by_slug = {s["slug"]: s.get("odds_api_key") for s in _sports.SPORTS}
    today_d = date.today()
    scanned = resettled = still_pending = 0
    now = datetime.utcnow()
    for ds in all_logged_dates(days_back + 7):
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d > today_d or (today_d - d).days > days_back:
            continue
        existing = read_graded(ds)
        if existing:
            pending = [p for p in existing.get("picks", [])
                       if p.get("result") not in ("W", "L", "P")]
            if not pending and not force_rescan:
                continue
            # Only re-grade when at least one pending pick's game is well
            # past its start_time — otherwise we're just hammering ESPN
            # for a game that isn't done yet.
            any_ended = False
            for p in pending:
                st = p.get("start_time")
                if not st:
                    any_ended = True
                    break
                try:
                    pst = datetime.fromisoformat(st.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
                hours_past = (now - pst.replace(tzinfo=None)).total_seconds() / 3600.0
                if hours_past >= grace_hours:
                    any_ended = True
                    break
            if not any_ended and not force_rescan:
                continue
        scanned += 1
        prev_pending = (len([p for p in (existing or {}).get("picks", [])
                             if p.get("result") not in ("W", "L", "P")])
                        if existing else 0)
        g = grade_date(ds, sport_key_by_slug, force=True)
        if g:
            now_pending = len([p for p in g.get("picks", [])
                               if p.get("result") not in ("W", "L", "P")])
            resettled += max(0, prev_pending - now_pending)
            still_pending += now_pending
    return {"dates_scanned": scanned, "resettled": resettled,
            "still_pending": still_pending}


_GRADE_ALL_CACHE = {}   # limit_dates -> (ts, result dict)
_GRADE_ALL_TTL_S = 60   # keep hot callers (/logged, TODAY, dispatcher,
                        # notify-changes) from re-reading every pick-
                        # file on every tick.


def grade_all(limit_dates=90):
    """Grade every logged date up to `limit_dates` back. Returns dict by date.
    Cached for _GRADE_ALL_TTL_S so repeat calls within the window reuse
    the computed result instead of re-reading every picks_YYYY-MM-DD.json
    file. Call invalidate_grade_all_cache() if you need fresh data (eg
    right after a save_daily_picks or grade_pending_backfill)."""
    import time as _t
    now = _t.time()
    hit = _GRADE_ALL_CACHE.get(limit_dates)
    if hit and (now - hit[0]) < _GRADE_ALL_TTL_S:
        return hit[1]
    import sports as _sports
    sport_key_by_slug = {s["slug"]: s.get("odds_api_key") for s in _sports.SPORTS}
    out = {}
    for ds in all_logged_dates(limit_dates):
        g = grade_date(ds, sport_key_by_slug)
        if g:
            out[ds] = g["picks"]
    _GRADE_ALL_CACHE[limit_dates] = (now, out)
    return out


def invalidate_grade_all_cache():
    """Call after save_daily_picks / grade_pending_backfill so the next
    grade_all() call sees fresh data instead of a 60s-stale snapshot."""
    _GRADE_ALL_CACHE.clear()


# ---------------------------------------------------------------------------
# Record aggregation
# ---------------------------------------------------------------------------

def _conf_tier_for(pk):
    """Mirror of mlb_ui._conf_tier — reproduced here to avoid an import cycle."""
    prob = pk.get("consensus_prob") or pk.get("fair_prob")
    if prob is None:
        return 1
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return 1
    if p >= 0.68: return 5
    if p >= 0.62: return 4
    if p >= 0.57: return 3
    if p >= 0.52: return 2
    return 1


def _backtest_signal_score(pk):
    """Reconstruct the current signal_score from fields that WERE in the
    log at save time. Book-agreement (+1 for DK/Kalshi divergence) is
    excluded because historical logs don't carry those fields — so the
    max achievable in a backtest is 4, same cap as live today while the
    cross-book data path is dark."""
    score = 0.0
    try:
        ev = float(pk.get("ev_pct") or 0.0)
    except (TypeError, ValueError):
        ev = 0.0
    if ev >= 6.0:
        score += 2
    elif ev >= 3.0:
        score += 1
    try:
        move = float(pk.get("line_move_pp") or 0.0)
    except (TypeError, ValueError):
        move = 0.0
    if move > 0.0:
        score += 1
    try:
        fp = float(pk.get("fair_prob") or 0.0)
    except (TypeError, ValueError):
        fp = 0.0
    if fp >= 0.60 and pk.get("source") != "alt":
        score += 1
    return int(min(5, max(0, score)))


def _passes_current_filter_live(pk):
    """Live recompute of the current-filter judgment from this pk's
    CURRENT fields: not model-only, EV cushion met for its market class,
    signal composite >= 3. Drifts intra-day as price/EV move."""
    if pk.get("model_only"):
        return False
    try:
        ev = float(pk.get("ev_pct") or 0.0)
    except (TypeError, ValueError):
        return False
    # EV floor: 5% for alt-line picks, 3% for everything else. Matches the
    # emission-site thresholds in mlb_ui._model_fill_picks and picks.MIN_EV_PCT.
    is_alt = (pk.get("source") == "alt")
    ev_floor = 5.0 if is_alt else 3.0
    if ev < ev_floor:
        return False
    # Signal composite — same math as live, minus book-agreement (dark).
    if _backtest_signal_score(pk) < 3:
        return False
    return True


GAME_START_GRACE_MIN = 60  # tolerance for "last live" vs kickoff.


def _was_live_at_game_start(pk):
    """True iff this pick was in the recommended set within
    GAME_START_GRACE_MIN of its game start — i.e. it stayed on the live
    board until kickoff rather than being EV'd out hours before. Uses
    `last_live_at` (updated every save while the pick is still in the
    live picks arg). Picks that have no last_live_at (historical files
    from before this field landed) default to True so legacy records
    are not retroactively zeroed out."""
    start_str = pk.get("start_time")
    last_live = pk.get("last_live_at")
    if not start_str or not last_live:
        return True
    try:
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        live_dt  = datetime.fromisoformat(last_live.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return True
    gap_min = (start_dt - live_dt).total_seconds() / 60.0
    # gap_min > GAME_START_GRACE_MIN means last_live is MORE than 60
    # minutes before kickoff — the pick was pulled early, does not count.
    # gap_min <= GAME_START_GRACE_MIN (incl. negative = after kickoff) means
    # the pick made it to the pre-game window.
    return gap_min <= GAME_START_GRACE_MIN


def _passes_current_filter(pk):
    """True iff this pick's CURRENT (post-Phase 4) filter judgment
    counts: Signal composite >= 3, EV cushion, not model-only, AND it
    stayed in the recommended set up to within GAME_START_GRACE_MIN of
    kickoff. The last check is the "only count picks that stayed on
    until the game started" rule — a pick EV'd out hours before the
    game is no longer our pick."""
    if not _passes_current_filter_live(pk):
        return False
    return _was_live_at_game_start(pk)


def _passes_current_filter_v2(pk):
    """CURRENT FILTER 2.0 — same as _passes_current_filter PLUS the
    decimal >= 1.50 constraint shipped 2026-10-19. 30-day backtest
    found that heavy-favorite picks (decimal < 1.50, shorter than
    ~-200 American) consistently underperform expectation on this
    model — removing them lifted ROI from +16.2% to +22.1% and units
    from +13.78u to +15.00u on 20% fewer bets.

    Alt-line picks and Correct Score are exempt from the decimal
    floor (alts are informational companions; CS prices are
    structurally wide and not comparable to ML favorites)."""
    if not _passes_current_filter(pk):
        return False
    try:
        dec = float(pk.get("decimal") or 0.0)
    except (TypeError, ValueError):
        return False
    if dec >= 1.50:
        return True
    # Below 1.50 — exempt alt picks and Correct Score
    if pk.get("alt_line_flag") or pk.get("source") == "alt":
        return True
    if (pk.get("market") or "") == "Correct Score":
        return True
    return False


def _is_whale_pick(pk):
    """WHALE PICK (bankroll-maximum tier) — the pick is strong enough
    to justify a quarter-to-half-Kelly stake (5-10% of bankroll).

    Rules (REVISED 2026-10-21 based on 30-day backtest sweep):
      - mc_prob (raw model) >= 0.68    high conviction on the model's view
      - decimal between 1.50 and 1.90  (~-200 to +100 American)
      - ev_pct >= 1.5                  positive edge even at short price
      - NOT do_not_bet                 filter still holds pre-game
      - NO risk_flags with weather=high or injury

    Previous rule required signal_score >= 4 AND fair_prob >= 0.72 AND
    decimal 1.30-1.65. That combination returned ZERO picks over 30
    days because Signal 4+ is nearly unreachable (needs both line_move
    AND book_agreement which rarely both fire). Backtest found the new
    rule hits 80% / +33% ROI on ~10 picks per 30d — rare but valuable.

    Variance guard: 68%+ hit at 1.50-1.90 decimal pays ~+0.5-1.0u per
    bet; a losing streak of 3 in a row (prob ~3.3%) costs ~3u. At
    quarter-Kelly (~5-8% of a 100u bankroll), that's a drawdown you
    recover from in 1-2 weeks."""
    # Raw model prob (mc_prob) is what carries the conviction here —
    # we don't use fair_prob because the blend can pull it down and
    # mask a high-conviction pick as borderline.
    try:
        mc = float(pk.get("mc_prob") or pk.get("fair_prob") or 0.0)
    except (TypeError, ValueError):
        mc = 0.0
    try:
        ev = float(pk.get("ev_pct") or 0.0)
    except (TypeError, ValueError):
        ev = 0.0
    dec = pk.get("decimal")
    if dec is None:
        am = pk.get("american")
        if am is None:
            return False
        try:
            am_f = float(am)
            dec = (am_f / 100.0 + 1.0) if am_f > 0 else (100.0 / abs(am_f) + 1.0)
        except (TypeError, ValueError):
            return False
    try:
        dec = float(dec)
    except (TypeError, ValueError):
        return False
    if pk.get("do_not_bet"):
        return False
    rf = pk.get("risk_flags") or {}
    if rf.get("weather") == "high" or rf.get("injury"):
        return False
    return (mc >= 0.68) and (1.50 <= dec <= 1.90) and (ev >= 1.5)


def record_in_window(graded_by_date, days_back, strong_only=False, today=None,
                     source=None, conf_tier=None, plus_ev_only=False,
                     min_ev=None, current_filter=False, whale_only=False,
                     current_filter_v2=False):
    """Aggregate W/L/P/pending + units + ROI for picks that match the filters.
      source="ev"       → only EV-verified consensus picks.
      source="model"    → only model-only picks that scored 4/5 or 5/5.
      source="book_dk"  → only picks imported from the user's DK history.
      conf_tier=5       → only picks whose consensus_prob maps to the 5/5 tier
                          (prob >= 68%). Any int 1-5 works.
      plus_ev_only=True → only picks with ev_pct > 0 (any positive edge,
                          not just the +2% threshold used for EV tagging).
      min_ev=9.0        → only picks with ev_pct >= 9%. Combines with
                          plus_ev_only to form the "+EV ≥ 9%" tracker.
      current_filter=True → only picks that would pass the CURRENT (Phase 4)
                          filter: not model-only, EV >= 3% (5% for alts),
                          signal composite >= 3. Shows 'if we had used
                          today's filter all along' retroactively.
      strong_only=True  → Signal composite >= 3 (same math as live). The
                          "picks worth placing today" tracker.
      whale_only=True   → fair_prob >= 65% AND decimal <= 1.60 AND ev >= 0.
                          Heavy-favorite banker picks (UI chip criterion).
    """
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
            if current_filter_v2:
                if not _passes_current_filter_v2(pk):
                    continue
            elif current_filter:
                if not _passes_current_filter(pk):
                    continue
            elif strong_only:
                # Signal composite >= 3 (same math as live, less book-
                # agreement which isn't logged pre-Phase 3). This is the
                # tracker for "picks worth placing today" — the Signal >= 3
                # chip on the main board.
                if _backtest_signal_score(pk) < 3:
                    continue
            elif whale_only:
                if not _is_whale_pick(pk):
                    continue
            if source is not None:
                pk_source = pk.get("source") or ("model" if pk.get("model_only") else "ev")
                if pk_source != source:
                    continue
            if conf_tier is not None and _conf_tier_for(pk) != conf_tier:
                continue
            if plus_ev_only:
                ev = pk.get("ev_pct")
                if ev is None or ev <= 0:
                    continue
            if min_ev is not None:
                ev = pk.get("ev_pct")
                if ev is None or float(ev) < min_ev:
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
    """Return dict of 7D / 30D / 90D records sliced by the user-chosen
    tracker set:
      all_*       → every settled pick (overall record)
      strong_*    → Signal composite ≥ 3 (picks worth placing today)
      whale_*     → fair_prob ≥ 65% AND decimal ≤ 1.60 AND ev ≥ 0 (banker picks)
      plusev9_*   → only picks with ev_pct ≥ 9% (big-edge tracker)
      dk_*        → only picks imported from the user's DraftKings history
      model_* / ev_* kept for back-compat with embedded dashboards; not
      shown by default on /logged.
    """
    out = {}
    for label, days in (("7d", 7), ("30d", 30), ("90d", 90)):
        out[f"all_{label}"]      = record_in_window(graded_by_date, days, False, today)
        out[f"strong_{label}"]   = record_in_window(graded_by_date, days, True,  today)
        out[f"whale_{label}"]    = record_in_window(graded_by_date, days, False, today, whale_only=True)
        out[f"plusev9_{label}"]  = record_in_window(graded_by_date, days, False, today, min_ev=9.0)
        out[f"dk_{label}"]       = record_in_window(graded_by_date, days, False, today, source="book_dk")
        # Phase 11b: 1H Goals picks (saved from /first-half, source "1h_goals").
        out[f"fh_{label}"]       = record_in_window(graded_by_date, days, False, today, source="1h_goals")
        # Phase 11c: AI Analysis picks (saved from /ai-analysis non-Pass, source "ai_analysis").
        out[f"ai_{label}"]       = record_in_window(graded_by_date, days, False, today, source="ai_analysis")
        # Phase 6b: "if we had used today's filter all along" retro-backtest.
        # Picks that would have survived the current Signal >= 3 + EV >= 3%
        # (5% alt) + no-model-only filter stack, re-aggregated across the
        # window using the already-graded W/L on each log entry.
        out[f"curfilter_{label}"] = record_in_window(graded_by_date, days,
                                                     False, today,
                                                     current_filter=True)
        # Current Filter 2.0 — adds decimal >= 1.50 to the current
        # filter rules. 30-day backtest (2026-10-19) found this lifts
        # ROI from +16.2% → +22.1% and units from +13.78u → +15.00u
        # on 20% fewer bets. Shipped as the live default on the TODAY
        # board; this record tile shows how the retro-replay would
        # have looked.
        out[f"curfilter2_{label}"] = record_in_window(graded_by_date, days,
                                                      False, today,
                                                      current_filter_v2=True)
        # back-compat (not rendered)
        out[f"ev_{label}"]       = record_in_window(graded_by_date, days, False, today, source="ev")
        out[f"model_{label}"]    = record_in_window(graded_by_date, days, False, today, source="model")
    return out


def breakdown_by_key_current_filter(graded_by_date, key_fn, days_back=90, today=None):
    """Same breakdown shape as breakdown_by_key, but limits picks to
    those that WOULD pass the current (Signal >= 3 + EV >= 3%/5%)
    filter. Tells the user which (sport / market / prob band) buckets
    the live signal filter has been accurate on — a direct read on
    where model weights should be nudged up or down."""
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
            if not _passes_current_filter(pk):
                continue
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
