#!/usr/bin/env python3
"""Betting Tools — cross-sport Monte Carlo + AI analysis dashboard.

Data sources (free, no API keys):
  * MLB statsapi   — schedule (hydrated), standings, team stats, pitcher stats
  * mlb_model      — Elo + SP prediction model, walk-forward backtest

Install once:
    pip install flask

Run:
    python mlb_ui.py

Then open http://127.0.0.1:5000
"""
import csv
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from flask import Flask, Response, render_template_string, request

import mlb_model
import mlb_odds
import generic_odds
import sports

CENTRAL = ZoneInfo("America/Chicago")

STATSAPI = "https://statsapi.mlb.com/api/v1"
SCHEDULE_URL = (
    STATSAPI + "/schedule?sportId=1&date={date}"
    "&hydrate=probablePitcher,linescore,venue,weather,team,broadcasts(all)"
)
STANDINGS_URL = STATSAPI + "/standings?leagueId=103,104&season={season}"
TEAM_STATS_URL = (
    STATSAPI + "/teams/{tid}/stats?stats=season&group=hitting,pitching"
    "&season={season}&sportIds=1"
)
PITCHER_STATS_URL = (
    STATSAPI + "/people/{pid}/stats?stats=season&group=pitching"
)

_cache: dict = {}
_CACHE_TTL_S = 30 * 60

app = Flask(__name__)


@app.template_filter("ct")
def _ct_filter(iso_str, fmt="%I:%M %p CT"):
    """Jinja filter: convert a UTC ISO timestamp to a Central-time display.

    Usage in templates: {{ g.start_time|ct }}  -> "7:05 PM CT"
    Returns the input unchanged if it isn't a parseable UTC timestamp.
    """
    if not iso_str:
        return ""
    try:
        from datetime import datetime as _dt
        iso = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
        dt = _dt.fromisoformat(iso)
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(CENTRAL).strftime(fmt).lstrip("0").replace(" 0", " ")
    except Exception:
        return iso_str


# ============================================================================
# fetch primitives
# ============================================================================

def fetch_json(url):
    req = Request(url, headers={"User-Agent": "first-pitch-ui/3.0"})
    with urlopen(req, timeout=15) as resp:
        return json.load(resp)


def _cached(key, ttl, fetch_fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[1] < ttl:
        return hit[0]
    try:
        v = fetch_fn()
    except (URLError, ValueError, TimeoutError, ConnectionError):
        v = None
    _cache[key] = (v, now)
    return v


# ============================================================================
# data sources
# ============================================================================

def get_standings_by_team(season):
    key = ("standings", season)
    def _f():
        d = fetch_json(STANDINGS_URL.format(season=season))
        by_team = {}
        for rec in d.get("records", []):
            for tr in rec.get("teamRecords", []):
                tid = tr.get("team", {}).get("id")
                if not tid:
                    continue
                streak = tr.get("streak") or {}
                splits = {s.get("type"): s for s in (tr.get("records") or {}).get("splitRecords", [])}
                home = splits.get("home", {})
                away = splits.get("away", {})
                by_team[tid] = {
                    "wins": tr.get("wins"),
                    "losses": tr.get("losses"),
                    "streak_code": streak.get("streakCode"),
                    "streak_type": streak.get("streakType"),
                    "run_diff": tr.get("runDifferential"),
                    "runs_scored": tr.get("runsScored"),
                    "runs_allowed": tr.get("runsAllowed"),
                    "division_rank": tr.get("divisionRank"),
                    "home_wins": home.get("wins"),
                    "home_losses": home.get("losses"),
                    "away_wins": away.get("wins"),
                    "away_losses": away.get("losses"),
                }
        return by_team
    return _cached(key, _CACHE_TTL_S, _f) or {}


def get_team_stats(team_id, season):
    key = ("team_stats", team_id, season)
    def _f():
        d = fetch_json(TEAM_STATS_URL.format(tid=team_id, season=season))
        out = {"hitting": {}, "pitching": {}}
        for s in d.get("stats", []):
            group = (s.get("group") or {}).get("displayName")
            splits = s.get("splits") or []
            if group and splits:
                out[group] = splits[0].get("stat", {})
        return out
    return _cached(key, _CACHE_TTL_S, _f) or {"hitting": {}, "pitching": {}}


def get_pitcher_full(pid, season):
    if not pid:
        return None
    key = ("pitcher", pid, season)
    def _f():
        d = fetch_json(PITCHER_STATS_URL.format(pid=pid))
        stats = d.get("stats") or []
        splits = stats[0].get("splits") if stats else []
        if not splits:
            return {}
        return splits[0].get("stat", {}) or {}
    return _cached(key, _CACHE_TTL_S, _f)


# ============================================================================
# helpers
# ============================================================================

def era_bucket(era_str):
    if not era_str:
        return "none"
    try:
        era = float(era_str)
    except (TypeError, ValueError):
        return "none"
    if era < 3.75:
        return "good"
    if era < 4.50:
        return "warn"
    return "poor"


def streak_glyph(streak_type):
    if streak_type == "wins":
        return "▲"  # black up-pointing triangle
    if streak_type == "losses":
        return "▼"
    return ""


def format_time_et(iso_utc):
    if not iso_utc:
        return None
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(CENTRAL)


def parse_pitcher_stat(stat, key):
    if not stat:
        return None
    v = stat.get(key)
    if v in (None, "", "-.--"):
        return None
    return v


def _fmt_record(w, l):
    if w is None or l is None:
        return "-"
    return f"{w}-{l}"


def _parse_ip(ip_str):
    s = str(ip_str)
    if "." in s:
        whole, frac = s.split(".")
        try:
            return int(whole) + int(frac) / 3.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def sp_stats_for_model(stat):
    """Convert statsapi pitcher season stat dict into model input format."""
    if not stat:
        return None
    ip = _parse_ip(stat.get("inningsPitched", 0))
    if ip <= 0:
        return None
    try:
        era = float(stat.get("era", 0))
        whip = float(stat.get("whip", 0))
    except (TypeError, ValueError):
        return None
    so = int(stat.get("strikeOuts", 0) or 0)
    bb = int(stat.get("baseOnBalls", 0) or 0)
    return {
        "ip": ip, "era": era, "whip": whip,
        "k9": so * 9.0 / ip, "bb9": bb * 9.0 / ip,
    }


# ============================================================================
# model state (loaded from mlb_model cache)
# ============================================================================

def get_model_state():
    """Load fitted model state (final Elo only) for live predictions.

    Was previously loading the full 6MB multi-season cache (predictions +
    per-season metrics + per-game histories) just to pluck out `final_elo`.
    Switched to the slim side-file loader to keep the free-tier worker
    well under its 512 MB memory ceiling.
    """
    key = ("model_state_slim",)
    def _f():
        try:
            return mlb_model.get_final_state()
        except Exception as e:
            print(f"[warn] mlb model unavailable: {e}", flush=True)
            return None
    return _cached(key, 60 * 60, _f)


# ============================================================================
# game assembly
# ============================================================================

def get_games(date_str):
    schedule = fetch_json(SCHEDULE_URL.format(date=date_str))
    season = date_str[:4]

    team_ids = set()
    pitcher_ids = set()

    raw_games = []
    for date_block in schedule.get("dates", []):
        for game in date_block.get("games", []):
            raw_games.append(game)
            teams = game.get("teams", {}) or {}
            for side in ("home", "away"):
                t = teams.get(side, {}) or {}
                if t.get("team", {}).get("id"):
                    team_ids.add(t["team"]["id"])
                pp = t.get("probablePitcher") or {}
                if pp.get("id"):
                    pitcher_ids.add(pp["id"])

    # fan-out fetches
    standings = {}
    team_stats_by_id = {}
    pitcher_by_id = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {}
        futs[ex.submit(get_standings_by_team, season)] = ("standings", None)
        for tid in team_ids:
            futs[ex.submit(get_team_stats, tid, season)] = ("team_stats", tid)
        for pid in pitcher_ids:
            futs[ex.submit(get_pitcher_full, pid, season)] = ("pitcher", pid)
        for fut in as_completed(futs):
            kind, key = futs[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if kind == "standings":
                standings = res or {}
            elif kind == "team_stats":
                team_stats_by_id[key] = res or {"hitting": {}, "pitching": {}}
            elif kind == "pitcher":
                pitcher_by_id[key] = res

    # model state — for predictions
    state = get_model_state()
    final_elo = (state or {}).get("final_elo", {}) if state else {}

    games = []
    for game in raw_games:
        teams = game.get("teams", {}) or {}
        home = teams.get("home", {}) or {}
        away = teams.get("away", {}) or {}
        home_team_obj = home.get("team", {})
        away_team_obj = away.get("team", {})
        home_pitcher = home.get("probablePitcher") or {}
        away_pitcher = away.get("probablePitcher") or {}

        dt = format_time_et(game.get("gameDate", ""))
        first_pitch = dt.strftime("%I:%M %p CT").lstrip("0") if dt else "TBD"

        weather = game.get("weather") or {}
        wx_temp = weather.get("temp")
        wx_cond = weather.get("condition")
        wx_wind = weather.get("wind")

        broadcasts = game.get("broadcasts") or []
        tv_names = []
        for b in broadcasts:
            if b.get("type") == "TV":
                name = b.get("name") or b.get("callSign")
                if name and name not in tv_names:
                    tv_names.append(name)
        tv_str = " · ".join(tv_names[:2])

        def team_bundle(side_obj):
            t = side_obj.get("team", {}) or {}
            tid = t.get("id")
            rec = standings.get(tid, {}) or {}
            stats = team_stats_by_id.get(tid) or {"hitting": {}, "pitching": {}}
            hitting = stats.get("hitting") or {}
            pitching = stats.get("pitching") or {}
            # numeric derivatives for the totals/F5 model
            runs = hitting.get("runs")
            gp = hitting.get("gamesPlayed")
            rpg = (runs / gp) if (runs and gp) else None
            try:
                team_era_num = float(pitching.get("era")) if pitching.get("era") else None
            except (TypeError, ValueError):
                team_era_num = None
            return {
                "rpg": rpg,
                "team_era_num": team_era_num,
                "id": tid,
                "team": t.get("teamName") or t.get("name", ""),
                "team_full": t.get("name", ""),
                "team_abbr": t.get("abbreviation", ""),
                "wins": rec.get("wins"),
                "losses": rec.get("losses"),
                "run_diff": rec.get("run_diff"),
                "streak_code": rec.get("streak_code") or "",
                "streak_glyph": streak_glyph(rec.get("streak_type")),
                "streak_type": rec.get("streak_type") or "",
                "division_rank": rec.get("division_rank"),
                "home_record": _fmt_record(rec.get("home_wins"), rec.get("home_losses")),
                "away_record": _fmt_record(rec.get("away_wins"), rec.get("away_losses")),
                "avg": hitting.get("avg", ""),
                "obp": hitting.get("obp", ""),
                "slg": hitting.get("slg", ""),
                "ops": hitting.get("ops", ""),
                "hr": hitting.get("homeRuns"),
                "runs": hitting.get("runs"),
                # Keep the raw hitting dict so the plate-appearance simulator
                # can derive K% / BB% / 1B / 2B / 3B / HR per-PA rates.
                "hitting_raw": hitting,
                "team_era": pitching.get("era", ""),
                "team_whip": pitching.get("whip", ""),
                "team_k9": pitching.get("strikeoutsPer9Inn", ""),
            }

        def pitcher_bundle(pobj):
            pid = pobj.get("id")
            stat = pitcher_by_id.get(pid) if pid else None
            era = parse_pitcher_stat(stat, "era")
            return {
                "id": pid,
                "raw": stat,
                "name": pobj.get("fullName") or "TBD",
                "wl": (
                    f"{stat.get('wins')}-{stat.get('losses')}"
                    if stat and stat.get("wins") is not None else "-"
                ),
                "era": era or "-",
                "whip": parse_pitcher_stat(stat, "whip") or "-",
                "k9": parse_pitcher_stat(stat, "strikeoutsPer9Inn") or "-",
                "bb9": parse_pitcher_stat(stat, "walksPer9Inn") or "-",
                "ip": parse_pitcher_stat(stat, "inningsPitched") or "-",
                "bucket": era_bucket(era),
                "hand": (pobj.get("pitchHand") or {}).get("code", ""),
            }

        away_bundle = team_bundle(away)
        home_bundle = team_bundle(home)
        away_pb = pitcher_bundle(away_pitcher)
        home_pb = pitcher_bundle(home_pitcher)

        # ---- MODEL PREDICTION ----
        p_home = None
        if state and away_bundle["id"] and home_bundle["id"]:
            h_elo = float(final_elo.get(str(home_bundle["id"]), mlb_model.INITIAL_ELO))
            a_elo = float(final_elo.get(str(away_bundle["id"]), mlb_model.INITIAL_ELO))
            h_sp_model = sp_stats_for_model(home_pb.get("raw"))
            a_sp_model = sp_stats_for_model(away_pb.get("raw"))
            try:
                p_home = mlb_model.predict_win_prob(h_elo, a_elo, h_sp_model, a_sp_model)
            except Exception:
                p_home = None

        # scrub raw so template render stays lean
        away_pb.pop("raw", None)
        home_pb.pop("raw", None)

        venue = game.get("venue") or {}
        status = (game.get("status") or {}).get("detailedState", "")

        p_home_pct = round(p_home * 100) if p_home is not None else None
        p_away_pct = (100 - p_home_pct) if p_home_pct is not None else None

        # ---- ODDS (Pinnacle fair + bettable price; Odds API if available) ----
        # Build game context for the totals/RL/F5 scoring model
        h_sp_m = sp_stats_for_model(pitcher_by_id.get(home_pitcher.get("id")) if home_pitcher.get("id") else None)
        a_sp_m = sp_stats_for_model(pitcher_by_id.get(away_pitcher.get("id")) if away_pitcher.get("id") else None)
        game_ctx = {
            "home_rpg": home_bundle.get("rpg"),
            "away_rpg": away_bundle.get("rpg"),
            "home_team_era": home_bundle.get("team_era_num"),
            "away_team_era": away_bundle.get("team_era_num"),
            "home_sp_era": h_sp_m.get("era") if h_sp_m else None,
            "home_sp_ip": h_sp_m.get("ip") if h_sp_m else None,
            "away_sp_era": a_sp_m.get("era") if a_sp_m else None,
            "away_sp_ip": a_sp_m.get("ip") if a_sp_m else None,
        }
        odds = mlb_odds.build_game_odds(
            away_bundle["team_full"],
            home_bundle["team_full"],
            model_prob_home=p_home,
            game_ctx=game_ctx,
        )

        games.append({
            "first_pitch": first_pitch,
            "sort_key": dt or datetime.max.replace(tzinfo=CENTRAL),
            "status": status,
            "series_num": game.get("seriesGameNumber"),
            "games_in_series": game.get("gamesInSeries"),
            "away": away_bundle,
            "home": home_bundle,
            "away_pitcher": away_pb,
            "home_pitcher": home_pb,
            "venue": venue.get("name", ""),
            "wx_temp": wx_temp or "",
            "wx_condition": wx_cond or "",
            "wx_wind": wx_wind or "",
            "tv": tv_str,
            "p_home": p_home,
            "p_home_pct": p_home_pct,
            "p_away_pct": p_away_pct,
            "favors_home": (p_home is not None and p_home >= 0.5),
            "game_pk": game.get("gamePk"),
            "odds": odds,
            "game_ctx": game_ctx,
        })
    games.sort(key=lambda g: g["sort_key"])
    return games


# ============================================================================
# HTML — shared styles + layout
# ============================================================================

SHARED_STYLE = r"""
:root {
  /* Dark-only palette — late-night scoreboard look */
  color-scheme: dark;
  --bg: #0E0D11; --surface: #15141A; --card: #1C1B23;
  --ink: #ECEAF2; --muted: #8E8A99; --muted-2: #605D6E;
  --rule: #26242D; --rule-strong: #3A3744; --chip-bg: #15141A;
  --accent: #FF4D6A;
  --accent-glow: rgba(255, 77, 106, 0.4);
  --good: #22F0A0;                /* electric mint */
  --good-glow: rgba(34, 240, 160, 0.35);
  --warn: #FFC94A;
  --warn-glow: rgba(255, 201, 74, 0.3);
  --poor: #FF4D6A;
  --focus: #22F0A0;
  --card-shadow: 0 1px 0 rgba(0,0,0,0.3);
  --header-bg: rgba(14,13,17,0.88);
  --ev-strong-glow: 0 0 14px var(--good-glow);
  --card-hover-shadow:
    0 6px 24px rgba(0,0,0,0.45),
    0 0 0 1px var(--accent),
    0 0 20px rgba(255, 77, 106, 0.08);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--ink);
  font-family: "Public Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14.5px; line-height: 1.5;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 24px; }

header {
  position: sticky; top: 0; z-index: 10;
  background: var(--header-bg);
  backdrop-filter: saturate(1.2) blur(10px);
  -webkit-backdrop-filter: saturate(1.2) blur(10px);
  border-bottom: 1px solid var(--rule);
}
.header-row {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; height: 64px; flex-wrap: wrap;
}
.brand { display: flex; align-items: baseline; gap: 10px; }
.brand-mark {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); display: inline-block;
  transform: translateY(-2px);
}
.brand-name {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: 22px; letter-spacing: -0.01em;
  color: var(--ink);
  font-variation-settings: "opsz" 120;
}
.nav-tabs {
  display: flex; gap: 4px; align-items: center;
  margin-left: 20px;
}
.nav-tab {
  padding: 6px 12px; border-radius: 6px;
  color: var(--muted); text-decoration: none;
  font-size: 13px; font-weight: 500;
  border: 1px solid transparent;
  transition: color 120ms ease, background 120ms ease, border-color 120ms ease;
}
.nav-tab:hover { color: var(--ink); background: var(--surface); }
.nav-tab.active { color: var(--ink); border-color: var(--rule-strong); background: var(--card); }

/* ----- sport switcher strip ----- */
.sport-strip {
  border-bottom: 1px solid var(--rule);
  background: color-mix(in oklab, var(--surface) 60%, var(--bg));
}
.sport-strip-inner {
  display: flex; align-items: center; gap: 4px;
  padding: 10px 0;
  overflow-x: auto;
  scrollbar-width: thin;
}
.sport-pill {
  padding: 6px 14px; border-radius: 999px;
  color: var(--muted); text-decoration: none;
  font-size: 12px; font-weight: 500;
  border: 1px solid transparent;
  white-space: nowrap;
  transition: color 120ms ease, background 120ms ease, border-color 120ms ease;
}
.sport-pill:hover { color: var(--ink); background: var(--card); }
.sport-pill.active {
  color: var(--ink); background: var(--card);
  border-color: var(--ink);
  font-weight: 600;
}
.sport-pill .lbl-sub {
  margin-left: 6px; font-weight: 400; color: var(--muted-2);
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.08em; text-transform: uppercase;
}

/* Primary top bar: brand | centered tabs | theme toggle */
.sport-strip-inner.primary-nav {
  display: grid; grid-template-columns: auto 1fr auto;
  align-items: center; gap: 16px; padding: 12px 0;
}
.brand {
  font-family: "JetBrains Mono", monospace;
  font-size: 13px; font-weight: 600; letter-spacing: 0.14em;
  color: var(--ink); text-decoration: none;
  white-space: nowrap;
  padding-right: 8px;
  border-right: 1px solid var(--rule);
}
.brand:hover { color: var(--accent, var(--ink)); }
.primary-tabs {
  display: flex; justify-content: center; align-items: center;
  gap: 10px; flex-wrap: wrap;
}
.primary-nav .sport-pill {
  padding: 7px 16px; font-size: 13px;
  font-family: "Public Sans", system-ui, sans-serif;
  font-weight: 500; letter-spacing: 0.02em;
  border: 1px solid var(--rule-strong);
}
.primary-nav .sport-pill.active {
  background: var(--card);
  border-color: var(--accent, var(--ink));
  color: var(--ink); font-weight: 600;
  box-shadow: 0 0 0 1px var(--accent, transparent) inset;
}
@media (max-width: 640px) {
  .sport-strip-inner.primary-nav {
    grid-template-columns: 1fr;
    row-gap: 10px;
  }
  .brand { grid-column: 1; grid-row: 1; border-right: none; padding-right: 0; }
  .primary-tabs { grid-column: 1; grid-row: 2; justify-content: flex-start; overflow-x: auto; }
}

/* Secondary sport chip bar (inside MC / AI pages) */
.sport-chip-bar {
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 14px 0 18px; margin-bottom: 20px;
  border-bottom: 1px solid var(--rule);
}
.sport-chip {
  padding: 7px 14px; border-radius: 999px;
  color: var(--muted); text-decoration: none;
  font-size: 12.5px; font-weight: 500;
  border: 1px solid var(--rule-strong);
  background: var(--surface);
  white-space: nowrap;
}
.sport-chip:hover { color: var(--ink); background: var(--card); }
.sport-chip.active {
  color: var(--ink); background: var(--card);
  border-color: var(--accent, var(--ink));
  box-shadow: 0 0 0 1px var(--accent, transparent) inset;
}

.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.btn, input[type=date] {
  height: 36px;
  border: 1px solid var(--rule-strong);
  background: var(--surface); color: var(--ink);
  border-radius: 8px;
  font-family: inherit; font-size: 13px;
  transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
}
.btn {
  padding: 0 12px; font-weight: 500; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  text-decoration: none;
}
.btn.icon { width: 36px; padding: 0; font-size: 16px; line-height: 1; }
.btn:hover { background: var(--card); border-color: var(--ink); }
.btn:focus-visible, input[type=date]:focus-visible {
  outline: 2px solid var(--focus); outline-offset: 2px;
}
.btn.csv {
  color: var(--accent);
  border-color: color-mix(in oklab, var(--accent) 45%, var(--rule-strong));
}
.btn.csv:hover {
  background: color-mix(in oklab, var(--accent) 8%, var(--surface));
  border-color: var(--accent);
}
input[type=date] {
  padding: 0 10px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}
main { padding: 40px 0 60px; }
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
.reveal { animation: rise 240ms ease both; }
@keyframes rise { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

footer {
  margin: 60px auto 24px;
  padding: 20px 24px 0;
  border-top: 1px solid var(--rule);
  color: var(--muted); font-size: 12px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
}

/* ===== neon / gaming accents ===== */
.brand-mark {
  box-shadow: 0 0 10px var(--accent-glow),
              0 0 20px var(--accent-glow);
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 8px var(--accent-glow), 0 0 16px var(--accent-glow);
    transform: translateY(-2px) scale(1);
  }
  50% {
    box-shadow: 0 0 14px var(--accent-glow), 0 0 28px var(--accent-glow);
    transform: translateY(-2px) scale(1.15);
  }
}
.brand-name {
  text-shadow: 0 0 18px rgba(255, 77, 106, 0.14);
}
.nav-tab.active {
  box-shadow: 0 0 0 1px var(--accent-glow);
}
.sport-pill.active {
  box-shadow: inset 0 0 0 1px var(--accent),
              0 0 12px var(--accent-glow);
}
.ev-cell.strong,
.bets td.num.ev-cell.strong,
.edges .num.ev-strong {
  text-shadow: var(--ev-strong-glow);
}
.streak.up { text-shadow: 0 0 6px var(--good-glow); }
.streak.down { text-shadow: 0 0 6px var(--accent-glow); }

/* ===== mobile layout ===== */
@media (max-width: 640px) {
  body { font-size: 15px; }
  .wrap { padding: 0 14px; }
  header .wrap, main.wrap { padding-left: 14px; padding-right: 14px; }
  .header-row {
    height: auto; min-height: 56px;
    padding: 10px 0; gap: 10px;
  }
  .brand-name { font-size: 19px; }
  .nav-tabs {
    margin-left: 0; order: 2;
    flex-wrap: nowrap; overflow-x: auto;
    width: 100%; -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }
  .nav-tabs::-webkit-scrollbar { display: none; }
  .nav-tab { padding: 8px 12px; font-size: 13px; flex-shrink: 0; }
  .controls {
    width: 100%; order: 3;
    justify-content: flex-start;
  }
  .btn, input[type=date] {
    height: 40px; min-width: 40px; font-size: 14px;
  }
  .btn.icon { width: 40px; }
  .sport-pill {
    padding: 7px 12px; font-size: 13px;
  }
  main { padding: 24px 0 48px; }
  .hero, .hero-block {
    margin-bottom: 20px; padding-bottom: 16px;
  }
  .hero h1, .hero-block h1 {
    font-size: 28px;
  }
}

/* Tables on narrow screens: horizontal scroll inside a wrapper */
.table-scroll { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
@media (max-width: 760px) {
  .bets, .edges { overflow-x: auto; }
  .bets table, .edges table,
  .limits-table, .elo-table, .calib-table, .pred-table {
    min-width: 640px;
  }
  .elo-cols, .calib-wrap, .grid-metrics {
    grid-template-columns: 1fr !important;
  }
  .metric .value { font-size: 20px; }
  .model-def { grid-template-columns: 1fr 1fr; }
  .section { padding: 16px 18px; }
  .card { padding: 16px 16px 14px; }
  .team-name { font-size: 19px; }
}

/* ===== card hover reinforced for dark theme ===== */
.card:hover {
  box-shadow: var(--card-hover-shadow);
}

/* Positive EV rows a touch more visible under neon palette */
.bets tr.pos.strong td.num.ev-cell { color: var(--good); }
"""

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;1,9..144,400'
    '&family=Public+Sans:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500&display=swap">'
)
# Injected into every template's <head> via {{ fonts_link|safe }}
FONTS_LINK = _FONTS  # keep historical name; THEME_SCRIPT appended below


_PRIMARY_SECTIONS = [
    ("picks",      "Picks",        "/"),
    ("montecarlo", "Monte Carlo",  "/montecarlo"),
    ("analysis",   "AI Analysis",  "/ai-analysis"),
    ("logged",     "Logged Plays", "/logged"),
    ("backtest",   "Backtest",     "/backtest"),
]


def render_sport_strip(active_slug):
    """Top bar: BETTING TOOLS brand (left) | centered primary-tab nav. Dark-only."""
    active_primary = active_slug if active_slug in {s[0] for s in _PRIMARY_SECTIONS} else None
    parts = ['<div class="sport-strip"><div class="wrap sport-strip-inner primary-nav">']
    parts.append('<a class="brand" href="/">BETTING TOOLS</a>')
    parts.append('<nav class="primary-tabs">')
    for key, label, href in _PRIMARY_SECTIONS:
        cls = "sport-pill active" if key == active_primary else "sport-pill"
        parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    parts.append('</nav>')
    parts.append('</div></div>')
    return "".join(parts)


def render_sport_chip_bar(selected_slug, base_url, date_str=None):
    """Horizontal sport chip selector for Monte Carlo / AI Analysis pages.

    Renders every sport as a chip link to the same base_url with ?sport=slug.
    Preserves the date query param so the day you picked carries across sports.
    """
    date_q = f"&date={date_str}" if date_str else ""
    parts = ['<div class="sport-chip-bar">']
    for sp in sports.SPORTS:
        cls = "sport-chip active" if sp["slug"] == selected_slug else "sport-chip"
        parts.append(
            f'<a class="{cls}" href="{base_url}?sport={sp["slug"]}{date_q}">{sp["name"]}</a>'
        )
    parts.append('</div>')
    return "".join(parts)


def render_date_toggle(base_url, sport_slug, date_str, prev_date, next_date, is_today, today_str):
    """Prev / Today / Next date controls for MC + AI Analysis pages."""
    sport_q = f"&sport={sport_slug}" if sport_slug else ""
    today_btn = (f'<a class="date-btn" href="{base_url}?date={today_str}{sport_q}">Today</a>'
                 if not is_today else '')
    return (
        '<div class="date-toggle">'
        f'<a class="date-btn icon" href="{base_url}?date={prev_date}{sport_q}" aria-label="Previous day">&lsaquo;</a>'
        f'<span class="date-label">{date_str}</span>'
        f'<a class="date-btn icon" href="{base_url}?date={next_date}{sport_q}" aria-label="Next day">&rsaquo;</a>'
        f'{today_btn}'
        '</div>'
    )


# Which tabs each sport exposes. Order matters for display.
_SPORT_TABS = {
    "mlb": [
        ("schedule",   "Schedule",    "/mlb/schedule"),
        ("montecarlo", "Monte Carlo", "/montecarlo"),
        ("analyst",    "AI Analyst",  "/analyst"),
        ("backtest",   "Backtest",    "/backtest"),
        ("market",     "Market",      "/market"),
    ],
    "nfl": [
        ("schedule",  "Schedule",   "/sport/nfl"),
        ("backtest",  "Backtest",   "/sport/nfl/backtest"),
    ],
}
# Soccer leagues share the same tab set — Schedule / Monte Carlo / Backtest
_SOCCER_SLUGS_WITH_MODEL = {"epl", "laliga", "ligamx"}
_SOCCER_SLUGS_ALL = {"epl", "laliga", "ligamx", "ucl", "europa", "international"}
for _s in _SOCCER_SLUGS_WITH_MODEL:
    _SPORT_TABS[_s] = [
        ("schedule",   "Schedule",    f"/sport/{_s}"),
        ("montecarlo", "Monte Carlo", f"/sport/{_s}/montecarlo"),
        ("backtest",   "Backtest",    f"/sport/{_s}/backtest"),
    ]
# Soccer leagues without historical fit (UCL, Europa, Int'l) still get MC —
# Dixon-Coles falls back to league-average goal priors when teams aren't in
# the historical CSV.
for _s in (_SOCCER_SLUGS_ALL - _SOCCER_SLUGS_WITH_MODEL):
    _SPORT_TABS[_s] = [
        ("schedule",   "Schedule",    f"/sport/{_s}"),
        ("montecarlo", "Monte Carlo", f"/sport/{_s}/montecarlo"),
    ]


def _default_sport_tabs(slug):
    """Fallback tab set for sports with no bespoke list (UFC)."""
    return [
        ("schedule",  "Schedule",   f"/sport/{slug}"),
    ]


def render_sport_nav(slug, active):
    """Return nav-tabs HTML scoped to this sport, with `active` highlighted."""
    tabs = _SPORT_TABS.get(slug) or _default_sport_tabs(slug)
    parts = ['<nav class="nav-tabs">']
    for key, label, href in tabs:
        cls = "nav-tab active" if key == active else "nav-tab"
        parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    parts.append('</nav>')
    return "".join(parts)


# Dark-only: no theme toggle. Kept as empty strings so templates that still
# reference THEME_SCRIPT / FONTS_LINK don't need to be touched one by one.
THEME_SCRIPT = ""
FONTS_LINK = _FONTS


# ============================================================================
# main page template
# ============================================================================

INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools — {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 24px; margin: 0 0 32px;
  padding-bottom: 20px; border-bottom: 1px solid var(--rule);
}
.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 52px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0; text-wrap: balance;
  font-variation-settings: "opsz" 144;
}
.count {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
  white-space: nowrap; font-variant-numeric: tabular-nums;
}
.count strong { color: var(--ink); font-weight: 500; }

.grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 760px)  { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 1180px) { .grid { grid-template-columns: repeat(3, 1fr); } }

.card {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 22px 18px;
  box-shadow: var(--card-shadow);
  display: flex; flex-direction: column; gap: 16px;
  transition: border-color 160ms ease, transform 160ms ease;
}
.card:hover { border-color: var(--rule-strong); transform: translateY(-1px); }

.card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.time {
  font-family: "JetBrains Mono", monospace;
  font-weight: 500; font-size: 13px; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.meta-chip {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); padding: 3px 8px;
  border: 1px solid var(--rule); border-radius: 999px; background: var(--chip-bg);
}
.meta-chip.status { color: var(--accent); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); }

.teams { display: flex; flex-direction: column; gap: 12px; }
.team-row { display: grid; grid-template-columns: 1fr auto; align-items: baseline; gap: 6px 12px; }
.team-name {
  font-weight: 600; font-size: 21px; line-height: 1.15;
  letter-spacing: -0.01em; color: var(--ink);
}
.team-record {
  font-family: "JetBrains Mono", monospace;
  font-size: 13px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
  display: inline-flex; align-items: baseline; gap: 6px; white-space: nowrap;
}
.streak {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px; font-weight: 500;
  padding: 2px 6px; border-radius: 4px;
  color: var(--muted); background: var(--chip-bg);
}
.streak.up { color: var(--good); background: color-mix(in oklab, var(--good) 10%, var(--chip-bg)); }
.streak.down { color: var(--poor); background: color-mix(in oklab, var(--poor) 10%, var(--chip-bg)); }

.team-stats {
  grid-column: 1 / -1;
  font-family: "JetBrains Mono", monospace;
  font-size: 11px; letter-spacing: 0.02em;
  color: var(--muted); font-variant-numeric: tabular-nums;
  display: flex; flex-wrap: wrap; gap: 12px;
  margin-top: -4px;
}
.team-stats .k { color: var(--muted-2); letter-spacing: 0.06em; font-size: 9.5px; text-transform: uppercase; margin-right: 4px; }
.team-stats .v { color: var(--ink); font-weight: 500; }

.at-divider {
  display: flex; align-items: center; gap: 12px;
  color: var(--muted); font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-size: 13px; padding-left: 4px;
}
.at-divider::before, .at-divider::after {
  content: ""; flex: 1; height: 1px; background: var(--rule);
}
.at-divider::before { flex: 0 0 12px; }

/* ----- MODEL PREDICTION ----- */
.prediction {
  border-top: 1px dashed var(--rule);
  padding-top: 14px;
  display: flex; flex-direction: column; gap: 6px;
}
.prediction .section-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.prob-bar {
  display: flex; height: 30px; border-radius: 8px; overflow: hidden;
  background: var(--chip-bg); border: 1px solid var(--rule);
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  font-variant-numeric: tabular-nums; letter-spacing: 0.01em;
}
.prob-bar .side {
  display: flex; align-items: center; gap: 6px;
  padding: 0 10px; color: var(--muted);
  min-width: 0; white-space: nowrap; overflow: hidden;
}
.prob-bar .side.away { background: color-mix(in oklab, var(--muted) 20%, var(--chip-bg)); justify-content: flex-start; }
.prob-bar .side.home { background: color-mix(in oklab, var(--muted) 12%, var(--chip-bg)); justify-content: flex-end; }
.prob-bar[data-favors="home"] .side.home { background: color-mix(in oklab, var(--good) 26%, var(--chip-bg)); color: var(--ink); }
.prob-bar[data-favors="home"] .side.home .team,
.prob-bar[data-favors="home"] .side.home .pct { color: var(--ink); font-weight: 600; }
.prob-bar[data-favors="away"] .side.away { background: color-mix(in oklab, var(--good) 26%, var(--chip-bg)); color: var(--ink); }
.prob-bar[data-favors="away"] .side.away .team,
.prob-bar[data-favors="away"] .side.away .pct { color: var(--ink); font-weight: 600; }
.prob-bar .team { text-overflow: ellipsis; overflow: hidden; }
.prob-bar .pct { font-weight: 500; }
.prediction .no-pred {
  font-size: 12px; color: var(--muted-2); font-style: italic;
  padding: 6px 0;
}

/* ----- ODDS COMPARISON ----- */
.odds-block {
  border-top: 1px dashed var(--rule);
  padding-top: 14px;
  display: flex; flex-direction: column; gap: 8px;
}
.odds-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px;
}
.odds-head .limit {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; color: var(--muted-2); letter-spacing: 0.06em;
}
.odds-table {
  display: grid;
  grid-template-columns: 1fr auto auto auto auto;
  gap: 4px 10px;
  font-family: "JetBrains Mono", monospace;
  font-size: 11.5px; font-variant-numeric: tabular-nums;
  align-items: center;
}
.odds-table .hd {
  color: var(--muted-2); font-size: 9.5px;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.odds-table .side-name { color: var(--ink); font-weight: 500; font-size: 12px; }
.odds-table .model, .odds-table .fair, .odds-table .price { color: var(--muted); }
.odds-table .ev { font-weight: 500; text-align: right; }
.odds-table .ev.pos { color: var(--good); }
.odds-table .ev.pos.strong {
  color: var(--good);
  background: color-mix(in oklab, var(--good) 14%, transparent);
  padding: 1px 6px; border-radius: 4px;
}
.odds-table .ev.neg { color: var(--muted-2); }
.odds-no-odds {
  font-size: 12px; color: var(--muted-2); font-style: italic;
}
.ev-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 1px 6px; border-radius: 4px;
  background: color-mix(in oklab, var(--good) 18%, transparent);
  color: var(--good); font-weight: 600;
}
.book-row {
  display: flex; justify-content: space-between;
  padding-top: 6px; border-top: 1px dotted var(--rule);
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted);
}
.book-row .b { color: var(--ink); }

/* ----- PITCHERS ----- */
.pitchers {
  border-top: 1px dashed var(--rule); padding-top: 14px;
  display: flex; flex-direction: column; gap: 14px;
}
.section-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.pitcher {
  display: grid; grid-template-columns: 10px 1fr auto;
  align-items: baseline; gap: 12px;
}
.dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--muted); align-self: center;
}
.dot.good { background: var(--good); }
.dot.warn { background: var(--warn); }
.dot.poor { background: var(--poor); }
.dot.none { background: transparent; border: 1px dashed var(--muted); }
.pitcher-name { font-size: 14.5px; font-weight: 600; color: var(--ink); line-height: 1.2; }
.pitcher-role {
  display: block; font-family: "JetBrains Mono", monospace;
  font-size: 9.5px; letter-spacing: 0.12em; font-weight: 500;
  text-transform: uppercase; color: var(--muted); margin-bottom: 3px;
}
.pitcher-wl {
  font-family: "JetBrains Mono", monospace;
  font-size: 12.5px; font-weight: 500;
  color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap;
}
.pitcher-stats {
  grid-column: 2 / -1;
  display: flex; flex-wrap: wrap; gap: 10px;
  font-family: "JetBrains Mono", monospace;
  font-size: 11px; font-variant-numeric: tabular-nums;
  color: var(--muted); letter-spacing: 0.01em;
  margin-top: 2px;
}
.pitcher-stats span { white-space: nowrap; }
.pitcher-stats .v { color: var(--ink); font-weight: 500; }

.context {
  display: flex; flex-wrap: wrap; gap: 6px 12px;
  color: var(--muted); font-size: 12px;
  border-top: 1px solid var(--rule);
  margin-top: 2px; padding-top: 12px;
}
.context .item { display: inline-flex; align-items: center; gap: 6px; }
.context .item .icon { color: var(--muted-2); font-size: 11px; }
.context .item.tv { color: var(--ink); font-weight: 500; }

.empty, .error {
  max-width: 520px; margin: 80px auto; text-align: center;
}
.empty h2 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: 30px; margin: 0 0 8px; letter-spacing: -0.01em;
}
.empty p { color: var(--muted); margin: 0; }
.error {
  padding: 20px 24px; border: 1px solid var(--poor);
  border-radius: 12px;
  background: color-mix(in oklab, var(--poor) 8%, var(--card));
  color: var(--ink); text-align: left;
}

.legend {
  display: flex; gap: 14px; align-items: center;
  color: var(--muted); font-size: 11px;
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend .dot { width: 7px; height: 7px; }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab active" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab" href="/analyst">AI Analyst</a>
        <a class="nav-tab" href="/market">Market</a>
        <a class="nav-tab" href="/backtest">Model</a>
      </nav>
    </div>
    <form class="controls" method="get" action="/">
      <a class="btn icon" href="/?date={{ prev_date }}" aria-label="Previous day" title="Previous day">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()" aria-label="Choose date">
      <a class="btn icon" href="/?date={{ next_date }}" aria-label="Next day" title="Next day">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/?date={{ today }}">Today</a>{% endif %}
      <a class="btn csv" href="/export.csv?date={{ date_str }}" title="Download CSV for this date">CSV</a>
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>{{ date_pretty }}</h1>
    <div class="count">
      {% if game_count %}<strong>{{ game_count }}</strong> game{{ '' if game_count == 1 else 's' }}{% else %}No games{% endif %}
    </div>
  </div>

  {% if error %}
  <div class="error"><strong>Couldn't load games.</strong><span>{{ error }}</span></div>
  {% elif not games %}
  <div class="empty"><h2>No games on the slate.</h2><p>Off-day, break, or the season is between phases. Try a different date.</p></div>
  {% else %}
  <div class="grid">
    {% for g in games %}
    <article class="card">
      <div class="card-head">
        <span class="time">{{ g.first_pitch }}</span>
        {% if g.series_num and g.games_in_series %}<span class="meta-chip">Gm {{ g.series_num }} of {{ g.games_in_series }}</span>{% endif %}
        {% if g.status and g.status not in ('Scheduled', 'Pre-Game', 'Warmup') %}<span class="meta-chip status">{{ g.status }}</span>{% endif %}
      </div>

      <div class="teams">
        <div class="team-row">
          <div class="team-name">{{ g.away.team }}</div>
          <div class="team-record">
            <span>{{ g.away.wins if g.away.wins is not none else '-' }}-{{ g.away.losses if g.away.losses is not none else '-' }}</span>
            {% if g.away.streak_code %}
              <span class="streak {{ 'up' if g.away.streak_type == 'wins' else ('down' if g.away.streak_type == 'losses' else '') }}">
                {{ g.away.streak_glyph }} {{ g.away.streak_code }}
              </span>
            {% endif %}
          </div>
          <div class="team-stats">
            {% if g.away.avg %}<span><span class="k">AVG</span><span class="v">{{ g.away.avg }}</span></span>{% endif %}
            {% if g.away.ops %}<span><span class="k">OPS</span><span class="v">{{ g.away.ops }}</span></span>{% endif %}
            {% if g.away.hr is not none %}<span><span class="k">HR</span><span class="v">{{ g.away.hr }}</span></span>{% endif %}
            {% if g.away.team_era %}<span><span class="k">ERA</span><span class="v">{{ g.away.team_era }}</span></span>{% endif %}
            {% if g.away.run_diff is not none %}<span><span class="k">RUN DIFF</span><span class="v">{{ '%+d'|format(g.away.run_diff) }}</span></span>{% endif %}
          </div>
        </div>
        <div class="at-divider">at</div>
        <div class="team-row">
          <div class="team-name">{{ g.home.team }}</div>
          <div class="team-record">
            <span>{{ g.home.wins if g.home.wins is not none else '-' }}-{{ g.home.losses if g.home.losses is not none else '-' }}</span>
            {% if g.home.streak_code %}
              <span class="streak {{ 'up' if g.home.streak_type == 'wins' else ('down' if g.home.streak_type == 'losses' else '') }}">
                {{ g.home.streak_glyph }} {{ g.home.streak_code }}
              </span>
            {% endif %}
          </div>
          <div class="team-stats">
            {% if g.home.avg %}<span><span class="k">AVG</span><span class="v">{{ g.home.avg }}</span></span>{% endif %}
            {% if g.home.ops %}<span><span class="k">OPS</span><span class="v">{{ g.home.ops }}</span></span>{% endif %}
            {% if g.home.hr is not none %}<span><span class="k">HR</span><span class="v">{{ g.home.hr }}</span></span>{% endif %}
            {% if g.home.team_era %}<span><span class="k">ERA</span><span class="v">{{ g.home.team_era }}</span></span>{% endif %}
            {% if g.home.run_diff is not none %}<span><span class="k">RUN DIFF</span><span class="v">{{ '%+d'|format(g.home.run_diff) }}</span></span>{% endif %}
          </div>
        </div>
      </div>

      <div class="prediction">
        <div class="section-label">Model Win Probability</div>
        {% if g.p_home is not none %}
        <div class="prob-bar" data-favors="{{ 'home' if g.favors_home else 'away' }}">
          <div class="side away" style="width: {{ g.p_away_pct }}%">
            <span class="team">{{ g.away.team }}</span><span class="pct">{{ g.p_away_pct }}%</span>
          </div>
          <div class="side home" style="width: {{ g.p_home_pct }}%">
            <span class="pct">{{ g.p_home_pct }}%</span><span class="team">{{ g.home.team }}</span>
          </div>
        </div>
        {% else %}
        <div class="no-pred">Prediction unavailable (missing pitcher or model state)</div>
        {% endif %}
      </div>

      {# ---- MARKET ODDS vs MODEL ---- #}
      {% set o = g.odds %}
      {% if o and o.pinnacle and o.pinnacle.moneyline and o.pinnacle.moneyline.away_am %}
      <div class="odds-block">
        <div class="odds-head">
          <span class="section-label">Pinnacle Moneyline vs Model</span>
          {% if o.pinnacle.ml_limit %}
          <span class="limit" title="Max accepted bet — proxy for market confidence">max ${{ '{:,}'.format(o.pinnacle.ml_limit|int) }}</span>
          {% endif %}
        </div>
        <div class="odds-table">
          <span class="hd"></span>
          <span class="hd">Model</span>
          <span class="hd">Fair</span>
          <span class="hd">Price</span>
          <span class="hd">EV</span>

          {% set p_fair_a = o.pin_fair.away if o.pin_fair else none %}
          {% set p_fair_h = o.pin_fair.home if o.pin_fair else none %}
          {% set am_a = o.pinnacle.moneyline.away_am %}
          {% set am_h = o.pinnacle.moneyline.home_am %}
          {% set dec_a = o.pin_decimal.away %}
          {% set dec_h = o.pin_decimal.home %}
          {% set ev_a = o.ev_pinnacle.away %}
          {% set ev_h = o.ev_pinnacle.home %}

          <span class="side-name">{{ g.away.team }}</span>
          <span class="model">{{ g.p_away_pct ~ '%' if g.p_away_pct is not none else '-' }}</span>
          <span class="fair">{{ (p_fair_a * 100)|round|int ~ '%' if p_fair_a else '-' }}</span>
          <span class="price">{{ ('+' if am_a > 0 else '') ~ am_a }}{% if dec_a %} <span style="color:var(--muted-2)">({{ '%.2f'|format(dec_a) }})</span>{% endif %}</span>
          <span class="ev {{ 'pos strong' if ev_a and ev_a >= 2 else ('pos' if ev_a and ev_a > 0 else 'neg') }}">
            {{ ('%+.1f'|format(ev_a)) ~ '%' if ev_a is not none else '-' }}
          </span>

          <span class="side-name">{{ g.home.team }}</span>
          <span class="model">{{ g.p_home_pct ~ '%' if g.p_home_pct is not none else '-' }}</span>
          <span class="fair">{{ (p_fair_h * 100)|round|int ~ '%' if p_fair_h else '-' }}</span>
          <span class="price">{{ ('+' if am_h > 0 else '') ~ am_h }}{% if dec_h %} <span style="color:var(--muted-2)">({{ '%.2f'|format(dec_h) }})</span>{% endif %}</span>
          <span class="ev {{ 'pos strong' if ev_h and ev_h >= 2 else ('pos' if ev_h and ev_h > 0 else 'neg') }}">
            {{ ('%+.1f'|format(ev_h)) ~ '%' if ev_h is not none else '-' }}
          </span>
        </div>
        {% if o.odds_api_available and o.best_decimal.home and o.best_decimal.away %}
        <div class="book-row">
          <span>Best US book: <span class="b">{{ g.away.team }}</span> {{ '%.2f'|format(o.best_decimal.away) }} @ {{ o.best_book.away }}</span>
          <span><span class="b">{{ g.home.team }}</span> {{ '%.2f'|format(o.best_decimal.home) }} @ {{ o.best_book.home }}</span>
        </div>
        {% endif %}
      </div>
      {% else %}
      <div class="odds-block">
        <div class="section-label">Pinnacle Moneyline vs Model</div>
        <div class="odds-no-odds">Odds not available for this game</div>
      </div>
      {% endif %}

      <div class="pitchers">
        <div class="section-label">Starting Pitchers</div>
        <div class="pitcher">
          <span class="dot {{ g.away_pitcher.bucket }}" aria-hidden="true"></span>
          <span class="pitcher-name">
            <span class="pitcher-role">Away{% if g.away_pitcher.hand %} &middot; {{ g.away_pitcher.hand }}HP{% endif %}</span>
            {{ g.away_pitcher.name }}
          </span>
          <span class="pitcher-wl">{{ g.away_pitcher.wl }}</span>
          <div class="pitcher-stats">
            <span><span class="v">{{ g.away_pitcher.era }}</span> ERA</span>
            <span><span class="v">{{ g.away_pitcher.whip }}</span> WHIP</span>
            <span><span class="v">{{ g.away_pitcher.k9 }}</span> K/9</span>
            <span><span class="v">{{ g.away_pitcher.bb9 }}</span> BB/9</span>
            <span><span class="v">{{ g.away_pitcher.ip }}</span> IP</span>
          </div>
        </div>
        <div class="pitcher">
          <span class="dot {{ g.home_pitcher.bucket }}" aria-hidden="true"></span>
          <span class="pitcher-name">
            <span class="pitcher-role">Home{% if g.home_pitcher.hand %} &middot; {{ g.home_pitcher.hand }}HP{% endif %}</span>
            {{ g.home_pitcher.name }}
          </span>
          <span class="pitcher-wl">{{ g.home_pitcher.wl }}</span>
          <div class="pitcher-stats">
            <span><span class="v">{{ g.home_pitcher.era }}</span> ERA</span>
            <span><span class="v">{{ g.home_pitcher.whip }}</span> WHIP</span>
            <span><span class="v">{{ g.home_pitcher.k9 }}</span> K/9</span>
            <span><span class="v">{{ g.home_pitcher.bb9 }}</span> BB/9</span>
            <span><span class="v">{{ g.home_pitcher.ip }}</span> IP</span>
          </div>
        </div>
      </div>

      <div class="context">
        {% if g.venue %}<span class="item"><span class="icon">&diams;</span>{{ g.venue }}</span>{% endif %}
        {% if g.wx_temp %}<span class="item">{{ g.wx_temp }}&deg;{% if g.wx_condition %} {{ g.wx_condition|lower }}{% endif %}</span>{% endif %}
        {% if g.wx_wind %}<span class="item">{{ g.wx_wind }}</span>{% endif %}
        {% if g.tv %}<span class="item tv">{{ g.tv }}</span>{% endif %}
      </div>
    </article>
    {% endfor %}
  </div>
  {% endif %}

  <footer>
    <div class="legend">
      <span><i class="dot good"></i>&lt; 3.75</span>
      <span><i class="dot warn"></i>3.75-4.50</span>
      <span><i class="dot poor"></i>&gt; 4.50</span>
      <span><i class="dot none"></i>none</span>
    </div>
    <div>Data &middot; MLB statsapi &middot; updated {{ now }}</div>
  </footer>
</main>
</body>
</html>
"""


# ============================================================================
# Shared helpers for today's game board (fed into the unified Picks page)
# ============================================================================

def _mlb_schedule_today_lite(date_str):
    """Lightweight MLB schedule — no team-stats/pitcher hydrate. Just the games
    for `date_str` with team names and gameDate (UTC ISO). We optionally merge
    Pinnacle ML quotes via mlb_odds for a quick price preview on the hub."""
    url = f"{STATSAPI}/schedule?sportId=1&date={date_str}"
    try:
        data = _cached(f"mlb_sched_lite_{date_str}", 10 * 60, lambda: fetch_json(url))
    except Exception:
        return []
    games = []
    for date_block in (data or {}).get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams") or {}
            home = ((teams.get("home") or {}).get("team") or {})
            away = ((teams.get("away") or {}).get("team") or {})
            games.append({
                "home_name": home.get("name", ""),
                "away_name": away.get("name", ""),
                "start_time": g.get("gameDate", ""),
                "ml": None,
            })

    # Overlay Pinnacle ML quotes where we have them (free Pinnacle guest API)
    try:
        pinn_games = generic_odds.parse_pinnacle_games(246, ml_outcomes=2) or []
    except Exception:
        pinn_games = []
    pin_by_pair = {}
    for pg in pinn_games:
        key = (
            generic_odds._team_key(pg.get("away_name", "")),
            generic_odds._team_key(pg.get("home_name", "")),
        )
        pin_by_pair[key] = pg
    for g in games:
        k = (generic_odds._team_key(g["away_name"]),
             generic_odds._team_key(g["home_name"]))
        pg = pin_by_pair.get(k)
        if pg and pg.get("ml"):
            g["ml"] = pg["ml"]
    return games


def _sport_games_for_hub(sport, today_date):
    """Return today's games for a given sport, as plain dicts the SCHEDULE_HUB
    template understands. Uses Pinnacle (free) for odds; skips the Odds API to
    stay inside the free tier's credit budget."""
    slug = sport["slug"]
    today_str = today_date.isoformat()
    raw = []
    try:
        if slug == "mlb":
            raw = _mlb_schedule_today_lite(today_str)
        else:
            raw = generic_odds.parse_pinnacle_games(
                sport["pinnacle_league_id"],
                ml_outcomes=sport["ml_outcomes"],
                has_halves=sport.get("has_halves", False),
            ) or []
    except Exception:
        raw = []

    out = []
    for g in raw:
        st = g.get("start_time")
        if not st:
            continue
        try:
            dt = datetime.fromisoformat(st.replace("Z", "+00:00")).astimezone(CENTRAL)
        except Exception:
            continue
        if dt.date() != today_date:
            continue
        out.append({
            "away":       g.get("away_name", ""),
            "home":       g.get("home_name", ""),
            "start_time": st,
            "ml":         g.get("ml"),
            "spread":     g.get("spread"),
            "total":      g.get("total"),
        })
    out.sort(key=lambda x: x["start_time"] or "")
    return out


# /schedule is a legacy alias — now that the Picks page shows all games,
# redirect so old deep links keep working.
@app.route("/schedule")
def schedule_hub_redirect():
    from flask import redirect
    return redirect("/", code=302)


# ============================================================================
# backtest report template
# ============================================================================

BACKTEST_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Model &amp; Backtest</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero-block {
  padding-bottom: 24px; margin-bottom: 32px;
  border-bottom: 1px solid var(--rule);
}
.hero-block h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 10px; font-variation-settings: "opsz" 144;
}
.hero-block .sub {
  color: var(--muted); max-width: 720px;
  font-size: 14.5px; line-height: 1.6;
}

.grid-metrics {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 32px;
}
.metric {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 16px 18px;
}
.metric .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.metric .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 28px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.metric .value.accent { color: var(--accent); }
.metric .value.good { color: var(--good); }
.metric .foot {
  margin-top: 6px; font-size: 11.5px; color: var(--muted);
  font-family: "JetBrains Mono", monospace;
}
.metric .delta.up { color: var(--good); }
.metric .delta.down { color: var(--poor); }

.section {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 24px 28px;
  margin-bottom: 20px;
}
.section h2 {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 500; font-size: 20px; margin: 0 0 12px;
  letter-spacing: -0.01em;
}
.section p.lead {
  color: var(--muted); margin: 0 0 20px;
  font-size: 13.5px; line-height: 1.6; max-width: 720px;
}

/* calibration diagram */
.calib-wrap { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 900px) { .calib-wrap { grid-template-columns: 1fr 1fr; } }
.calib-svg-wrap {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
  padding: 20px; display: flex; justify-content: center;
}
svg.calib { max-width: 100%; height: auto; }

.calib-table {
  width: 100%; border-collapse: collapse;
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 12.5px;
}
.calib-table th, .calib-table td {
  padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--rule);
}
.calib-table th {
  text-align: right; color: var(--muted);
  font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
  font-weight: 500;
}
.calib-table th:first-child, .calib-table td:first-child { text-align: left; }

/* elo table */
.elo-table {
  width: 100%; border-collapse: collapse;
  font-family: "Public Sans", sans-serif;
  font-size: 13.5px;
}
.elo-table th, .elo-table td {
  padding: 8px 10px; border-bottom: 1px solid var(--rule);
}
.elo-table th { text-align: left; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
.elo-table td.num, .elo-table th.num { text-align: right; font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; }
.elo-table td.rank { color: var(--muted); font-family: "JetBrains Mono", monospace; }
.elo-table tbody tr:hover { background: color-mix(in oklab, var(--rule) 30%, transparent); }
.elo-cols { display: grid; grid-template-columns: 1fr; gap: 20px; }
@media (min-width: 900px) { .elo-cols { grid-template-columns: 1fr 1fr; } }

/* predictions sample */
.pred-table {
  width: 100%; border-collapse: collapse;
  font-family: "Public Sans", sans-serif; font-size: 13.5px;
}
.pred-table th, .pred-table td {
  padding: 8px 10px; text-align: left;
  border-bottom: 1px solid var(--rule); vertical-align: middle;
}
.pred-table th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
.pred-table td.num { font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; text-align: right; }
.pred-bar {
  display: inline-block; height: 8px; border-radius: 3px;
  background: color-mix(in oklab, var(--good) 60%, transparent);
  vertical-align: middle;
}
.pred-bar.away { background: color-mix(in oklab, var(--muted) 40%, transparent); }
.pred-correct { color: var(--good); font-weight: 600; }
.pred-wrong { color: var(--muted-2); }

.model-def {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin-top: 12px;
}
.model-def .item {
  padding: 12px 14px; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 8px;
}
.model-def .k {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
}
.model-def .v {
  color: var(--ink); font-size: 13px;
  font-family: "JetBrains Mono", monospace;
}
.model-def .expl {
  color: var(--muted); font-size: 12px; margin-top: 4px;
  line-height: 1.4; font-family: inherit;
}
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab" href="/analyst">AI Analyst</a>
        <a class="nav-tab" href="/market">Market</a>
        <a class="nav-tab active" href="/backtest">Model</a>
      </nav>
    </div>
    <div class="controls">
      <a class="btn" href="/backtest?refresh=1" title="Refit against latest data">Refit</a>
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  {% if not state %}
  <div class="error"><strong>Model not yet fit.</strong> Try again in a moment or click Refit.</div>
  {% else %}

  <div class="hero-block">
    <h1>MLB model &middot; 12-season backtest</h1>
    <p class="sub">
      An Elo rating model with a pregame starting-pitcher adjustment (season-to-date ERA and WHIP vs league,
      weighted by workload). Fit chronologically across <strong>{{ state.first_season }}&ndash;{{ state.last_season }}</strong>
      ({{ state.num_seasons }} seasons, {{ state.total_games }} regular-season games). Team Elo regresses toward 1500 by 1/3
      at each season boundary; pitcher stats reset per season. Predictions are logged <em>before</em> each game using
      only stats available pre-first-pitch; the first {{ state.warmup_seasons }} seasons are pure warm-up and the remaining
      <strong>{{ state.last_season - state.scored_from_season + 1 }} seasons are scored</strong> against actual outcomes below.
    </p>
  </div>

  <div class="grid-metrics">
    <div class="metric">
      <div class="label">Accuracy</div>
      <div class="value accent">{{ '%.1f'|format(m.accuracy * 100) }}%</div>
      <div class="foot">
        <span class="delta up">+{{ '%.1f'|format((m.accuracy - m.home_baseline_accuracy) * 100) }} pts</span>
        vs home-team baseline ({{ '%.1f'|format(m.home_baseline_accuracy * 100) }}%)
      </div>
    </div>
    <div class="metric">
      <div class="label">Log loss</div>
      <div class="value">{{ '%.4f'|format(m.log_loss) }}</div>
      <div class="foot">baseline {{ '%.4f'|format(m.home_baseline_log_loss) }} &middot; random 0.6931</div>
    </div>
    <div class="metric">
      <div class="label">Brier score</div>
      <div class="value">{{ '%.4f'|format(m.brier_score) }}</div>
      <div class="foot">lower is better; 0.25 = coin flip</div>
    </div>
    <div class="metric">
      <div class="label">Scored games</div>
      <div class="value">{{ m.n }}</div>
      <div class="foot">seasons {{ state.scored_from_season }} &rarr; {{ state.last_season }}</div>
    </div>
    {% if m.record_baseline_accuracy is not none %}
    <div class="metric">
      <div class="label">vs better record</div>
      <div class="value">{{ '%.1f'|format(m.record_baseline_accuracy * 100) }}%</div>
      <div class="foot">picking the team with better W-L (rolling)</div>
    </div>
    {% endif %}
    <div class="metric">
      <div class="label">Warm-up games</div>
      <div class="value">{{ state.total_games - m.n }}</div>
      <div class="foot">used only to fit Elo, not scored</div>
    </div>
  </div>

  <div class="section">
    <h2>How the prediction is made</h2>
    <p class="lead">
      Each team carries an Elo rating that updates after every game with a K-factor of {{ state.hyperparams.K }} and a
      margin-of-victory multiplier so blowouts move ratings more than one-run wins. Home field advantage adds
      +{{ state.hyperparams.HFA|int }} Elo points. Before each game, both starters' ratings are adjusted by their season-to-date
      ERA and WHIP relative to league average, weighted by innings pitched (a starter with fewer than {{ state.hyperparams.IP_FULL|int }} IP
      counts partially). Prediction: <code>P(home) = 1 / (1 + 10^((adj_away &minus; adj_home)/400))</code>.
    </p>
    <div class="model-def">
      <div class="item"><div class="k">K factor</div><div class="v">{{ state.hyperparams.K }}</div><div class="expl">Elo update rate per game</div></div>
      <div class="item"><div class="k">Home field</div><div class="v">+{{ state.hyperparams.HFA|int }} Elo</div><div class="expl">Roughly a 54% baseline</div></div>
      <div class="item"><div class="k">ERA weight</div><div class="v">{{ state.hyperparams.ERA_WEIGHT|int }} Elo / 1 ERA</div><div class="expl">vs league {{ state.hyperparams.LEAGUE_ERA }}</div></div>
      <div class="item"><div class="k">WHIP weight</div><div class="v">{{ state.hyperparams.WHIP_WEIGHT|int }} Elo / 1 WHIP</div><div class="expl">vs league {{ state.hyperparams.LEAGUE_WHIP }}</div></div>
      <div class="item"><div class="k">IP full weight</div><div class="v">{{ state.hyperparams.IP_FULL|int }} IP</div><div class="expl">Workload for full pitcher adjustment</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Per-season breakdown</h2>
    <p class="lead">
      Model performance for each scored season. MLB is famously hard to predict and sits around
      55&ndash;59% accuracy across sharp models. 2020's short COVID season (900 games) is included
      but noisier.
    </p>
    <div class="table-scroll">
    <table class="calib-table">
      <thead><tr>
        <th>Season</th><th>Games</th><th>Accuracy</th><th>Log loss</th><th>Brier</th>
      </tr></thead>
      <tbody>
        {% for r in per_season_rows %}
        <tr>
          <td>{{ r.season }}</td>
          <td>{{ r.n }}</td>
          <td>{{ (r.accuracy * 100)|round(2) }}%</td>
          <td>{{ '%.4f'|format(r.log_loss) }}</td>
          <td>{{ '%.4f'|format(r.brier) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </div>

  <div class="section">
    <h2>Calibration</h2>
    <p class="lead">Predicted probabilities are grouped into 10% bins. A well-calibrated model has each bin's average prediction match the observed win rate. The 45&deg; line is perfect calibration.</p>
    <div class="calib-wrap">
      <div class="calib-svg-wrap">
        {{ calibration_svg|safe }}
      </div>
      <div>
        <table class="calib-table">
          <thead><tr>
            <th>Bin</th><th>N</th><th>Avg pred</th><th>Actual</th><th>Delta</th>
          </tr></thead>
          <tbody>
            {% for c in m.calibration %}
            <tr>
              <td>{{ '%.0f'|format(c.bin_lo*100) }}-{{ '%.0f'|format(c.bin_hi*100) }}%</td>
              <td>{{ c.n }}</td>
              <td>{{ '%.3f'|format(c.avg_pred) }}</td>
              <td>{{ '%.3f'|format(c.actual) }}</td>
              <td>{{ '%+.3f'|format(c.actual - c.avg_pred) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Final Elo ratings</h2>
    <p class="lead">End-of-season team ratings after the walk-forward fit. The final ratings feed today's predictions on the Schedule page.</p>
    <div class="elo-cols">
      <div>
        <table class="elo-table">
          <thead><tr><th class="num">#</th><th>Team</th><th class="num">Elo</th><th class="num">W-L</th></tr></thead>
          <tbody>
            {% for tid, elo, wl in top_elo %}
            <tr>
              <td class="rank num">{{ loop.index }}</td>
              <td>{{ state.team_names[tid] }}</td>
              <td class="num">{{ '%.0f'|format(elo) }}</td>
              <td class="num">{{ wl[0] }}-{{ wl[1] }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div>
        <table class="elo-table">
          <thead><tr><th class="num">#</th><th>Team</th><th class="num">Elo</th><th class="num">W-L</th></tr></thead>
          <tbody>
            {% for tid, elo, wl in bottom_elo %}
            <tr>
              <td class="rank num">{{ loop.index + 15 }}</td>
              <td>{{ state.team_names[tid] }}</td>
              <td class="num">{{ '%.0f'|format(elo) }}</td>
              <td class="num">{{ wl[0] }}-{{ wl[1] }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Sample of scored predictions</h2>
    <p class="lead">Last 30 games in the scored window. Prediction was made using only pregame information; the actual outcome is shown for comparison.</p>
    <table class="pred-table">
      <thead><tr>
        <th>Date</th>
        <th>Matchup</th>
        <th class="num">Pred</th>
        <th>Actual</th>
        <th></th>
      </tr></thead>
      <tbody>
      {% for p in sample_preds %}
        <tr>
          <td class="num">{{ p.date }}</td>
          <td>{{ p.away }} at {{ p.home }}</td>
          <td class="num">
            {% set pct = (p.p_home * 100)|int %}
            {% if p.p_home >= 0.5 %}{{ pct }}% home{% else %}{{ 100 - pct }}% away{% endif %}
          </td>
          <td class="num">{{ p.away_score }}&ndash;{{ p.home_score }}</td>
          <td>
            {% set model_pick_home = p.p_home >= 0.5 %}
            {% if model_pick_home == p.home_won %}
              <span class="pred-correct">&check;</span>
            {% else %}
              <span class="pred-wrong">&times;</span>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <footer>
    <div>Model fit on {{ state.total_games }} games &middot; last refit {{ state.generated_at }}</div>
    <div>Data &middot; MLB statsapi</div>
  </footer>

  {% endif %}
</main>
</body>
</html>
"""


# ============================================================================
# calibration diagram (inline SVG)
# ============================================================================

def render_calibration_svg(calibration):
    """Small reliability diagram: predicted vs observed win rate, 45-degree line."""
    W, H = 340, 340
    pad_l, pad_r, pad_t, pad_b = 44, 14, 14, 40

    def x(v): return pad_l + v * (W - pad_l - pad_r)
    def y(v): return H - pad_b - v * (H - pad_t - pad_b)

    parts = [f'<svg class="calib" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']

    # grid
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(t):.1f}" y1="{y(0):.1f}" x2="{x(t):.1f}" y2="{y(1):.1f}" stroke="var(--rule)" stroke-width="1" />')
        parts.append(f'<line x1="{x(0):.1f}" y1="{y(t):.1f}" x2="{x(1):.1f}" y2="{y(t):.1f}" stroke="var(--rule)" stroke-width="1" />')

    # 45-degree reference line
    parts.append(f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" stroke="var(--muted-2)" stroke-width="1" stroke-dasharray="4 4" />')

    # ticks + labels
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{x(t):.1f}" y="{H - pad_b + 16:.1f}" font-size="10" fill="var(--muted)" text-anchor="middle" font-family="JetBrains Mono, monospace">{int(t*100)}%</text>')
        parts.append(f'<text x="{pad_l - 8:.1f}" y="{y(t) + 3:.1f}" font-size="10" fill="var(--muted)" text-anchor="end" font-family="JetBrains Mono, monospace">{int(t*100)}%</text>')

    # axis labels
    parts.append(f'<text x="{W/2:.1f}" y="{H-6:.1f}" font-size="10" fill="var(--muted)" text-anchor="middle" font-family="JetBrains Mono, monospace" letter-spacing="1.5">PREDICTED</text>')
    parts.append(f'<text x="14" y="{H/2:.1f}" font-size="10" fill="var(--muted)" text-anchor="middle" font-family="JetBrains Mono, monospace" letter-spacing="1.5" transform="rotate(-90 14 {H/2:.1f})">OBSERVED</text>')

    # data points sized by n
    if calibration:
        max_n = max(c["n"] for c in calibration)
        for c in calibration:
            r = 4 + 10 * (c["n"] / max_n) ** 0.5
            cx = x(c["avg_pred"])
            cy = y(c["actual"])
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.2"/>')

    parts.append('</svg>')
    return "".join(parts)


# ============================================================================
# routes
# ============================================================================

PICKS_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding: 20px 0 12px; margin-bottom: 12px; border-bottom: 1px solid var(--rule); }
.hero h1 {
  font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400;
  font-size: clamp(26px, 4vw, 36px); line-height: 1.05;
  margin: 0 0 6px;
}
.hero .sub { color: var(--muted); font-size: 12.5px; margin: 0; }
.hero .sub b { color: var(--ink); font-weight: 500; }

.summary-strip { display: flex; flex-wrap: wrap; gap: 18px; margin: 10px 0 14px; }
.summary-strip .stat {
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  color: var(--muted); letter-spacing: 0.04em;
}
.summary-strip .stat b { color: var(--ink); font-weight: 500; font-size: 13px; margin-right: 4px; }
.summary-strip .stat.good b { color: var(--good); text-shadow: var(--ev-strong-glow); }

.filter-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.filter-chips .chip {
  padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--rule-strong);
  background: var(--surface); color: var(--muted);
  font-size: 11.5px; font-weight: 500; text-decoration: none;
}
.filter-chips .chip:hover { color: var(--ink); background: var(--card); }
.filter-chips .chip.active { color: var(--ink); background: var(--card); border-color: var(--ink); }

.board-wrap {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  overflow: hidden;
}
.board-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.board-table thead th {
  text-align: left; padding: 10px 12px;
  font-family: "JetBrains Mono", monospace; font-weight: 500;
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); border-bottom: 1px solid var(--rule);
  background: var(--surface);
}
.board-table tbody td {
  padding: 12px; border-top: 1px solid var(--rule);
  vertical-align: top;
}
.board-table tr.row-main:first-child td { border-top: none; }
.board-table .col-status  { width: 90px; }
.board-table .col-time    { width: 110px; }
.board-table .col-match   { min-width: 180px; }
.board-table .col-score   { width: 70px; text-align: center; }
.board-table .col-line    { width: 180px; }
.board-table .col-close   { width: 150px; }
.board-table .col-picks   { min-width: 230px; }
.board-table .col-view    { width: 70px; text-align: right; }

.status-chip {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-family: "JetBrains Mono", monospace; font-size: 9.5px;
  letter-spacing: 0.14em; text-transform: uppercase; font-weight: 500;
  background: var(--surface); border: 1px solid var(--rule);
  color: var(--muted);
}
.status-chip.st-upcoming { color: var(--accent); border-color: var(--accent); }
.status-chip.st-live     { color: #ef4444; border-color: #ef4444;
  background: color-mix(in oklab, #ef4444 10%, transparent); animation: pulse 1.6s infinite; }
.status-chip.st-final    { color: var(--muted); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

.col-time { font-family: "JetBrains Mono", monospace; color: var(--muted); font-size: 11.5px; }
.col-time .countdown { color: var(--ink); display: block; font-size: 12px; }
.col-time .absolute  { color: var(--muted-2); display: block; font-size: 10.5px; margin-top: 2px; }

.col-match .sport-tag {
  display: inline-block;
  font-family: "JetBrains Mono", monospace; font-size: 9px;
  letter-spacing: 0.12em; text-transform: uppercase; font-weight: 500;
  padding: 1px 6px; border-radius: 3px;
  background: var(--surface); color: var(--muted); border: 1px solid var(--rule);
  margin-bottom: 4px;
}
.col-match .team { display: block; color: var(--ink); font-weight: 500; font-size: 13px; line-height: 1.35; }
.col-match .at { color: var(--muted); font-size: 11px; display: block; margin: 1px 0; }

.col-score { font-family: "JetBrains Mono", monospace; font-size: 15px; font-weight: 600; color: var(--ink); }
.col-score .pending { color: var(--muted-2); font-weight: 400; font-size: 13px; }
.col-score .score-away { color: var(--ink); }
.col-score .score-dash { color: var(--muted); margin: 0 3px; }

.lines-block { font-family: "JetBrains Mono", monospace; font-size: 11.5px; color: var(--muted); line-height: 1.5; }
.lines-block .ln-row { display: flex; gap: 6px; align-items: baseline; }
.lines-block .ln-k { color: var(--muted-2); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; min-width: 32px; }
.lines-block .ln-v { color: var(--ink); font-variant-numeric: tabular-nums; }
.lines-block .ln-v .book { color: var(--muted-2); font-size: 10px; margin-left: 4px; }

.pick-badge-row {
  display: flex; gap: 8px; align-items: center;
  padding: 5px 0; font-size: 12px;
}
.pick-badge-row + .pick-badge-row { border-top: 1px dashed var(--rule); margin-top: 2px; padding-top: 7px; }
.pick-badge-row .pick-ico {
  display: inline-block; width: 18px; height: 18px; border-radius: 50%;
  text-align: center; line-height: 18px; font-size: 10px; font-weight: 700;
  background: var(--surface); color: var(--muted); border: 1px solid var(--rule);
  flex-shrink: 0;
}
.pick-badge-row .pick-ico.ml   { background: color-mix(in oklab, var(--good) 20%, var(--card)); color: var(--good); border-color: color-mix(in oklab, var(--good) 40%, var(--rule)); }
.pick-badge-row .pick-ico.spr  { background: color-mix(in oklab, var(--accent) 20%, var(--card)); color: var(--accent); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); }
.pick-badge-row .pick-ico.tot  { background: color-mix(in oklab, #f59e0b 20%, var(--card)); color: #f59e0b; border-color: color-mix(in oklab, #f59e0b 40%, var(--rule)); }
.pick-badge-row .pick-ico.dc   { background: color-mix(in oklab, #a78bfa 20%, var(--card)); color: #a78bfa; border-color: color-mix(in oklab, #a78bfa 40%, var(--rule)); font-size: 8px; }
.pick-badge-row .pick-ico.btts { background: color-mix(in oklab, #ec4899 20%, var(--card)); color: #ec4899; border-color: color-mix(in oklab, #ec4899 40%, var(--rule)); }
.pick-badge-row .pick-lbl { color: var(--ink); font-weight: 500; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.pick-badge-row .pick-conf {
  font-family: "JetBrains Mono", monospace; font-size: 10.5px;
  color: var(--muted); font-variant-numeric: tabular-nums;
  padding: 1px 7px; border-radius: 4px; background: var(--surface);
  border: 1px solid var(--rule);
}
.pick-badge-row .pick-conf.c5 { color: var(--good); border-color: var(--good); background: color-mix(in oklab, var(--good) 12%, transparent); }
.pick-badge-row .pick-conf.c4 { color: var(--good); }
.pick-badge-row .pick-res { font-size: 13px; font-weight: 700; margin-left: 2px; }
.pick-badge-row .pick-res.W { color: var(--good); }
.pick-badge-row .pick-res.L { color: #ef4444; }
.pick-badge-row .pick-res.P { color: var(--muted-2); }
.pick-badge-row .strong-star { color: var(--good); font-size: 11px; margin-left: 2px; text-shadow: var(--ev-strong-glow); }
.pick-badge-row.model-only { opacity: 0.75; }
.pick-badge-row.model-only .pick-lbl { font-weight: 400; color: var(--muted); }
.pick-badge-row.model-only .pick-lbl b { color: var(--ink); font-weight: 500; }

.btn-view {
  padding: 5px 12px; border-radius: 6px;
  border: 1px solid var(--rule-strong);
  background: var(--surface); color: var(--ink);
  font-size: 11.5px; font-weight: 500;
  cursor: pointer; font-family: inherit;
}
.btn-view:hover { background: var(--card); border-color: var(--ink); }

.no-picks-dash { color: var(--muted-2); font-size: 13px; }
.col-close .close-pending { color: var(--muted-2); font-size: 11px; font-style: italic; }

.row-expanded { background: color-mix(in oklab, var(--accent) 4%, transparent); }
.row-expanded td { padding: 14px 20px; }
.expanded-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 16px;
}
.pick-detail {
  background: var(--card); border: 1px solid var(--rule); border-radius: 8px;
  padding: 12px 16px;
}
.pick-detail.strong { box-shadow: inset 3px 0 0 var(--good); }
.pick-detail-head {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;
  margin-bottom: 8px;
}
.pick-detail-head strong { font-size: 15px; color: var(--ink); }
.pick-detail-head .market-tag {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  color: var(--accent); letter-spacing: 0.1em; text-transform: uppercase;
}
.pick-detail-head .price {
  font-family: "JetBrains Mono", monospace; font-size: 12px;
  color: var(--muted); font-variant-numeric: tabular-nums;
}
.pick-detail-head .book { font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--muted-2); }
.pick-detail-head .strong-badge {
  font-family: "JetBrains Mono", monospace; font-size: 9.5px;
  letter-spacing: 0.14em; color: var(--good); font-weight: 700;
  text-shadow: var(--ev-strong-glow);
}
.pick-stats-detail {
  display: flex; flex-wrap: wrap; gap: 14px;
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); margin-bottom: 10px;
}
.pick-stats-detail .ev { color: var(--good); font-weight: 600; text-shadow: var(--ev-strong-glow); }
.pick-stats-detail b { color: var(--ink); font-weight: 500; }
.pick-stats-detail .label { color: var(--muted-2); margin-right: 4px; font-size: 9.5px; letter-spacing: 0.06em; text-transform: uppercase; }
.pick-bulletin {
  color: var(--muted); font-size: 12.5px; line-height: 1.55;
  padding: 8px 10px; border-left: 2px solid var(--accent);
  background: color-mix(in oklab, var(--accent) 6%, transparent);
  border-radius: 0 6px 6px 0; margin-bottom: 8px;
}
.pick-bulletin b { color: var(--ink); }
.poly-signal {
  margin: 6px 0; padding: 6px 10px; border-radius: 6px;
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  border-left: 3px solid var(--rule-strong);
  background: var(--surface); color: var(--ink);
}
.poly-signal.poly-good { border-left-color: var(--good); background: color-mix(in oklab, var(--good) 10%, var(--card)); }
.poly-signal.poly-bad  { border-left-color: #ef4444;   background: color-mix(in oklab, #ef4444 10%, var(--card)); }
.poly-signal.poly-info { border-left-color: var(--accent); color: var(--muted); }
.pick-detail .btn-ai {
  padding: 5px 12px; border-radius: 6px;
  border: 1px solid var(--accent);
  background: color-mix(in oklab, var(--accent) 8%, transparent);
  color: var(--accent); font-weight: 500; font-size: 11.5px;
  cursor: pointer;
}
.pick-detail .btn-ai:hover { background: color-mix(in oklab, var(--accent) 18%, transparent); box-shadow: 0 0 10px var(--accent-glow); }
.pick-detail .btn-ai:disabled { opacity: 0.6; cursor: wait; }
.pick-detail .ai-analysis { padding: 10px 0 0; font-size: 12.5px; display: none; margin-top: 10px; border-top: 1px dashed var(--rule); }
.pick-detail .ai-analysis.visible { display: block; }
.pick-detail .ai-analysis h5 {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); margin: 10px 0 6px; font-weight: 500;
}
.pick-detail .ai-analysis ul { margin: 0 0 8px; padding-left: 20px; }
.pick-detail .ai-analysis li { margin-bottom: 4px; color: var(--ink); }
.pick-detail .ai-analysis .ai-pick {
  background: color-mix(in oklab, var(--good) 15%, transparent);
  box-shadow: inset 3px 0 0 var(--good);
  padding: 10px 14px; border-radius: 6px; margin-top: 10px;
}
.pick-detail .ai-analysis .ai-err { color: var(--accent); font-style: italic; font-size: 12px; }
.pick-detail .ai-analysis .loading { color: var(--muted); font-style: italic; }

.model-src { font-family: "JetBrains Mono", monospace; font-size: 9.5px; color: var(--muted-2); letter-spacing: 0.04em; margin-left: 10px; }

.empty-board {
  padding: 60px 24px; text-align: center; color: var(--muted);
  background: var(--card); border: 1px dashed var(--rule); border-radius: 10px;
}
.empty-board h2 {
  font-family: "Fraunces", Georgia, serif; font-style: italic;
  font-size: 24px; margin: 0 0 8px; color: var(--ink);
}

@media (max-width: 900px) {
  .board-table .col-close { display: none; }
  .board-table .col-line { width: auto; }
}
@media (max-width: 680px) {
  .board-wrap { border: none; background: transparent; }
  .board-table thead { display: none; }
  .board-table tbody td { display: block; border: none; padding: 4px 0; }
  .board-table tr.row-main {
    display: block; background: var(--card); border: 1px solid var(--rule);
    border-radius: 10px; padding: 12px; margin-bottom: 10px;
  }
  .board-table .col-score, .board-table .col-line, .board-table .col-close { text-align: left; }
}
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
    </div>
    <form class="controls" method="get" action="/">
      <a class="btn icon" href="/?date={{ prev_date }}">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()">
      <a class="btn icon" href="/?date={{ next_date }}">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/">Today</a>{% endif %}
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>{{ date_pretty }}</h1>
    <p class="sub">
      <b>{{ total_games }}</b> games across <b>{{ sports_with_games }}</b> sports &middot;
      <b>{{ picks_count }}</b> picks ({{ ev_count }} EV-verified, {{ strong_count }} strong) &middot;
      all times Central. Every game shows our model's preferred ML / spread / total
      side; EV-verified picks (model + Pinnacle devig consensus clears a +EV bar)
      are bolded, model-only picks are muted. N/5 = confidence tier from consensus
      probability.
    </p>
  </div>

  {% if active_adj and active_adj.any %}
  <div class="active-adj" style="margin: 4px 0 14px;">
    <strong>Analyzer-tuned:</strong>
    {% for sport, w in active_adj.weights.items() %}
      <span class="adj-chip">{{ sport }} weights {{ (w.pricing*100)|int }}/{{ (w.mc*100)|int }}</span>
    {% endfor %}
    {% for sport, f in active_adj.floors.items() %}
      <span class="adj-chip">{{ sport }} min prob {{ (f*100)|int }}%</span>
    {% endfor %}
    {% for m in active_adj.blacklist %}
      <span class="adj-chip bad">drop {{ m }}</span>
    {% endfor %}
    <a href="/logged" class="adj-chip" style="text-decoration:none">why?</a>
  </div>
  {% endif %}

  <div class="filter-chips">
    <a class="chip {% if sport_filter == 'all' %}active{% endif %}" href="/?date={{ date_str }}">All ({{ total_games }})</a>
    {% for s in sport_counts %}
      <a class="chip {% if sport_filter == s.slug %}active{% endif %}" href="/?date={{ date_str }}&sport={{ s.slug }}">{{ s.name }} ({{ s.count }})</a>
    {% endfor %}
  </div>

  {% if board_rows %}
  <div class="board-wrap">
    <table class="board-table">
      <thead>
        <tr>
          <th class="col-status">Status</th>
          <th class="col-time">Time</th>
          <th class="col-match">Match</th>
          <th class="col-score">Score</th>
          <th class="col-line">Line</th>
          <th class="col-close">Close</th>
          <th class="col-picks">Picks</th>
          <th class="col-view"></th>
        </tr>
      </thead>
      <tbody>
        {% for row in board_rows %}
        <tr class="row-main" data-row-id="{{ row.id }}">
          <td class="col-status">
            <span class="status-chip st-{{ row.status }}">{{ row.status|upper }}</span>
          </td>
          <td class="col-time">
            {% if row.countdown %}<span class="countdown">{{ row.countdown }}</span>{% endif %}
            <span class="absolute">{{ row.time_abs }}</span>
          </td>
          <td class="col-match">
            <span class="sport-tag">{{ row.sport_name }}</span>
            <span class="team">{{ row.away }}</span>
            <span class="at">@</span>
            <span class="team">{{ row.home }}</span>
          </td>
          <td class="col-score">
            {% if row.score_home is not none %}
              <span class="score-away">{{ row.score_away }}</span><span class="score-dash">&ndash;</span>{{ row.score_home }}
            {% else %}
              <span class="pending">&mdash;</span>
            {% endif %}
          </td>
          <td class="col-line">
            {% if row.lines %}
              {% for ln in row.lines %}
                <div class="ln-row"><span class="ln-k">{{ ln.k }}</span><span class="ln-v">{{ ln.v }}</span></div>
              {% endfor %}
            {% else %}
              <span class="no-picks-dash">&mdash;</span>
            {% endif %}
          </td>
          <td class="col-close">
            {% if row.close_lines %}
              {% for ln in row.close_lines %}
                <div class="ln-row"><span class="ln-k">{{ ln.k }}</span><span class="ln-v">{{ ln.v }}</span></div>
              {% endfor %}
            {% else %}
              <span class="close-pending">pending</span>
            {% endif %}
          </td>
          <td class="col-picks">
            {% if row.picks %}
              {% for p in row.picks %}
                <div class="pick-badge-row {{ 'model-only' if p.model_only else '' }}" title="{{ 'Model-only preference (no EV filter)' if p.model_only else 'EV-verified pick' }}">
                  <span class="pick-ico {{ p.icon_cls }}">{{ p.icon }}</span>
                  <span class="pick-lbl">{{ p.pick_short }}</span>
                  <span class="pick-conf c{{ p.conf }}">{{ p.conf }}/5</span>
                  {% if p.strong %}<span class="strong-star" title="Model &ge; 60% AND EV &ge; +4%">&#9733;</span>{% endif %}
                  {% if p.result %}<span class="pick-res {{ p.result }}" title="{{ p.result }}">{{ '✓' if p.result == 'W' else ('✗' if p.result == 'L' else '—') }}</span>{% endif %}
                </div>
              {% endfor %}
            {% else %}
              <span class="no-picks-dash">no pick</span>
            {% endif %}
          </td>
          <td class="col-view">
            {% if row.picks %}
              <button class="btn-view" onclick="toggleRow('{{ row.id }}')">View</button>
            {% endif %}
          </td>
        </tr>
        {% if row.picks %}
        <tr class="row-expanded" id="exp-{{ row.id }}" style="display:none">
          <td colspan="8">
            <div class="expanded-grid">
              {% for p in row.picks %}
                <div class="pick-detail {{ 'strong' if p.strong else '' }}" data-pick-id="{{ p.id }}">
                  <div class="pick-detail-head">
                    <strong>{{ p.pick }}</strong>
                    <span class="market-tag">{{ p.market }}</span>
                    {% if p.american is not none and p.decimal is not none %}
                      <span class="price">{{ ('+' if p.american > 0 else '') ~ p.american }} ({{ '%.2f'|format(p.decimal) }})</span>
                    {% else %}
                      <span class="price" style="color:var(--muted-2)">price &mdash; verify</span>
                    {% endif %}
                    <span class="book">@ {{ p.book }}</span>
                    {% if p.strong %}<span class="strong-badge">&#9733; STRONG</span>{% endif %}
                  </div>
                  <div class="pick-stats-detail">
                    <span><span class="label">Model</span><b>{{ (p.fair_prob * 100)|round|int }}%</b></span>
                    {% if p.pinnacle_prob %}<span><span class="label">Pinnacle</span><b>{{ (p.pinnacle_prob * 100)|round|int }}%</b></span>{% endif %}
                    {% if p.consensus_prob %}<span><span class="label">Consensus</span><b>{{ (p.consensus_prob * 100)|round|int }}%</b></span>{% endif %}
                    <span>
                      {% if p.ev_pct is none %}<span style="color:var(--muted-2)">EV &mdash;</span>
                      {% elif p.ev_pct > 0.1 %}<span class="ev">EV +{{ '%.1f'|format(p.ev_pct) }}%</span>
                      {% else %}<span style="color:var(--muted-2)">EV {{ '%+.1f'|format(p.ev_pct) }}%</span>{% endif %}
                    </span>
                    {% if p.kelly_pct and p.kelly_pct > 0 %}<span><span class="label">Stake (1/4 K)</span><b>{{ '%.1f'|format(p.kelly_pct) }}%</b></span>{% endif %}
                  </div>
                  <div class="pick-bulletin">{{ p.bulletin }}</div>
                  {% if p.polymarket_signal %}
                    <div class="poly-signal poly-{{ p.polymarket_signal.tone }}">{{ p.polymarket_signal.label }}</div>
                  {% endif %}
                  <button class="btn-ai" onclick="analyzePick('{{ p.id }}', this)">Expand with AI</button>
                  <span class="model-src">{{ p.model_source }}</span>
                  <div class="ai-analysis" id="ai-{{ p.id }}"></div>
                </div>
              {% endfor %}
            </div>
          </td>
        </tr>
        {% endif %}
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty-board">
    <h2>No games scheduled.</h2>
    <p>No tracked sport has a game on this date. Try another date, or check back later.</p>
  </div>
  {% endif %}

  <footer>
    <div>Picks blended via 40% model / 60% Pinnacle devig &middot; one row per game &middot; scan {{ scan_time_ms }} ms</div>
    <div>updated {{ now }}</div>
  </footer>
</main>

<script>
function toggleRow(rowId) {
  const el = document.getElementById('exp-' + rowId);
  if (!el) return;
  el.style.display = (el.style.display === 'none' || !el.style.display) ? 'table-row' : 'none';
}

async function analyzePick(pickId, btn) {
  const target = document.getElementById('ai-' + pickId);
  if (btn) { btn.disabled = true; btn.textContent = 'Thinking…'; }
  target.classList.add('visible');
  target.innerHTML = '<div class="loading">Claude is reviewing this pick…</div>';
  try {
    const res = await fetch('/picks/api/analyze?id=' + encodeURIComponent(pickId) + '&date={{ date_str|urlencode }}');
    const data = await res.json();
    target.innerHTML = renderAI(data);
  } catch (e) {
    target.innerHTML = '<div class="ai-err">Fetch error: ' + e + '</div>';
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Re-analyze'; }
  }
}
function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function renderAI(d) {
  if (d.error) {
    return '<div class="ai-err">' + esc(d.error) + '</div>'
      + (d.raw ? '<pre style="white-space:pre-wrap;color:var(--muted-2);font-size:11px">' + esc(d.raw) + '</pre>' : '');
  }
  if (d.narrative) {
    return '<div style="font-size:12.5px;line-height:1.55;white-space:pre-wrap;color:var(--ink)">'
         + esc(d.narrative) + '</div>'
         + (d.partial ? '<div style="color:var(--muted);font-size:10px;margin-top:8px;letter-spacing:0.08em;text-transform:uppercase">partial response — re-run for a full structured read</div>' : '');
  }
  const pick = d.pick || {};
  const stars = '★'.repeat(pick.confidence || 1) + '☆'.repeat(3 - (pick.confidence || 1));
  const sideLabel = (pick.side || '').replace(/^./, c => c.toUpperCase());
  return `
    <h5>Stat Read</h5>
    <ul>${(d.stat_read||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <h5>Matchup Factors</h5>
    <ul>${(d.matchup_factors||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <h5>Risk Flags</h5>
    <ul>${(d.risk_flags||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <div class="ai-pick">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted)">AI Pick · <span style="color:var(--good);text-shadow:var(--ev-strong-glow)">${stars}</span></div>
      <div style="font-weight:600;font-size:14px;color:var(--ink);margin-top:4px">${esc(pick.market || '?')} — ${esc(sideLabel)}</div>
      <div style="color:var(--muted);font-size:12.5px;margin-top:6px">${esc(pick.rationale || '')}</div>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted-2);margin-top:6px;letter-spacing:0.08em;text-transform:uppercase">${esc(d.model || 'claude')} · ${esc(d.generated_at || 'now')}</div>
  `;
}
</script>
</body>
</html>
"""


def _conf_tier(prob):
    """Map a 0..1 probability to a 1..5 confidence badge."""
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


def _pick_icon(market):
    """Return (glyph, css class) for a pick based on its market."""
    m = (market or "").lower()
    if "btts" in m or "both teams" in m: return ("B", "btts")
    if "double" in m or "dc" in m or "or draw" in m: return ("2x", "dc")
    if "spread" in m or "line" in m: return ("±", "spr")
    if "total" in m or "over" in m or "under" in m: return ("T", "tot")
    return ("ML", "ml")


def _snap_to_half(line):
    """Snap a spread/total line to a .5 multiple (0.5, 1.5, 2.5, 3.5, ...)
    to match how Polymarket / DK / FD display lines. Pinnacle serves Asian
    quarter-point splits (-0.25, -0.75, ...) and integer "key number" lines
    (-3, -7) that aren't shown on half-point books, so we snap for display
    consistency.

    Policy: for quarter-point lines (.25 / .75), bias DOWN toward zero so
    both halves of the Asian split pick the easier side (home -0.75 → -0.5,
    "home wins outright" at Polymarket), matching how Polymarket labels its
    low-end handicap. Integer lines bump UP to the next half-point so a
    push-prone -3 becomes -3.5.
    """
    if line is None:
        return None
    try:
        v = float(line)
    except (TypeError, ValueError):
        return None
    sign = -1.0 if v < 0 else 1.0
    av = abs(v)
    frac = av - int(av)
    if abs(frac - 0.5) < 1e-9:
        snapped = av                      # already .5
    elif frac < 0.5:
        snapped = int(av) + 0.5           # 0.25 → 0.5, 2.25 → 2.5
    else:
        snapped = int(av) + 0.5           # 0.75 → 0.5, 2.75 → 2.5 (bias toward .5)
    if frac == 0.0 and av > 0:
        snapped = int(av) + 0.5           # integer → next half (push protection)
    if av == 0:
        snapped = 0.5
    return sign * snapped


def _pick_short_label(p):
    """One-line label for the pick badge. Keep it compact."""
    return p.get("pick") or "—"


def _derive_status(start_iso, now_dt, has_score):
    """upcoming | live | final — rough heuristic based on time + whether we
    already graded a score for the game."""
    if has_score:
        return "final"
    if not start_iso:
        return "upcoming"
    try:
        st = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except Exception:
        return "upcoming"
    if st > now_dt:
        return "upcoming"
    # Game started. Heuristic for "live": within an outer window of 5 hours
    # (NFL/soccer can run long, NHL/MLB usually finish in 3). After that we
    # call it final even without a score.
    from datetime import timedelta as _td
    if st + _td(hours=5) > now_dt:
        return "live"
    return "final"


def _countdown_str(start_iso, now_dt):
    """Hh Mm Ss countdown — only when the game hasn't started yet."""
    if not start_iso:
        return None
    try:
        st = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    delta = st - now_dt
    secs = int(delta.total_seconds())
    if secs <= 0:
        return None
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h >= 24:
        d = h // 24
        return f"{d}d {h % 24}h"
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def _format_ml_line(ml, ml_outcomes):
    """Compact ML line: 'TB +118 / NYY -140' (or draw-inclusive for 3-way)."""
    if not ml or ml.get("home_am") is None:
        return None
    parts = [f"{_signed_am_str(ml['away_am'])}"]
    if ml.get("draw_am") is not None and ml_outcomes == 3:
        parts.append(f"D {_signed_am_str(ml['draw_am'])}")
    parts.append(f"{_signed_am_str(ml['home_am'])}")
    return " / ".join(parts)


def _format_spread_line(spread, away, home):
    """Compact spread: 'KC -3.5 (-110)' — snapped to half-point."""
    if not spread or spread.get("line_home") is None:
        return None
    line_h = _snap_to_half(spread["line_home"])
    if line_h is None:
        return None
    if line_h < 0:
        label = f"{home.split()[-1] if home else 'HM'} {line_h:g}"
        am = spread.get("home_am")
    else:
        label = f"{away.split()[-1] if away else 'AW'} {-line_h:g}"
        am = spread.get("away_am")
    if am is not None:
        return f"{label} ({_signed_am_str(am)})"
    return label


def _format_total_line(total):
    """Compact total: 'O/U 47.5' — snapped to half-point."""
    if not total or total.get("line") is None:
        return None
    line = _snap_to_half(total["line"])
    if line is None:
        return None
    return f"O/U {line:g}"


_PICK_MARKET_ORDER = {"ML": 0, "DC": 1, "Spread": 2, "Total": 3, "BTTS": 4}


def _market_category(market):
    """Reduce a full market string ('1H Spread', 'Total (BTTS)') to its base
    category — 'ML', 'DC', 'Spread', 'Total', 'BTTS', or raw if no match."""
    m = (market or "").upper()
    if "BTTS" in m or "BOTH TEAMS" in m: return "BTTS"
    if "DOUBLE CHANCE" in m or " OR DRAW" in m or m == "DC": return "DC"
    if "SPREAD" in m or "RUNLINE" in m or "RL" in m or "PUCK LINE" in m: return "Spread"
    if "TOTAL" in m or "O/U" in m or "OVER" in m or "UNDER" in m: return "Total"
    if "ML" in m or "MONEYLINE" in m: return "ML"
    return (market or "").title()


def _model_fill_picks(sport_slug, home, away, pin_ml, pin_spread, pin_total,
                       ml_outcomes, existing_cats):
    """Produce up to 3 model-opinion picks per game (ML + Spread + Total),
    skipping markets we already have an EV pick for. These are purely
    'which side does our model prefer' — no EV / price filtering, matched
    to bettingtools.ai's "always 3 picks per game" style.

    Memory-cheap on Render free tier: we only reuse a sim that's ALREADY
    in picks._SIM_CACHE (populated by the EV pass that just ran); we do
    NOT trigger new 1500-trial Monte Carlo sims here. Games without a
    cached sim fall back to Pinnacle devig for direction + confidence.

    Returns a list of pick dicts (not saved to plays_log — tracking stays
    gated to the EV-filtered picks that have genuine edge).
    """
    import picks as picks_mod

    total_line = (pin_total or {}).get("line")
    # Prefer a sim the EV pass already paid for. For soccer we also allow a
    # cold sim: Dixon-Coles is light enough to run per-game without blowing
    # the Render dyno, and we specifically need its draw_pct + btts_yes_pct
    # for the Double-Chance and BTTS picks. For NHL / NFL / NCAAF we stay
    # cache-only — those sims are the memory-expensive ones.
    sim_cache = getattr(picks_mod, "_SIM_CACHE", {}) or {}
    sim_key = (sport_slug, home or "", away or "", total_line)
    sim = sim_cache.get(sim_key)
    if sim is None and sport_slug in ("epl", "laliga", "ligamx", "ucl",
                                      "europa", "international"):
        try:
            sim = picks_mod._sim_for(sport_slug, home, away,
                                     market_total=total_line)
        except Exception:
            sim = None
    out = []

    # ---- ML ----
    if "ML" not in existing_cats:
        h_pct = a_pct = d_pct = 0.0
        if sim and sim.get("home_win_pct") is not None:
            h_pct = (sim.get("home_win_pct") or 0) / 100.0
            a_pct = (sim.get("away_win_pct") or 0) / 100.0
            d_pct = (sim.get("draw_pct") or 0) / 100.0
        elif pin_ml and pin_ml.get("home_am") is not None and pin_ml.get("away_am") is not None:
            # Fallback: Pinnacle devig (not our model, but still a sharp signal)
            p_h = generic_odds.american_to_prob(pin_ml["home_am"])
            p_a = generic_odds.american_to_prob(pin_ml["away_am"])
            d_am = pin_ml.get("draw_am")
            if d_am is not None and ml_outcomes == 3:
                p_d = generic_odds.american_to_prob(d_am)
                h_pct, d_pct, a_pct = generic_odds.devig_three_way(p_h, p_d, p_a)
            else:
                h_pct, a_pct = generic_odds.devig_two_sided(p_h, p_a)
        if max(h_pct, a_pct, d_pct) > 0:
            if h_pct >= max(a_pct, d_pct):
                out.append({"category": "ML", "market": "ML",
                            "pick": home, "fair_prob": h_pct})
            elif a_pct >= d_pct:
                out.append({"category": "ML", "market": "ML",
                            "pick": away, "fair_prob": a_pct})
            else:
                out.append({"category": "ML", "market": "ML",
                            "pick": "Draw", "fair_prob": d_pct})

    # ---- Spread (snapped to half-point to match Polymarket / DK / FD) ----
    line_h = None
    if "Spread" not in existing_cats and pin_spread and pin_spread.get("line_home") is not None:
        line_h = _snap_to_half(pin_spread["line_home"])
    if line_h is not None:
        margins_arr = (sim or {}).get("margins") or []
        numeric_margins = [float(m) for m in margins_arr if isinstance(m, (int, float))] \
            if isinstance(margins_arr, list) else []
        p_home_cover = None
        if numeric_margins:
            # Re-evaluate cover prob at the SNAPPED line (not Pinnacle's raw)
            p_home_cover = sum(1 for m in numeric_margins if m > -line_h) / len(numeric_margins)
        elif pin_spread.get("home_am") is not None and pin_spread.get("away_am") is not None:
            p_h = generic_odds.american_to_prob(pin_spread["home_am"])
            p_a = generic_odds.american_to_prob(pin_spread["away_am"])
            p_home_cover, _ = generic_odds.devig_two_sided(p_h, p_a)
        if p_home_cover is not None:
            if p_home_cover >= 0.5:
                label = f"{home} {line_h:+g}"
                prob = p_home_cover
            else:
                label = f"{away} {-line_h:+g}"
                prob = 1 - p_home_cover
            out.append({"category": "Spread", "market": "Spread",
                        "pick": label, "fair_prob": prob})

    # ---- Total (snapped to half-point to match Polymarket / DK / FD) ----
    if "Total" not in existing_cats and pin_total and pin_total.get("line") is not None:
        line = _snap_to_half(pin_total["line"])
        if line is not None:
            # Only trust `totals` when it's a plain numeric list (NHL path);
            # soccer returns {market_key: pct} which isn't iterable the same way.
            totals_arr = (sim or {}).get("totals") or []
            numeric_totals = []
            if isinstance(totals_arr, list):
                for t in totals_arr:
                    if isinstance(t, (int, float)):
                        numeric_totals.append(float(t))
            p_over = None
            if numeric_totals:
                p_over = sum(1 for t in numeric_totals if t > line) / len(numeric_totals)
            elif sim and isinstance(sim.get("expected_total"), (int, float)):
                # No distribution — directional only, flat ~55% confidence
                p_over = 0.55 if float(sim["expected_total"]) >= line else 0.45
            elif pin_total.get("over_am") is not None and pin_total.get("under_am") is not None:
                p_o = generic_odds.american_to_prob(pin_total["over_am"])
                p_u = generic_odds.american_to_prob(pin_total["under_am"])
                p_over, _ = generic_odds.devig_two_sided(p_o, p_u)
            if p_over is not None:
                if p_over >= 0.5:
                    label = f"Over {line:g}"
                    prob = p_over
                else:
                    label = f"Under {line:g}"
                    prob = 1 - p_over
                out.append({"category": "Total", "market": "Total",
                            "pick": label, "fair_prob": prob})

    # ---- Soccer-only extras: Double Chance (ML No) + Both Teams To Score ----
    is_soccer = sport_slug in ("epl", "laliga", "ligamx", "ucl", "europa", "international")
    if is_soccer and sim:
        # Double Chance: safer ML variant that also wins on a draw. Only
        # surface it when the straight ML favorite's draw-risk is material
        # (draw_pct >= 20%) AND the DC prob is meaningfully higher than the
        # straight ML (+8pp), otherwise it's just a weaker-priced duplicate.
        h_pct = (sim.get("home_win_pct") or 0) / 100.0
        a_pct = (sim.get("away_win_pct") or 0) / 100.0
        d_pct = (sim.get("draw_pct") or 0) / 100.0
        if "DC" not in existing_cats and d_pct >= 0.20:
            if h_pct >= a_pct:
                dc_label  = f"{home} or Draw (1X)"
                dc_prob   = h_pct + d_pct
                ml_prob   = h_pct
            else:
                dc_label  = f"{away} or Draw (X2)"
                dc_prob   = a_pct + d_pct
                ml_prob   = a_pct
            if dc_prob - ml_prob >= 0.08:
                out.append({"category": "DC", "market": "Double Chance",
                            "pick": dc_label, "fair_prob": dc_prob})

        # Both Teams To Score — pulled straight from the Dixon-Coles sim's
        # btts_yes_pct, which it computes from the joint-goal pmf.
        if "BTTS" not in existing_cats and sim.get("btts_yes_pct") is not None:
            byes = float(sim["btts_yes_pct"]) / 100.0
            if byes >= 0.5:
                out.append({"category": "BTTS", "market": "BTTS",
                            "pick": "BTTS: Yes", "fair_prob": byes})
            else:
                out.append({"category": "BTTS", "market": "BTTS",
                            "pick": "BTTS: No", "fair_prob": 1 - byes})

    return out


def _build_board_rows(date_str, today_date, sport_filter):
    """Return the unified board: one row per game with picks nested.

    `date_str` is YYYY-MM-DD being viewed. `today_date` is the CT reference
    for countdown/live-status computation (always real now())."""
    import picks as picks_mod
    import plays_log

    view_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    now_dt = datetime.now(CENTRAL)

    # 1) Today's scheduled games across every sport (schedule skeleton)
    rows_by_key = {}
    for sp in sports.SPORTS:
        if sport_filter not in ("all", sp["slug"]):
            continue
        try:
            games = _sport_games_for_hub(sp, view_date)
        except Exception:
            games = []
        for g in games:
            key = (sp["slug"],
                   generic_odds._team_key(g["away"]),
                   generic_odds._team_key(g["home"]))
            rows_by_key[key] = {
                "id":          f"{sp['slug']}-{generic_odds._team_key(g['away'])}-{generic_odds._team_key(g['home'])}",
                "sport_slug":  sp["slug"],
                "sport_name":  sp["name"],
                "ml_outcomes": sp["ml_outcomes"],
                "away":        g["away"],
                "home":        g["home"],
                "start_time":  g["start_time"],
                "_ml":         g.get("ml"),
                "_spread":     g.get("spread"),
                "_total":      g.get("total"),
                "picks":       [],
                "score_home":  None,
                "score_away":  None,
            }

    # 2) Today's picks across every sport
    try:
        all_picks = picks_mod.collect_picks(date_str)
    except Exception:
        all_picks = []

    # 3) Grading overlay (scores + CLV + W/L on picks graded today)
    graded_by_id = {}
    try:
        graded = plays_log.read_graded(date_str) or {}
        for gp in graded.get("picks", []):
            if gp.get("id"):
                graded_by_id[gp["id"]] = gp
    except Exception:
        pass

    # 4) Attach each pick to its game row, synthesizing a row when the pick's
    #    game isn't in the schedule skeleton (keeps us from dropping picks).
    for p in all_picks:
        if sport_filter not in ("all", p.get("sport_slug", "")):
            continue
        key = (p.get("sport_slug", ""),
               generic_odds._team_key(p.get("away_team", "")),
               generic_odds._team_key(p.get("home_team", "")))
        row = rows_by_key.get(key)
        if row is None:
            # Build a synthetic row from the pick's own metadata.
            row = {
                "id":          f"syn-{p.get('id', '')}",
                "sport_slug":  p.get("sport_slug", ""),
                "sport_name":  p.get("sport", ""),
                "ml_outcomes": 2,
                "away":        p.get("away_team", ""),
                "home":        p.get("home_team", ""),
                "start_time":  p.get("start_time"),
                "_ml": None, "_spread": None, "_total": None,
                "picks": [], "score_home": None, "score_away": None,
            }
            rows_by_key[key] = row

        # Pick badge data
        icon, icon_cls = _pick_icon(p.get("market"))
        row["picks"].append({
            **p,
            "icon":          icon,
            "icon_cls":      icon_cls,
            "pick_short":    _pick_short_label(p),
            "conf":          _conf_tier(p.get("consensus_prob") or p.get("fair_prob")),
            "result":        (graded_by_id.get(p.get("id"), {}) or {}).get("result"),
        })
        # Overlay score if this pick graded with a final score
        gp = graded_by_id.get(p.get("id"))
        if gp and gp.get("home_score") is not None and row["score_home"] is None:
            row["score_home"] = gp.get("home_score")
            row["score_away"] = gp.get("away_score")

    # 5) Fill in missing markets with pure-model picks so every game shows the
    #    full ML + Spread + Total triplet (bettingtools.ai-style). These are
    #    opinion-only — not EV-filtered and not persisted to plays_log.
    for row in rows_by_key.values():
        existing_cats = {_market_category(p.get("market")) for p in row["picks"]}
        fill = _model_fill_picks(
            row["sport_slug"], row["home"], row["away"],
            row["_ml"], row["_spread"], row["_total"],
            row["ml_outcomes"], existing_cats,
        )
        for mp in fill:
            icon, icon_cls = _pick_icon(mp["market"])
            key = generic_odds._team_key(row["home"]) + "-" + generic_odds._team_key(row["away"])
            row["picks"].append({
                "id":            f"model-{row['sport_slug']}-{key}-{mp['category'].lower()}",
                "sport":         row["sport_name"],
                "sport_slug":    row["sport_slug"],
                "market":        mp["market"],
                "pick":          mp["pick"],
                "fair_prob":     mp["fair_prob"],
                "consensus_prob": mp["fair_prob"],  # same for model-only (no market blend)
                "pinnacle_prob": None,
                "ev_pct":        None,
                "decimal":       None,
                "american":      None,
                "kelly_pct":     None,
                "book":          None,
                "strong":        False,
                "bulletin":      "Model-only pick — our simulator's preferred side at the posted line, no price filter applied.",
                "model_source":  "model only (no EV filter)",
                "model_only":    True,
                "home_team":     row["home"],
                "away_team":     row["away"],
                "icon":          icon,
                "icon_cls":      icon_cls,
                "pick_short":    mp["pick"],
                "conf":          _conf_tier(mp["fair_prob"]),
                "result":        None,
            })

        # Order picks within a game: ML → Spread → Total → anything else,
        # EV picks before model-only picks within the same category.
        row["picks"].sort(key=lambda p: (
            _PICK_MARKET_ORDER.get(_market_category(p.get("market")), 9),
            1 if p.get("model_only") else 0,
        ))

    # 6) Finalize each row — status, countdown, line/close text, sort key.
    rows = []
    for row in rows_by_key.values():
        has_score = row["score_home"] is not None
        row["status"]    = _derive_status(row["start_time"], now_dt, has_score)
        row["countdown"] = _countdown_str(row["start_time"], now_dt) if row["status"] == "upcoming" else None
        try:
            dt_ct = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00")).astimezone(CENTRAL)
            row["time_abs"] = dt_ct.strftime("%a %I:%M %p CT").replace(" 0", " ")
        except Exception:
            row["time_abs"] = row["start_time"] or ""
        # Pack line entries
        lines = []
        if row["_ml"]:
            v = _format_ml_line(row["_ml"], row["ml_outcomes"])
            if v: lines.append({"k": "ML", "v": v})
        if row["_spread"]:
            v = _format_spread_line(row["_spread"], row["away"], row["home"])
            if v: lines.append({"k": "SPR", "v": v})
        if row["_total"]:
            v = _format_total_line(row["_total"])
            if v: lines.append({"k": "TOT", "v": v})
        row["lines"]       = lines
        row["close_lines"] = []  # populated later from CLV snapshot
        rows.append(row)

    rows.sort(key=lambda r: r.get("start_time") or "")
    return rows, all_picks


@app.route("/")
def picks_landing():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    sport_filter = (request.args.get("sport") or "all").lower()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()

    t0 = time.time()
    try:
        board_rows, all_picks = _build_board_rows(date_str, d, sport_filter)
    except Exception:
        board_rows, all_picks = [], []
    scan_time_ms = int((time.time() - t0) * 1000)

    # Snapshot the picks for /logged (and GitHub mirror for Render spin-downs)
    try:
        import plays_log
        if all_picks:
            plays_log.save_daily_picks(date_str, all_picks)
    except Exception:
        pass

    # Analyzer-driven adjustments overlay
    try:
        import model_analytics
        active_adj = model_analytics.active_adjustments_summary()
    except Exception:
        active_adj = {"any": False}

    # Per-sport counts (over the whole slate, not just the filter)
    from collections import Counter
    all_games_counter = Counter(r["sport_slug"] for r in board_rows)
    all_name_by_slug  = {r["sport_slug"]: r["sport_name"] for r in board_rows}
    sport_counts = sorted(
        [{"slug": s, "name": all_name_by_slug[s], "count": all_games_counter[s]}
         for s in all_games_counter],
        key=lambda x: -x["count"],
    )

    picks_count   = sum(len(r["picks"]) for r in board_rows)
    ev_count      = sum(1 for r in board_rows for p in r["picks"] if not p.get("model_only"))
    strong_count  = sum(1 for r in board_rows for p in r["picks"] if p.get("strong"))
    total_games   = len(board_rows)
    sports_with_games = len({r["sport_slug"] for r in board_rows})
    today = datetime.now(CENTRAL).date().isoformat()

    return render_template_string(
        PICKS_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("picks"),
        board_rows=board_rows,
        sport_counts=sport_counts,
        sport_filter=sport_filter,
        picks_count=picks_count,
        ev_count=ev_count,
        strong_count=strong_count,
        total_games=total_games,
        sports_with_games=sports_with_games,
        scan_time_ms=scan_time_ms,
        date_str=date_str,
        date_pretty=d.strftime("%A, %B %d").replace(" 0", " "),
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
        active_adj=active_adj,
    )


@app.route("/picks/api/analyze")
def picks_api_analyze():
    """AI analysis for a single pick, by pick id."""
    import picks as picks_mod
    import analyst as analyst_mod
    pick_id = request.args.get("id") or ""
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        all_picks = picks_mod.collect_picks(date_str)
    except Exception:
        all_picks = []
    pick = next((p for p in all_picks if p["id"] == pick_id), None)
    if not pick:
        return Response(json.dumps({"error": "pick not found for this date"}),
                        mimetype="application/json")
    try:
        result = analyst_mod.analyze_pick(pick)
    except Exception as e:
        result = {"error": f"analyst error: {e}"}
    return Response(json.dumps(result), mimetype="application/json")


@app.route("/mlb/schedule")
@app.route("/mlb")
def index():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()

    error = None
    games = []
    try:
        games = get_games(date_str)
    except URLError as e:
        error = f"Could not reach MLB statsapi ({e.reason})."
    except Exception as e:
        error = f"Unexpected error: {e}"

    today = datetime.now(CENTRAL).date().isoformat()
    date_pretty = d.strftime("%A, %B %d").replace(" 0", " ")

    return render_template_string(
        INDEX_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        games=games,
        date_str=date_str,
        date_pretty=date_pretty,
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        error=error,
        game_count=len(games),
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


@app.route("/backtest")
def backtest():
    try:
        if request.args.get("refresh"):
            state = mlb_model.get_or_run_multi_season_backtest(refresh=True)
        else:
            state = mlb_model.get_or_run_multi_season_backtest()
    except Exception:
        state = None

    if not state:
        return render_template_string(
            BACKTEST_TEMPLATE,
            fonts_link=FONTS_LINK,
            shared_style=SHARED_STYLE,
            sport_strip=render_sport_strip("backtest"),
            state=None, m=None,
            top_elo=[], bottom_elo=[],
            per_season_rows=[],
            sample_preds=[],
            calibration_svg="",
        )

    m = state.get("metrics") or mlb_model.compute_metrics(state["predictions"])

    # Compute last-season W-L from predictions for the Final Elo display
    last_season = state["last_season"]
    final_wl = {}
    for p in state["predictions"]:
        if p["season"] != last_season:
            continue
        h, a = str(p["home_id"]), str(p["away_id"])
        final_wl.setdefault(h, [0, 0])
        final_wl.setdefault(a, [0, 0])
        if p["home_won"]:
            final_wl[h][0] += 1
            final_wl[a][1] += 1
        else:
            final_wl[h][1] += 1
            final_wl[a][0] += 1

    elo_items = sorted(state["final_elo"].items(), key=lambda kv: -kv[1])
    top_elo = [(tid, r, final_wl.get(tid, [0, 0])) for tid, r in elo_items[:15]]
    bottom_elo = [(tid, r, final_wl.get(tid, [0, 0])) for tid, r in elo_items[15:]]

    per_season_rows = []
    for s in sorted(state["per_season"]):
        st = state["per_season"][s]
        if st.get("n"):
            per_season_rows.append({
                "season": s, "n": st["n"],
                "accuracy": st["accuracy"],
                "log_loss": st["log_loss"],
                "brier": st["brier"],
            })

    sample_preds = state["predictions"][-30:]

    return render_template_string(
        BACKTEST_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("backtest"),
        state=state,
        m=m,
        top_elo=top_elo,
        bottom_elo=bottom_elo,
        per_season_rows=per_season_rows,
        sample_preds=sample_preds,
        calibration_svg=render_calibration_svg(m.get("calibration") or []),
    )


@app.route("/export.csv")
def export_csv():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        date_str = datetime.now(CENTRAL).date().isoformat()

    games = get_games(date_str)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "first_pitch_et", "status", "series_game", "venue",
        "wx_temp_f", "wx_condition", "wx_wind", "tv",
        "away_team", "away_record", "away_streak", "away_run_diff",
        "away_avg", "away_obp", "away_slg", "away_ops", "away_hr", "away_team_era",
        "home_team", "home_record", "home_streak", "home_run_diff",
        "home_avg", "home_obp", "home_slg", "home_ops", "home_hr", "home_team_era",
        "away_pitcher", "away_pitcher_hand", "away_pitcher_wl", "away_pitcher_era",
        "away_pitcher_whip", "away_pitcher_k9", "away_pitcher_bb9", "away_pitcher_ip",
        "home_pitcher", "home_pitcher_hand", "home_pitcher_wl", "home_pitcher_era",
        "home_pitcher_whip", "home_pitcher_k9", "home_pitcher_bb9", "home_pitcher_ip",
        "model_p_home", "model_favors",
    ])
    for g in games:
        a, h = g["away"], g["home"]
        ap, hp = g["away_pitcher"], g["home_pitcher"]
        series = f"{g['series_num']}/{g['games_in_series']}" if g.get("series_num") and g.get("games_in_series") else ""
        writer.writerow([
            g["first_pitch"], g["status"], series, g["venue"],
            g["wx_temp"], g["wx_condition"], g["wx_wind"], g["tv"],
            a["team_full"],
            f"{a['wins']}-{a['losses']}" if a["wins"] is not None else "",
            a["streak_code"], a["run_diff"],
            a["avg"], a["obp"], a["slg"], a["ops"], a["hr"], a["team_era"],
            h["team_full"],
            f"{h['wins']}-{h['losses']}" if h["wins"] is not None else "",
            h["streak_code"], h["run_diff"],
            h["avg"], h["obp"], h["slg"], h["ops"], h["hr"], h["team_era"],
            ap["name"], ap["hand"], ap["wl"], ap["era"],
            ap["whip"], ap["k9"], ap["bb9"], ap["ip"],
            hp["name"], hp["hand"], hp["wl"], hp["era"],
            hp["whip"], hp["k9"], hp["bb9"], hp["ip"],
            round(g["p_home"], 4) if g.get("p_home") is not None else "",
            "home" if g.get("favors_home") else ("away" if g.get("p_home") is not None else ""),
        ])

    filename = f"mlb_tonight_{date_str}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# /edges — recommended bets with positive expected value
# ============================================================================

EDGES_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Edges &mdash; {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero {
  padding-bottom: 20px; margin-bottom: 32px;
  border-bottom: 1px solid var(--rule);
}
.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero .sub {
  color: var(--muted); max-width: 760px; font-size: 14px; line-height: 1.6;
}

.summary {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 24px;
}
.summary .metric {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.summary .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.summary .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 24px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.summary .value.good { color: var(--good); }

.edges {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; overflow: hidden;
}
.edges table {
  width: 100%; border-collapse: collapse;
  font-family: "Public Sans", sans-serif; font-size: 13.5px;
}
.edges th {
  text-align: left; padding: 12px 14px;
  background: var(--surface); color: var(--muted);
  font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
  font-weight: 500; border-bottom: 1px solid var(--rule);
}
.edges td {
  padding: 10px 14px; border-bottom: 1px solid var(--rule);
  vertical-align: middle;
}
.edges tr:last-child td { border-bottom: none; }
.edges tr:hover { background: color-mix(in oklab, var(--rule) 30%, transparent); }
.edges .num {
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.edges .matchup { font-weight: 500; color: var(--ink); }
.edges .pick { font-weight: 600; }
.edges .ev-strong { color: var(--good); font-weight: 600; }
.edges .ev-mild { color: var(--good); }

.filter-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 16px; }
.filter-chips .chip {
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--rule-strong);
  background: var(--surface); color: var(--muted);
  font-size: 12px; font-weight: 500; text-decoration: none;
  transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
}
.filter-chips .chip:hover { color: var(--ink); background: var(--card); }
.filter-chips .chip.active {
  color: var(--ink); background: var(--card);
  border-color: var(--ink);
}

.mkt-pill {
  display: inline-block;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; font-weight: 500;
  padding: 2px 8px; border-radius: 4px;
  background: var(--chip-bg); color: var(--muted);
  border: 1px solid var(--rule);
}
.mkt-pill.mkt-ml       { color: var(--accent); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); }
.mkt-pill.mkt-total    { color: var(--warn);   border-color: color-mix(in oklab, var(--warn) 40%, var(--rule)); }
.mkt-pill.mkt-run_line { color: var(--good);   border-color: color-mix(in oklab, var(--good) 40%, var(--rule)); }
.mkt-pill.mkt-f5_ml    { color: var(--accent); background: color-mix(in oklab, var(--accent) 8%, var(--chip-bg)); }
.mkt-pill.mkt-f5_total { color: var(--warn);   background: color-mix(in oklab, var(--warn) 8%, var(--chip-bg)); }

.empty-edge {
  padding: 60px 24px; text-align: center; color: var(--muted);
}
.empty-edge h2 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: 24px; color: var(--ink); margin: 0 0 8px;
}

.note {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 18px;
  color: var(--muted); font-size: 13px; line-height: 1.6;
  margin-top: 20px; max-width: 900px;
}
.note strong { color: var(--ink); }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab" href="/analyst">AI Analyst</a>
        <a class="nav-tab" href="/market">Market</a>
        <a class="nav-tab" href="/backtest">Model</a>
      </nav>
    </div>
    <form class="controls" method="get" action="/edges">
      <a class="btn icon" href="/edges?date={{ prev_date }}" title="Previous day">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()">
      <a class="btn icon" href="/edges?date={{ next_date }}" title="Next day">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/edges?date={{ today }}">Today</a>{% endif %}
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>Positive-EV plays &middot; {{ date_pretty }}</h1>
    <p class="sub">
      Bets where the model's win probability exceeds the bettable (vig-included) Pinnacle price.
      EV = (model_prob &middot; decimal_odds) &minus; 1. Suggested stake is quarter-Kelly, floored at 0.
      This is a tool for identifying candidates, not a guarantee &mdash; the model is ~58% accurate and sample sizes per day are small.
    </p>
  </div>

  <div class="summary">
    <div class="metric">
      <div class="label">Games with odds</div>
      <div class="value">{{ games_with_odds }}</div>
    </div>
    <div class="metric">
      <div class="label">Positive EV picks</div>
      <div class="value {% if positive_count %}good{% endif %}">{{ positive_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Strong edges (&ge;2%)</div>
      <div class="value {% if strong_count %}good{% endif %}">{{ strong_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Odds API books</div>
      <div class="value" style="font-size:16px;line-height:1.3">{{ 'DK/FD/BMG/C' if odds_api_available else 'Pinnacle only' }}</div>
    </div>
  </div>

  <div class="filter-chips">
    {% set mkts = [('ALL','All'),('ML','Moneyline'),('TOTAL','Total'),('RUN_LINE','Run Line'),('F5_ML','F5 ML'),('F5_TOTAL','F5 Total')] %}
    {% for mk, lab in mkts %}
      <a class="chip {% if market_filter == mk %}active{% endif %}" href="/edges?date={{ date_str }}&market={{ mk|lower }}">{{ lab }}</a>
    {% endfor %}
  </div>

  <div class="edges">
    {% if edges %}
    <table>
      <thead>
        <tr>
          <th>Market</th>
          <th>Game</th>
          <th>Pick</th>
          <th class="num">Model</th>
          <th class="num">Fair</th>
          <th class="num">Pin price</th>
          <th class="num">EV</th>
          <th class="num">Stake (1/4 K)</th>
          {% if odds_api_available %}<th>Best US book</th>{% endif %}
        </tr>
      </thead>
      <tbody>
        {% for e in edges %}
        <tr>
          <td><span class="mkt-pill mkt-{{ e.market|replace(' ','_')|lower }}">{{ e.market }}</span></td>
          <td class="matchup">{{ e.away }} at {{ e.home }}<br>
            <small style="color:var(--muted);font-size:11px">{{ e.first_pitch }}</small></td>
          <td class="pick">{{ e.pick }}</td>
          <td class="num">{{ (e.model_prob * 100)|round(1) }}%</td>
          <td class="num" style="color:var(--muted)">{{ (e.fair_prob * 100)|round(1) if e.fair_prob else '—' }}{{ '%' if e.fair_prob else '' }}</td>
          <td class="num">
            {{ ('+' if e.american > 0 else '') ~ e.american }}
            <span style="color:var(--muted-2)">({{ '%.2f'|format(e.decimal) }})</span>
          </td>
          <td class="num {% if e.ev_pct >= 2 %}ev-strong{% else %}ev-mild{% endif %}">
            +{{ '%.1f'|format(e.ev_pct) }}%
          </td>
          <td class="num">{{ '%.1f'|format(e.kelly_pct) }}%</td>
          {% if odds_api_available %}
          <td>
            {% if e.best_book %}
              {{ e.best_book }} {{ '%.2f'|format(e.best_decimal) }}
              {% if e.ev_book_pct and e.ev_book_pct > e.ev_pct %}
              <span style="color:var(--good);font-size:11px">&nbsp;(EV +{{ '%.1f'|format(e.ev_book_pct) }}%)</span>
              {% endif %}
            {% else %}&mdash;{% endif %}
          </td>
          {% endif %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty-edge">
      <h2>No positive-EV plays {% if market_filter != 'ALL' %}in this market{% endif %} today.</h2>
      <p>Either the market agrees with the model, or Pinnacle hasn't posted this slate yet.</p>
    </div>
    {% endif %}
  </div>

  <div class="note">
    <strong>Markets covered:</strong> Full moneyline (Elo+SP model), Full total (Poisson, team RPG × opp SP+bullpen rate), Run Line -1.5 (Poisson margin), F5 moneyline + F5 total (Poisson with SP-heavy weighting). F5 ML has a tie-push rule baked into EV.
    <br><br>
    <strong>Betting splits (public bet %):</strong> Still not scraped — ScoresAndOdds/VegasInsider are client-rendered. The Market tab shows Pinnacle's limits per market as a sharper-money proxy.
  </div>

  <footer>
    <div>Pinnacle via guest API &middot; vig-included prices &middot; EV = model_prob &middot; dec - 1</div>
    <div>updated {{ now }}</div>
  </footer>
</main>
</body>
</html>
"""


@app.route("/edges")
def edges():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    market_filter = (request.args.get("market") or "all").upper()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()

    try:
        games = get_games(date_str)
    except Exception:
        games = []

    today = datetime.now(CENTRAL).date().isoformat()
    edges_list = []
    games_with_odds = 0
    for g in games:
        o = g.get("odds") or {}
        pin = o.get("pinnacle") or {}
        if not pin:
            continue
        games_with_odds += 1
        bets = o.get("bets") or []
        for b in bets:
            if b["ev_pct"] <= 0:
                continue
            if market_filter not in ("ALL", b["market"].upper().replace(" ", "_")):
                # allow filter like market=ml, market=total, market=run_line, market=f5_ml, market=f5_total
                mkt_key = b["market"].upper().replace(" ", "_")
                if market_filter != mkt_key:
                    continue
            # best US book price only applies to full-game ML + spreads + totals
            best_book = best_dec = ev_book = None
            if b["market"] in ("ML", "Run Line", "Total"):
                best_book = (o.get("best_book") or {}).get(b["side"])
                best_dec = (o.get("best_decimal") or {}).get(b["side"])
                if b["market"] == "ML":
                    ev_book = (o.get("ev_book") or {}).get(b["side"])
                elif best_dec and b.get("model_prob") is not None:
                    ev_book = (b["model_prob"] * best_dec - (1 - b.get("push_prob", 0))) * 100
            edges_list.append({
                "market": b["market"],
                "away": g["away"]["team"],
                "home": g["home"]["team"],
                "first_pitch": g["first_pitch"],
                "pick": b["pick"],
                "model_prob": b["model_prob"],
                "fair_prob": b["fair_prob"],
                "american": b["american"],
                "decimal": b["decimal"],
                "ev_pct": b["ev_pct"],
                "kelly_pct": b["kelly_pct"],
                "push_prob": b.get("push_prob", 0),
                "limit": b.get("limit"),
                "best_book": best_book,
                "best_decimal": best_dec,
                "ev_book_pct": ev_book,
            })

    edges_list.sort(key=lambda e: -e["ev_pct"])
    strong_count = sum(1 for e in edges_list if e["ev_pct"] >= 2.0)

    return render_template_string(
        EDGES_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        date_str=date_str,
        date_pretty=d.strftime("%A, %B %d").replace(" 0", " "),
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        edges=edges_list,
        games_with_odds=games_with_odds,
        positive_count=len(edges_list),
        strong_count=strong_count,
        odds_api_available=mlb_odds.odds_api_available(),
        market_filter=market_filter,
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


# ============================================================================
# /market — Pinnacle limits + Polymarket futures + splits context
# ============================================================================

MARKET_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Market &mdash; {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding-bottom: 20px; margin-bottom: 32px; border-bottom: 1px solid var(--rule); }
.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero .sub { color: var(--muted); max-width: 760px; font-size: 14px; line-height: 1.6; }

.section {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
}
.section h2 {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 500; font-size: 20px; margin: 0 0 6px;
  letter-spacing: -0.01em;
}
.section .lead { color: var(--muted); margin: 0 0 18px; font-size: 13px; line-height: 1.6; max-width: 760px; }

.limits-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.limits-table th, .limits-table td {
  padding: 10px 12px; border-bottom: 1px solid var(--rule); text-align: left;
}
.limits-table th {
  color: var(--muted); font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; font-weight: 500;
}
.limits-table td.num {
  font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; text-align: right;
}
.limits-table .confidence {
  display: inline-block; height: 6px; border-radius: 3px;
  background: var(--good); vertical-align: middle;
}

.futures-grid { display: grid; gap: 12px; grid-template-columns: 1fr; }
@media (min-width: 900px) { .futures-grid { grid-template-columns: 1fr 1fr; } }
.future-card {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 16px 18px;
}
.future-card h3 {
  font-family: "Public Sans", sans-serif; font-size: 14px; font-weight: 600;
  margin: 0 0 10px; color: var(--ink);
}
.future-rows { display: flex; flex-direction: column; gap: 4px; }
.future-row {
  display: grid; grid-template-columns: 1fr auto; gap: 10px;
  padding: 3px 0;
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  font-variant-numeric: tabular-nums;
}
.future-row .q { color: var(--ink); font-family: "Public Sans"; font-size: 13px; }
.future-row .p { color: var(--muted); }
.future-row .p.high { color: var(--good); font-weight: 600; }

.splits-note {
  background: color-mix(in oklab, var(--warn) 10%, var(--card));
  border: 1px solid color-mix(in oklab, var(--warn) 40%, var(--rule));
  padding: 16px 20px; border-radius: 10px;
  font-size: 13.5px; line-height: 1.6;
}
.splits-note strong { color: var(--ink); }
.splits-note ul { margin: 10px 0 0 20px; padding: 0; }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab" href="/analyst">AI Analyst</a>
        <a class="nav-tab active" href="/market">Market</a>
        <a class="nav-tab" href="/backtest">Model</a>
      </nav>
    </div>
    <form class="controls" method="get" action="/market">
      <a class="btn icon" href="/market?date={{ prev_date }}">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()">
      <a class="btn icon" href="/market?date={{ next_date }}">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/market?date={{ today }}">Today</a>{% endif %}
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>Market context &middot; {{ date_pretty }}</h1>
    <p class="sub">
      Sharp-money signals (Pinnacle limits, line context) and prediction-market futures (Polymarket).
      Retail public bet% from DraftKings/FanDuel aggregators requires a paid subscription &mdash;
      see the note at the bottom for options under $35/mo.
    </p>
  </div>

  <div class="section">
    <h2>Pinnacle limits per game</h2>
    <p class="lead">
      Pinnacle's maximum accepted wager on each moneyline. Higher limits = sharper price (more books copy it).
      When limits drop sharply before game time, Pinnacle has seen material sharp action &mdash; a signal retail
      splits don't capture.
    </p>
    {% if limits_rows %}
    <table class="limits-table">
      <thead>
        <tr>
          <th>Matchup</th>
          <th class="num">Away ML</th>
          <th class="num">Home ML</th>
          <th class="num">Total</th>
          <th class="num">Limit</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {% for r in limits_rows %}
        <tr>
          <td>{{ r.away }} @ {{ r.home }}
            <br><small style="color:var(--muted);font-size:11px">{{ r.first_pitch }}</small></td>
          <td class="num">{{ r.away_ml }}</td>
          <td class="num">{{ r.home_ml }}</td>
          <td class="num">{{ r.total or '—' }}</td>
          <td class="num">{% if r.limit %}${{ '{:,}'.format(r.limit|int) }}{% else %}—{% endif %}</td>
          <td><span class="confidence" style="width: {{ r.limit_bar }}px"></span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <p style="color:var(--muted);">No live Pinnacle markets for this slate.</p>
    {% endif %}
  </div>

  <div class="section">
    <h2>Polymarket futures</h2>
    <p class="lead">
      Live prediction-market prices on long-horizon MLB outcomes. Prices are implied probabilities
      directly &mdash; a 0.165 on "Yankees to win WS" means the market thinks there's a 16.5% chance.
    </p>
    <div class="futures-grid">
      {% for ev in futures_top %}
      <div class="future-card">
        <h3>{{ ev.title }}</h3>
        <div class="future-rows">
          {% for m in ev.top_markets %}
          <div class="future-row">
            <span class="q">{{ m.label }}</span>
            <span class="p {% if m.prob > 0.3 %}high{% endif %}">{{ (m.prob * 100)|round(1) }}%</span>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <footer>
    <div>Pinnacle limits &middot; Polymarket Gamma API &middot; updated {{ now }}</div>
    <div>No subscription required for any data on this page</div>
  </footer>
</main>
</body>
</html>
"""


@app.route("/market")
def market():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()

    try:
        games = get_games(date_str)
    except Exception:
        games = []

    today = datetime.now(CENTRAL).date().isoformat()

    # Limits rows for Pinnacle
    limits_rows = []
    max_limit = 1
    for g in games:
        o = g.get("odds") or {}
        pin = o.get("pinnacle") or {}
        ml = pin.get("moneyline") or {}
        if not ml.get("away_am"):
            continue
        lim = pin.get("ml_limit") or 0
        if lim > max_limit:
            max_limit = lim
    for g in games:
        o = g.get("odds") or {}
        pin = o.get("pinnacle") or {}
        ml = pin.get("moneyline") or {}
        if not ml.get("away_am"):
            continue
        total = (pin.get("total") or {}).get("line")
        am_a = ml.get("away_am")
        am_h = ml.get("home_am")
        lim = pin.get("ml_limit") or 0
        limits_rows.append({
            "away": g["away"]["team"],
            "home": g["home"]["team"],
            "first_pitch": g["first_pitch"],
            "away_ml": f"{'+' if am_a > 0 else ''}{am_a}",
            "home_ml": f"{'+' if am_h > 0 else ''}{am_h}",
            "total": total,
            "limit": lim,
            "limit_bar": int(round(140 * lim / max_limit)) if max_limit else 0,
        })

    # Polymarket futures (top by volume) — top 6
    pm = mlb_odds.get_polymarket_mlb() or {"futures": [], "single_game": []}
    futures_top = []
    for ev in pm.get("futures", [])[:6]:
        # Compress each event's markets to top 5 by prob (first outcome being "Yes")
        rows = []
        for m in ev["markets"]:
            if not m.get("probs"):
                continue
            # Yes/No outcome: use the Yes probability
            try:
                p_yes = m["probs"][0] if m["outcomes"][0].lower() == "yes" else max(m["probs"])
            except (IndexError, AttributeError):
                p_yes = max(m["probs"] or [0])
            rows.append({"label": m["question"], "prob": p_yes})
        rows.sort(key=lambda x: -x["prob"])
        futures_top.append({
            "title": ev["title"],
            "top_markets": rows[:5],
        })

    return render_template_string(
        MARKET_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        date_str=date_str,
        date_pretty=d.strftime("%A, %B %d").replace(" 0", " "),
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        limits_rows=limits_rows,
        futures_top=futures_top,
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


# ============================================================================
# Generic sport page (NFL, UFC, EPL, La Liga, Liga MX, UCL, Europa, Int'l)
# ============================================================================

SPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; {{ sport.name }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.summary {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 20px;
}
.summary .metric {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.summary .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.summary .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 24px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.summary .value.good { color: var(--good); }

.filter-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 16px; }
.filter-chips .chip {
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--rule-strong);
  background: var(--surface); color: var(--muted);
  font-size: 12px; font-weight: 500; text-decoration: none;
}
.filter-chips .chip:hover { color: var(--ink); background: var(--card); }
.filter-chips .chip.active { color: var(--ink); background: var(--card); border-color: var(--ink); }

.bets {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; overflow: hidden;
}
.bets table { width: 100%; border-collapse: collapse; font-size: 13px; }
.bets th {
  text-align: left; padding: 12px 14px;
  background: var(--surface); color: var(--muted);
  font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
  font-weight: 500; border-bottom: 1px solid var(--rule);
}
.bets td { padding: 10px 14px; border-bottom: 1px solid var(--rule); vertical-align: middle; }
.bets tr:last-child td { border-bottom: none; }
.bets tr.pos {
  background: color-mix(in oklab, var(--good) 10%, transparent);
  box-shadow: inset 3px 0 0 color-mix(in oklab, var(--good) 60%, transparent);
}
.bets tr.pos.strong {
  background: color-mix(in oklab, var(--good) 20%, transparent);
  box-shadow: inset 3px 0 0 var(--good);
}
.bets .num {
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums; text-align: right;
}
.bets .pick { font-weight: 500; color: var(--ink); }
.bets .matchup { color: var(--muted); }
.bets .matchup b { color: var(--ink); font-weight: 500; }

.mkt-pill {
  display: inline-block;
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; font-weight: 500;
  padding: 2px 8px; border-radius: 4px;
  background: var(--chip-bg); color: var(--muted);
  border: 1px solid var(--rule); white-space: nowrap;
}
.mkt-pill.mkt-ml       { color: var(--accent); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); }
.mkt-pill.mkt-total    { color: var(--warn);   border-color: color-mix(in oklab, var(--warn) 40%, var(--rule)); }
.mkt-pill.mkt-spread, .mkt-pill.mkt-run_line { color: var(--good); border-color: color-mix(in oklab, var(--good) 40%, var(--rule)); }
.mkt-pill.mkt-1h_ml, .mkt-pill.mkt-1h_spread, .mkt-pill.mkt-1h_total {
  background: color-mix(in oklab, var(--muted) 15%, var(--chip-bg));
}

.book-pill {
  padding: 1px 6px; border-radius: 3px;
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  background: var(--surface); border: 1px solid var(--rule);
  color: var(--muted);
}
.book-pill.pinnacle { color: var(--muted-2); }
.book-pill.draftkings { color: #1b7a4f; }
.book-pill.fanduel    { color: #0066cc; }
.book-pill.betmgm     { color: #a66f00; }
.book-pill.caesars    { color: #a63329; }

.ev-cell.strong { color: var(--good); font-weight: 700; }
.ev-cell.pos    { color: var(--good); }
.ev-cell.neg    { color: var(--muted-2); }

.empty-bets {
  padding: 60px 24px; text-align: center; color: var(--muted);
}
.empty-bets h2 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: 24px; color: var(--ink); margin: 0 0 8px;
}
.note {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 18px;
  color: var(--muted); font-size: 12.5px; line-height: 1.6;
  margin-top: 20px; max-width: 900px;
}
.note strong { color: var(--ink); }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      {{ nav|safe }}
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>{{ sport.name }} &middot; edges</h1>
    <p class="sub">
      Positive-EV plays identified by comparing each US sportsbook's price to
      <strong>Pinnacle's devigged fair probability</strong> &mdash; the industry
      consensus "true" market line.
      {% if has_model %}
      A dedicated {{ sport.name }} Elo+QB model is available in the
      <a href="/sport/{{ sport.slug }}/backtest" style="color:var(--accent);text-decoration:underline">backtest &rarr;</a>
      (accuracy, calibration, QB rankings) but it does not drive the EV below &mdash; longshot
      model/market disagreements inflate EV past realistic levels, so the fair column stays on
      Pinnacle.
      {% endif %}
    </p>
    {% if has_model %}
    <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap">
      <a class="btn" href="/sport/{{ sport.slug }}/backtest">View model &amp; backtest</a>
    </div>
    {% endif %}
  </div>

  <div class="summary">
    <div class="metric">
      <div class="label">Games with lines</div>
      <div class="value">{{ game_count }}</div>
    </div>
    <div class="metric">
      <div class="label">+EV plays</div>
      <div class="value {% if positive_count %}good{% endif %}">{{ positive_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Strong (&ge;2%)</div>
      <div class="value {% if strong_count %}good{% endif %}">{{ strong_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Odds API books</div>
      <div class="value" style="font-size:14px;line-height:1.3">{{ 'DK/FD/BMG/C' if odds_api_available else 'Pinnacle only' }}</div>
    </div>
  </div>

  <div class="filter-chips">
    {% set mkts = available_markets %}
    <a class="chip {% if market_filter == 'ALL' %}active{% endif %}" href="/sport/{{ sport.slug }}">All</a>
    {% for mk, lab in mkts %}
      <a class="chip {% if market_filter == mk %}active{% endif %}" href="/sport/{{ sport.slug }}?market={{ mk|lower }}">{{ lab }}</a>
    {% endfor %}
    <a class="chip {% if show == 'pos' %}active{% endif %}" href="/sport/{{ sport.slug }}?market={{ market_filter|lower }}&show=pos">+EV only</a>
  </div>

  <div class="bets">
    {% if rows %}
    <table>
      <thead>
        <tr>
          <th>Market</th>
          <th>Game</th>
          <th>Pick</th>
          <th class="num">Fair</th>
          <th class="num">Pin price</th>
          <th class="num">Best price</th>
          <th>Book</th>
          <th class="num">EV</th>
          <th class="num">Stake 1/4K</th>
        </tr>
      </thead>
      <tbody>
        {% for r in rows %}
        <tr class="{% if r.ev_pct >= 2 %}pos strong{% elif r.ev_pct > 0 %}pos{% endif %}">
          <td><span class="mkt-pill mkt-{{ r.market|replace(' ','_')|lower }}">{{ r.market }}</span></td>
          <td class="matchup"><b>{{ r.away }}</b> at <b>{{ r.home }}</b><br>
            <small style="font-size:11px">{{ r.start_time_et }}</small></td>
          <td class="pick">{{ r.pick }}</td>
          <td class="num">{{ (r.fair_prob * 100)|round(1) }}%</td>
          <td class="num">
            {% if r.pin_american is not none %}
              {{ ('+' if r.pin_american > 0 else '') ~ r.pin_american }}
              <span style="color:var(--muted-2)">({{ '%.2f'|format(r.pin_decimal) }})</span>
            {% else %}&mdash;{% endif %}
          </td>
          <td class="num">
            {% if r.book_american is not none %}
              {{ ('+' if r.book_american > 0 else '') ~ r.book_american }}
              <span style="color:var(--muted-2)">({{ '%.2f'|format(r.book_decimal) }})</span>
            {% else %}&mdash;{% endif %}
          </td>
          <td><span class="book-pill {{ r.book }}">{{ r.book }}</span></td>
          <td class="num ev-cell {% if r.ev_pct >= 2 %}strong{% elif r.ev_pct > 0 %}pos{% else %}neg{% endif %}">
            {{ '%+.2f'|format(r.ev_pct) }}%
          </td>
          <td class="num">{{ '%.1f'|format(r.kelly_pct) }}%</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty-bets">
      <h2>No bets to show.</h2>
      <p>{% if game_count %}The current filter has no matches.{% else %}Pinnacle hasn't posted this slate yet.{% endif %}</p>
    </div>
    {% endif %}
  </div>

  <div class="note">
    <strong>How this works:</strong> Pinnacle has the lowest vig in the industry; after devigging
    its two-way (or three-way, for soccer) market we treat the result as a close estimate of the
    "true" probability. Each US book is then compared against that fair probability using
    <code>EV = fair &middot; book_decimal &minus; 1</code>. Rows where no US book has posted get
    Pinnacle's own price in the Best column (which, by construction, will be slightly negative EV).
    {% if sport.has_halves %}<br><br>
    <strong>1H markets:</strong> First-half lines from Pinnacle's period&nbsp;1 markets.
    {% endif %}
  </div>

  <footer>
    <div>Pinnacle leagueId={{ sport.pinnacle_league_id }} &middot; {% if sport.odds_api_key %}Odds API key: <code>{{ sport.odds_api_key }}</code>{% else %}no Odds API key for this league{% endif %}</div>
    <div>updated {{ now }}</div>
  </footer>
</main>
</body>
</html>
"""


def _format_et(iso_utc):
    if not iso_utc:
        return ""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(CENTRAL)
        return dt.strftime("%a %b %d, %I:%M %p CT").replace(" 0", " ")
    except (ValueError, TypeError):
        return iso_utc


SPORT_SCHEDULE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; {{ sport.name }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero h1 { font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400; font-size: clamp(32px, 5vw, 48px); line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 8px; font-variation-settings: "opsz" 144; }
.hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.count { font-family: "JetBrains Mono", monospace; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-variant-numeric: tabular-nums; }
.count strong { color: var(--ink); font-weight: 500; }
.hero-row { display: flex; justify-content: space-between; align-items: baseline; gap: 24px; }

.grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 760px)  { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 1180px) { .grid { grid-template-columns: repeat(3, 1fr); } }

.card {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 18px 20px 16px;
  display: flex; flex-direction: column; gap: 14px;
  transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
.card:hover { border-color: var(--rule-strong); transform: translateY(-1px); box-shadow: var(--card-hover-shadow); }

.card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.time { font-family: "JetBrains Mono", monospace; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.chip-live {
  font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent); padding: 2px 7px; border: 1px solid color-mix(in oklab, var(--accent) 40%, var(--rule));
  border-radius: 999px;
}

.matchup { display: flex; flex-direction: column; gap: 2px; }
.team-name { font-weight: 600; font-size: 20px; line-height: 1.15; letter-spacing: -0.01em; color: var(--ink); }
.at { font-family: "Fraunces", Georgia, serif; font-style: italic; font-size: 13px; color: var(--muted); padding: 2px 0 2px 10px; line-height: 1; }

.market { border-top: 1px dashed var(--rule); padding-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.market-label { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }
.market-row { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; font-size: 13px; align-items: baseline; }
.market-row .side { color: var(--ink); font-weight: 500; }
.market-row .price { font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; color: var(--muted); }
.market-row .price b { color: var(--ink); font-weight: 500; }
.market-row .fair { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted-2); font-variant-numeric: tabular-nums; }

.prob-bar {
  display: flex; height: 28px; border-radius: 8px; overflow: hidden;
  background: var(--chip-bg); border: 1px solid var(--rule);
  font-family: "JetBrains Mono", monospace; font-size: 11px; font-variant-numeric: tabular-nums; letter-spacing: 0.01em;
}
.prob-bar .side { display: flex; align-items: center; padding: 0 10px; gap: 6px; color: var(--muted); min-width: 0; white-space: nowrap; }
.prob-bar .side.away { justify-content: flex-start; background: color-mix(in oklab, var(--muted) 15%, var(--chip-bg)); }
.prob-bar .side.home { justify-content: flex-end; background: color-mix(in oklab, var(--muted) 10%, var(--chip-bg)); }
.prob-bar[data-favors="home"] .side.home { background: color-mix(in oklab, var(--good) 26%, var(--chip-bg)); color: var(--ink); }
.prob-bar[data-favors="home"] .side.home .pct { color: var(--ink); font-weight: 600; text-shadow: var(--ev-strong-glow); }
.prob-bar[data-favors="away"] .side.away { background: color-mix(in oklab, var(--good) 26%, var(--chip-bg)); color: var(--ink); }
.prob-bar[data-favors="away"] .side.away .pct { color: var(--ink); font-weight: 600; text-shadow: var(--ev-strong-glow); }

.best-book {
  padding-top: 6px; border-top: 1px dotted var(--rule);
  display: flex; justify-content: space-between;
  font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--muted);
}
.best-book b { color: var(--ink); font-weight: 500; }

.empty-sched {
  padding: 60px 24px; text-align: center; color: var(--muted);
  grid-column: 1 / -1;
}
.empty-sched h2 { font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400; font-size: 24px; color: var(--ink); margin: 0 0 8px; }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      {{ nav|safe }}
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <div class="hero-row">
      <h1>{{ sport.name }} &middot; schedule</h1>
      <span class="count">{% if cards %}<strong>{{ cards|length }}</strong> game{{ '' if cards|length == 1 else 's' }}{% else %}No games{% endif %}</span>
    </div>
    <p class="sub">
      Each card shows Pinnacle's moneyline, {% if sport.spread_label %}{{ sport.spread_label|lower }}, {% endif %}and main total, with devigged fair probability per side.
      {% if sport.slug == 'nfl' %}NFL games also show the Elo+QB model's win probability.{% endif %}
      Daily best-EV plays live on the <a href="/" style="color:var(--accent)">Picks</a> page.
    </p>
  </div>

  <div class="grid">
    {% if cards %}
    {% for g in cards %}
    <article class="card">
      <div class="card-head">
        <span class="time">{{ g.start_time_et }}</span>
        {% if g.is_live %}<span class="chip-live">Live</span>{% endif %}
      </div>

      <div class="matchup">
        <div class="team-name">{{ g.away }}</div>
        <div class="at">at</div>
        <div class="team-name">{{ g.home }}</div>
      </div>

      {% if g.p_home_pct is not none %}
      <div>
        <div class="market-label" style="margin-bottom:6px">Model probability{% if g.p_draw_pct is not none %} (3-way){% endif %}</div>
        {% if g.p_draw_pct is not none %}
        <div class="prob-bar" data-favors="{{ 'home' if g.p_home_pct >= g.p_away_pct and g.p_home_pct >= g.p_draw_pct else ('away' if g.p_away_pct >= g.p_draw_pct else 'draw') }}">
          <div class="side away" style="width: {{ g.p_away_pct }}%"><span>{{ g.away_short }}</span> <span class="pct">{{ g.p_away_pct }}%</span></div>
          <div class="side draw" style="width: {{ g.p_draw_pct }}%; justify-content:center; background: color-mix(in oklab, var(--warn) 25%, var(--chip-bg)); color: var(--ink);"><span class="pct">D {{ g.p_draw_pct }}%</span></div>
          <div class="side home" style="width: {{ g.p_home_pct }}%"><span class="pct">{{ g.p_home_pct }}%</span> <span>{{ g.home_short }}</span></div>
        </div>
        {% else %}
        <div class="prob-bar" data-favors="{{ 'home' if g.p_home_pct >= 50 else 'away' }}">
          <div class="side away" style="width: {{ 100 - g.p_home_pct }}%"><span>{{ g.away_short }}</span> <span class="pct">{{ 100 - g.p_home_pct }}%</span></div>
          <div class="side home" style="width: {{ g.p_home_pct }}%"><span class="pct">{{ g.p_home_pct }}%</span> <span>{{ g.home_short }}</span></div>
        </div>
        {% endif %}
      </div>
      {% endif %}

      {% if g.ml %}
      <div class="market">
        <div class="market-label">Moneyline</div>
        {% for row in g.ml %}
        <div class="market-row">
          <span class="side">{{ row.label }}</span>
          <span class="price"><b>{{ row.price }}</b> <span style="color:var(--muted-2)">({{ row.decimal }})</span></span>
          <span class="fair">{{ row.fair_pct }}% fair</span>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      {% if g.spread %}
      <div class="market">
        <div class="market-label">{{ sport.spread_label or 'Spread' }}</div>
        {% for row in g.spread %}
        <div class="market-row">
          <span class="side">{{ row.label }}</span>
          <span class="price"><b>{{ row.price }}</b> <span style="color:var(--muted-2)">({{ row.decimal }})</span></span>
          <span class="fair">&nbsp;</span>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      {% if g.total %}
      <div class="market">
        <div class="market-label">Total {{ g.total_line }}</div>
        {% for row in g.total %}
        <div class="market-row">
          <span class="side">{{ row.label }}</span>
          <span class="price"><b>{{ row.price }}</b> <span style="color:var(--muted-2)">({{ row.decimal }})</span></span>
          <span class="fair">{{ row.fair_pct }}% fair</span>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      {% if g.best_book_home or g.best_book_away %}
      <div class="best-book">
        {% if g.best_book_away %}<span><b>{{ g.away_short }}</b> {{ g.best_dec_away }} @ {{ g.best_book_away }}</span>{% endif %}
        {% if g.best_book_home %}<span><b>{{ g.home_short }}</b> {{ g.best_dec_home }} @ {{ g.best_book_home }}</span>{% endif %}
      </div>
      {% endif %}
    </article>
    {% endfor %}
    {% else %}
    <div class="empty-sched">
      <h2>No games to show.</h2>
      <p>Pinnacle hasn't posted this slate yet, or {{ sport.name }} is between phases.</p>
    </div>
    {% endif %}
  </div>

  <footer>
    <div>Pinnacle leagueId={{ sport.pinnacle_league_id }} &middot; {% if odds_api_available %}DK/FanDuel/BetMGM/Caesars via Odds API{% else %}Pinnacle only (set ODDS_API_KEY to enable US books){% endif %}</div>
    <div>updated {{ now }}</div>
  </footer>
</main>
</body>
</html>
"""


def _short_name(full):
    """Best-effort short name: last token for 2-word teams, else as-is."""
    parts = (full or "").split()
    if len(parts) >= 2 and len(parts[-1]) > 2:
        return parts[-1]
    return full


def _signed_am_str(am):
    if am is None:
        return "—"
    return f"+{am}" if am > 0 else str(am)


@app.route("/sport/<slug>")
def sport_schedule(slug):
    sport = sports.by_slug(slug)
    if not sport or sport.get("dedicated"):
        from flask import redirect
        return redirect("/", code=302)

    # Sport-specific model probability function
    model_prob_fn = None
    if slug == "nfl":
        try:
            import nfl_model
            # Lightweight final_state (reads ~5KB side-file, not 770KB cache)
            nfl_state = nfl_model.get_final_state()
            final_elo = nfl_state.get("final_elo", {}) if nfl_state else {}
            def _nfl_prob(g):
                h = nfl_model.abbr_from_name(g.get("home_name", ""))
                a = nfl_model.abbr_from_name(g.get("away_name", ""))
                if not h or not a:
                    return None
                h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
                a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
                p = nfl_model.predict_win_prob(h_elo, a_elo)
                return {"home": p, "away": 1 - p, "draw": None}
            model_prob_fn = _nfl_prob
        except Exception:
            model_prob_fn = None
    elif slug in _SOCCER_BACKTEST_SLUGS:
        try:
            import soccer_model
            # Lightweight final_elo (reads ~700B side-file, not ~1.4MB cache)
            final_elo = soccer_model.get_final_elo(slug) or {}
            def _soccer_prob(g):
                h_name = g.get("home_name", "")
                a_name = g.get("away_name", "")
                h_elo = final_elo.get(h_name, soccer_model.INITIAL_ELO)
                a_elo = final_elo.get(a_name, soccer_model.INITIAL_ELO)
                p_h, p_d, p_a = soccer_model.predict_3way(h_elo, a_elo)
                return {"home": p_h, "away": p_a, "draw": p_d}
            model_prob_fn = _soccer_prob
        except Exception:
            model_prob_fn = None

    try:
        games = generic_odds.build_sport_games(sport, model_prob_fn=model_prob_fn)
    except Exception:
        games = []

    cards = []
    for g in games:
        ml_rows, spread_rows, total_rows = [], [], []
        pin_ml = g.get("ml") or {}
        if pin_ml.get("home_am") is not None:
            # Devig for fair %
            p_h = mlb_odds.american_to_prob(pin_ml["home_am"])
            p_a = mlb_odds.american_to_prob(pin_ml["away_am"])
            p_d = mlb_odds.american_to_prob(pin_ml.get("draw_am")) if sport["ml_outcomes"] == 3 else None
            if sport["ml_outcomes"] == 3 and p_d:
                total = p_h + p_d + p_a
                fair_h = p_h / total; fair_d = p_d / total; fair_a = p_a / total
            else:
                fh, fa = mlb_odds.devig_two_sided(p_h, p_a)
                fair_h, fair_a, fair_d = fh, fa, None

            for label, am, fair in [
                (g["away_name"], pin_ml.get("away_am"), fair_a),
                (g["home_name"], pin_ml.get("home_am"), fair_h),
            ]:
                dec = mlb_odds.american_to_decimal(am)
                ml_rows.append({
                    "label": label,
                    "price": _signed_am_str(am),
                    "decimal": f"{dec:.2f}" if dec else "—",
                    "fair_pct": f"{round((fair or 0) * 100)}" if fair else "—",
                })
            if fair_d is not None:
                ml_rows.insert(1, {
                    "label": "Draw",
                    "price": _signed_am_str(pin_ml.get("draw_am")),
                    "decimal": f"{mlb_odds.american_to_decimal(pin_ml['draw_am']):.2f}",
                    "fair_pct": f"{round(fair_d * 100)}",
                })

        pin_spread = g.get("spread") or {}
        if pin_spread.get("home_am") is not None:
            hpt = pin_spread.get("line_home")
            apt = -hpt if hpt is not None else None
            for label, am in [
                (f"{g['away_name']} {('' if (apt or 0) < 0 else '+')}{apt}", pin_spread.get("away_am")),
                (f"{g['home_name']} {('' if (hpt or 0) < 0 else '+')}{hpt}", pin_spread.get("home_am")),
            ]:
                dec = mlb_odds.american_to_decimal(am)
                spread_rows.append({
                    "label": label,
                    "price": _signed_am_str(am),
                    "decimal": f"{dec:.2f}" if dec else "—",
                })

        pin_total = g.get("total") or {}
        total_line = None
        if pin_total.get("line") is not None:
            total_line = pin_total["line"]
            p_o = mlb_odds.american_to_prob(pin_total.get("over_am"))
            p_u = mlb_odds.american_to_prob(pin_total.get("under_am"))
            fo, fu = mlb_odds.devig_two_sided(p_o, p_u)
            for label, am, fair in [
                (f"Over {total_line}", pin_total.get("over_am"), fo),
                (f"Under {total_line}", pin_total.get("under_am"), fu),
            ]:
                dec = mlb_odds.american_to_decimal(am)
                total_rows.append({
                    "label": label,
                    "price": _signed_am_str(am),
                    "decimal": f"{dec:.2f}" if dec else "—",
                    "fair_pct": f"{round((fair or 0) * 100)}" if fair else "—",
                })

        # Best US book pulled from bets list (any ML side's best_book/best_decimal)
        best_book_home = best_book_away = None
        best_dec_home = best_dec_away = None
        for b in (g.get("bets") or []):
            if b["market"] == "ML":
                if b["side"] == "home" and b.get("book") not in (None, "pinnacle"):
                    best_book_home = b["book"]
                    best_dec_home = f"{b['book_decimal']:.2f}" if b.get("book_decimal") else None
                elif b["side"] == "away" and b.get("book") not in (None, "pinnacle"):
                    best_book_away = b["book"]
                    best_dec_away = f"{b['book_decimal']:.2f}" if b.get("book_decimal") else None

        # Model probability (2-way for NFL, 3-way for soccer)
        p_home_pct = p_draw_pct = p_away_pct = None
        mp = g.get("model_prob")
        if mp and mp.get("home") is not None:
            p_home_pct = round(mp["home"] * 100)
            if mp.get("draw") is not None:
                p_draw_pct = round(mp["draw"] * 100)
                p_away_pct = max(0, 100 - p_home_pct - p_draw_pct)
            else:
                p_away_pct = 100 - p_home_pct

        cards.append({
            "away": g["away_name"], "home": g["home_name"],
            "away_short": _short_name(g["away_name"]), "home_short": _short_name(g["home_name"]),
            "start_time_et": _format_et(g.get("start_time")),
            "is_live": bool(g.get("is_live")),
            "ml": ml_rows, "spread": spread_rows, "total": total_rows,
            "total_line": total_line,
            "best_book_home": best_book_home, "best_book_away": best_book_away,
            "best_dec_home": best_dec_home, "best_dec_away": best_dec_away,
            "p_home_pct": p_home_pct,
            "p_draw_pct": p_draw_pct,
            "p_away_pct": p_away_pct,
        })

    return render_template_string(
        SPORT_SCHEDULE_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip(slug),
        nav=render_sport_nav(slug, "schedule"),
        sport=sport,
        cards=cards,
        odds_api_available=generic_odds.odds_api_available(),
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


@app.route("/sport/<slug>/edges")
def sport_edges(slug):
    sport = sports.by_slug(slug)
    if not sport or sport.get("dedicated"):
        from flask import redirect
        return redirect("/", code=302)

    market_filter = (request.args.get("market") or "all").upper()
    show = request.args.get("show") or "all"

    # NFL gets a trained model (same walk-forward Elo approach as MLB);
    # other sports fall back to Pinnacle devig for fair probabilities.
    # Live predictions use the end-of-last-season state from the 12-season
    # multi-season fit, so team Elo and QB ratings carry career-level signal.
    model_prob_fn = None
    if slug == "nfl":
        try:
            import nfl_model
            nfl_state = nfl_model.get_final_state()
            final_elo = nfl_state.get("final_elo", {}) if nfl_state else {}
            final_qb_elo = nfl_state.get("final_qb_elo", {}) if nfl_state else {}
            def _nfl_prob(g):
                h = nfl_model.abbr_from_name(g.get("home_name", ""))
                a = nfl_model.abbr_from_name(g.get("away_name", ""))
                if not h or not a:
                    return None
                h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
                a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
                p = nfl_model.predict_win_prob(h_elo, a_elo)
                return {"home": p, "away": 1 - p, "draw": None}
            model_prob_fn = _nfl_prob
        except Exception:
            model_prob_fn = None

    try:
        games = generic_odds.build_sport_games(sport, model_prob_fn=model_prob_fn)
    except Exception:
        games = []

    rows = []
    for g in games:
        for b in g.get("bets") or []:
            mk_key = b["market"].upper().replace(" ", "_")
            if market_filter != "ALL" and market_filter != mk_key:
                continue
            if show == "pos" and b["ev_pct"] <= 0:
                continue
            rows.append({**b,
                         "away": g["away_name"], "home": g["home_name"],
                         "start_time_et": _format_et(g.get("start_time"))})
    rows.sort(key=lambda r: -r["ev_pct"])
    positive_count = sum(1 for r in rows if r["ev_pct"] > 0)
    strong_count = sum(1 for r in rows if r["ev_pct"] >= 2.0)

    # Available market chips depend on sport
    markets_meta = [("ML", "Moneyline"), ("TOTAL", "Total")]
    if sport.get("spread_label"):
        markets_meta.insert(1, ("SPREAD", sport["spread_label"]))
    else:
        markets_meta.insert(1, ("SPREAD", "Spread"))
    if sport.get("has_halves"):
        markets_meta += [("1H_ML", "1H ML"), ("1H_SPREAD", "1H Spread"), ("1H_TOTAL", "1H Total")]

    return render_template_string(
        SPORT_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip(slug),
        nav=render_sport_nav(slug, "edges"),
        sport=sport,
        rows=rows,
        game_count=len(games),
        positive_count=positive_count,
        strong_count=strong_count,
        odds_api_available=generic_odds.odds_api_available(),
        market_filter=market_filter,
        available_markets=markets_meta,
        show=show,
        has_model=(model_prob_fn is not None),
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


# ============================================================================
# NFL backtest page
# ============================================================================

NFL_BACKTEST_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; NFL Model &amp; Backtest</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero-block { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero-block h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero-block .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.grid-metrics {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 24px;
}
.metric {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.metric .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.metric .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 24px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.metric .value.accent { color: var(--accent); }
.metric .value.good { color: var(--good); }
.metric .foot { margin-top: 6px; font-size: 11.5px; color: var(--muted); font-family: "JetBrains Mono", monospace; }
.metric .delta.up { color: var(--good); }

.section {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
}
.section h2 {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 500; font-size: 20px; margin: 0 0 10px;
  letter-spacing: -0.01em;
}
.section .lead { color: var(--muted); margin: 0 0 16px; font-size: 13px; line-height: 1.6; max-width: 760px; }

.calib-wrap { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 900px) { .calib-wrap { grid-template-columns: 1fr 1fr; } }
.calib-svg-wrap {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 8px;
  padding: 20px; display: flex; justify-content: center;
}
svg.calib { max-width: 100%; height: auto; }

.calib-table, .elo-table, .pred-table {
  width: 100%; border-collapse: collapse;
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 12.5px;
}
.calib-table th, .calib-table td,
.elo-table th, .elo-table td,
.pred-table th, .pred-table td {
  padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: right;
}
.calib-table th:first-child, .elo-table th:nth-child(2), .pred-table th { text-align: left; }
.calib-table td:first-child, .elo-table td:nth-child(2), .pred-table td { text-align: left; }
.calib-table th, .elo-table th, .pred-table th {
  color: var(--muted); font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; font-weight: 500;
}
.elo-cols { display: grid; grid-template-columns: 1fr; gap: 20px; }
@media (min-width: 900px) { .elo-cols { grid-template-columns: 1fr 1fr; } }

.pred-correct { color: var(--good); font-weight: 600; }
.pred-wrong { color: var(--muted-2); }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      {{ nav|safe }}
    </div>
    <div class="controls">
      <a class="btn" href="/sport/nfl/backtest?refresh=1">Refit</a>
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero-block">
    <h1>NFL model &amp; 12-season backtest</h1>
    <p class="sub">
      Walk-forward Elo with margin-of-victory damping, rest-day adjustment, and per-starter QB rating.
      Fit chronologically across {{ state.total_games }} regular-season games spanning
      <strong>{{ state.first_season }}&ndash;{{ state.last_season }}</strong> ({{ state.num_seasons }} seasons).
      Team Elo regresses toward 1500 by 1/3 at each season boundary; QB Elo persists across seasons.
      Predictions are logged <em>before</em> each game using Elo as-of first kickoff; the first
      {{ state.warmup_seasons }} seasons are pure warm-up, and the remaining
      <strong>{{ state.last_season - state.scored_from_season + 1 }} seasons are scored</strong> against actual outcomes below.
    </p>
  </div>

  <div class="grid-metrics">
    <div class="metric">
      <div class="label">Accuracy</div>
      <div class="value accent">{{ '%.1f'|format(m.accuracy * 100) }}%</div>
      <div class="foot">
        <span class="delta up">+{{ '%.1f'|format((m.accuracy - m.home_baseline_accuracy) * 100) }} pts</span>
        vs home-team ({{ '%.1f'|format(m.home_baseline_accuracy * 100) }}%)
      </div>
    </div>
    <div class="metric">
      <div class="label">Log loss</div>
      <div class="value">{{ '%.4f'|format(m.log_loss) }}</div>
      <div class="foot">baseline {{ '%.4f'|format(m.home_baseline_log_loss) }} &middot; coin 0.6931</div>
    </div>
    <div class="metric">
      <div class="label">Brier</div>
      <div class="value">{{ '%.4f'|format(m.brier_score) }}</div>
      <div class="foot">0.25 = coin flip</div>
    </div>
    <div class="metric">
      <div class="label">Scored games</div>
      <div class="value">{{ m.n }}</div>
      <div class="foot">seasons {{ state.scored_from_season }}&ndash;{{ state.last_season }}</div>
    </div>
    {% if m.record_baseline_accuracy is not none %}
    <div class="metric">
      <div class="label">vs better record</div>
      <div class="value">{{ '%.1f'|format(m.record_baseline_accuracy * 100) }}%</div>
      <div class="foot">naive heuristic</div>
    </div>
    {% endif %}
    <div class="metric">
      <div class="label">Hyperparams</div>
      <div class="value" style="font-size:14px;line-height:1.4">
        K={{ state.hyperparams.K|int }} &middot; HFA=+{{ state.hyperparams.HFA|int }}
      </div>
      <div class="foot">rest weight {{ state.hyperparams.REST_WEIGHT }} Elo/day</div>
    </div>
  </div>

  <div class="section">
    <h2>Per-season breakdown</h2>
    <p class="lead">
      Model performance for each scored season. A stable model should hover near its aggregate
      accuracy; big dips reveal years when the Elo + rest + QB signal mix underfit the slate
      (injuries, scheme changes, 2021 COVID effects).
    </p>
    <div class="table-scroll">
    <table class="calib-table">
      <thead><tr>
        <th>Season</th><th>Games</th><th>Accuracy</th><th>Log loss</th><th>Brier</th>
      </tr></thead>
      <tbody>
        {% for r in per_season_rows %}
        <tr>
          <td>{{ r.season }}</td>
          <td>{{ r.n }}</td>
          <td>{{ (r.accuracy * 100)|round(2) }}%</td>
          <td>{{ '%.4f'|format(r.log_loss) }}</td>
          <td>{{ '%.4f'|format(r.brier) }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </div>

  <div class="section">
    <h2>Calibration</h2>
    <p class="lead">Predicted probabilities in 10% bins vs observed home win rate. 45&deg; line = perfect calibration.</p>
    <div class="calib-wrap">
      <div class="calib-svg-wrap">{{ calibration_svg|safe }}</div>
      <div>
        <table class="calib-table">
          <thead><tr><th>Bin</th><th>N</th><th>Avg pred</th><th>Actual</th><th>Delta</th></tr></thead>
          <tbody>
            {% for c in m.calibration %}
            <tr>
              <td>{{ (c.bin_lo*100)|int }}-{{ (c.bin_hi*100)|int }}%</td>
              <td>{{ c.n }}</td>
              <td>{{ '%.3f'|format(c.avg_pred) }}</td>
              <td>{{ '%.3f'|format(c.actual) }}</td>
              <td>{{ '%+.3f'|format(c.actual - c.avg_pred) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Final team Elo ratings</h2>
    <p class="lead">End-of-regular-season team ratings. Feed today's live NFL predictions on the NFL tab (blended with the starter's QB Elo at prediction time).</p>
    <div class="elo-cols">
      <div>
        <table class="elo-table">
          <thead><tr><th>#</th><th>Team</th><th>Elo</th><th>W-L</th></tr></thead>
          <tbody>
            {% for ab, elo, wl in top_elo %}
            <tr>
              <td style="color:var(--muted)">{{ loop.index }}</td>
              <td>{{ state.team_names[ab] }}</td>
              <td>{{ elo|round|int }}</td>
              <td>{{ wl[0] }}-{{ wl[1] }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div>
        <table class="elo-table">
          <thead><tr><th>#</th><th>Team</th><th>Elo</th><th>W-L</th></tr></thead>
          <tbody>
            {% for ab, elo, wl in bottom_elo %}
            <tr>
              <td style="color:var(--muted)">{{ loop.index + 16 }}</td>
              <td>{{ state.team_names[ab] }}</td>
              <td>{{ elo|round|int }}</td>
              <td>{{ wl[0] }}-{{ wl[1] }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  {% if top_qbs %}
  <div class="section">
    <h2>Quarterback ratings</h2>
    <p class="lead">
      Per-QB Elo (minimum 8 games started this season). The model blends
      <code>{{ (state.hyperparams.QB_WEIGHT * 100)|int }}%</code> of each QB's rating-diff from
      1500 into their team's effective Elo at prediction time &mdash; so a top-rated starter at
      a mid-tier team can flip the favorite.
    </p>
    <div class="elo-cols">
      <div>
        <table class="elo-table">
          <thead><tr><th>#</th><th>QB</th><th>Elo</th><th>Games</th></tr></thead>
          <tbody>
            {% for qb, elo, games in top_qbs %}
            <tr>
              <td style="color:var(--muted)">{{ loop.index }}</td>
              <td>{{ qb }}</td>
              <td>{{ elo|round|int }}</td>
              <td>{{ games }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div>
        <table class="elo-table">
          <thead><tr><th>#</th><th>QB</th><th>Elo</th><th>Games</th></tr></thead>
          <tbody>
            {% for qb, elo, games in bottom_qbs %}
            <tr>
              <td style="color:var(--muted)">{{ bottom_qb_rank_start + loop.index }}</td>
              <td>{{ qb }}</td>
              <td>{{ elo|round|int }}</td>
              <td>{{ games }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  {% endif %}

  <div class="section">
    <h2>Sample of scored predictions</h2>
    <p class="lead">Last 24 games in the scored window.</p>
    <table class="pred-table">
      <thead><tr>
        <th>Date</th><th>Wk</th><th>Matchup</th>
        <th>Model pick</th><th>Actual</th><th></th>
      </tr></thead>
      <tbody>
      {% for p in sample_preds %}
        <tr>
          <td>{{ p.date }}</td>
          <td>{{ p.week }}</td>
          <td>{{ p.away }} at {{ p.home }}</td>
          <td>{% if p.p_home >= 0.5 %}{{ p.home }} {{ (p.p_home * 100)|int }}%{% else %}{{ p.away }} {{ ((1-p.p_home) * 100)|int }}%{% endif %}</td>
          <td>{{ p.away_score }}&ndash;{{ p.home_score }}</td>
          <td>
            {% if (p.p_home >= 0.5) == p.home_won %}<span class="pred-correct">&check;</span>
            {% else %}<span class="pred-wrong">&times;</span>{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <footer>
    <div>Fit on {{ state.total_games }} games &middot; data: nflverse community dataset</div>
    <div>last refit {{ state.generated_at }}</div>
  </footer>
</main>
</body>
</html>
"""


@app.route("/sport/nfl/backtest")
def nfl_backtest():
    try:
        import nfl_model
        if request.args.get("refresh"):
            state = nfl_model.get_or_run_multi_season_backtest(refresh=True)
        else:
            state = nfl_model.get_or_run_multi_season_backtest()
    except Exception as e:
        state = None
    if not state:
        return render_template_string(
            NFL_BACKTEST_TEMPLATE,
            fonts_link=FONTS_LINK,
            shared_style=SHARED_STYLE,
            sport_strip=render_sport_strip("nfl"),
            nav=render_sport_nav("nfl", "backtest"),
            state=None, m=None, top_elo=[], bottom_elo=[],
            top_qbs=[], bottom_qbs=[], bottom_qb_rank_start=0,
            per_season_rows=[],
            sample_preds=[], calibration_svg="",
        )
    m = state.get("metrics") or {}

    # Compute last-season W-L from predictions (for the Final Elo tables)
    last_season = state["last_season"]
    final_wl = {}
    for p in state["predictions"]:
        if p["season"] != last_season:
            continue
        h, a = p["home_abbr"], p["away_abbr"]
        final_wl.setdefault(h, [0, 0])
        final_wl.setdefault(a, [0, 0])
        if p["home_won"]:
            final_wl[h][0] += 1
            final_wl[a][1] += 1
        else:
            final_wl[h][1] += 1
            final_wl[a][0] += 1

    elo_items = sorted(state["final_elo"].items(), key=lambda kv: -kv[1])
    top_elo = [(ab, r, final_wl.get(ab, [0, 0])) for ab, r in elo_items[:16]]
    bottom_elo = [(ab, r, final_wl.get(ab, [0, 0])) for ab, r in elo_items[16:]]

    # QB rankings (min 24 career games across the 12-season window)
    qb_games = state.get("qb_games", {}) or {}
    qb_items = sorted(
        [(q, e, qb_games.get(q, 0)) for q, e in (state.get("final_qb_elo") or {}).items()
         if qb_games.get(q, 0) >= 24],
        key=lambda x: -x[1],
    )
    top_qbs = qb_items[:12]
    bottom_qbs = qb_items[-12:] if len(qb_items) > 24 else qb_items[12:]
    bottom_qb_rank_start = len(qb_items) - len(bottom_qbs) if bottom_qbs else 0

    # Per-season rows (sorted by year ascending for the table)
    per_season_rows = []
    for s in sorted(state["per_season"]):
        st = state["per_season"][s]
        if st.get("n"):
            per_season_rows.append({
                "season": s, "n": st["n"],
                "accuracy": st["accuracy"],
                "log_loss": st["log_loss"],
                "brier": st["brier"],
            })

    sample_preds = state["predictions"][-24:]
    return render_template_string(
        NFL_BACKTEST_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("nfl"),
        nav=render_sport_nav("nfl", "backtest"),
        state=state, m=m,
        top_elo=top_elo, bottom_elo=bottom_elo,
        top_qbs=top_qbs, bottom_qbs=bottom_qbs,
        bottom_qb_rank_start=bottom_qb_rank_start,
        per_season_rows=per_season_rows,
        sample_preds=sample_preds,
        calibration_svg=render_calibration_svg(m.get("calibration") or []),
    )


# ============================================================================
# Soccer backtest (EPL / La Liga / Liga MX)
# ============================================================================

SOCCER_BACKTEST_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; {{ state.league_name }} &mdash; Backtest</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero-block { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero-block h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero-block .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.grid-metrics {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 24px;
}
.metric {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.metric .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.metric .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 24px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.metric .value.accent { color: var(--accent); }
.metric .value.good { color: var(--good); text-shadow: var(--ev-strong-glow); }
.metric .foot { margin-top: 6px; font-size: 11.5px; color: var(--muted); font-family: "JetBrains Mono", monospace; }

.section {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
}
.section h2 {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 500; font-size: 20px; margin: 0 0 10px;
  letter-spacing: -0.01em;
}
.section .lead { color: var(--muted); margin: 0 0 16px; font-size: 13px; line-height: 1.6; max-width: 760px; }

.data-table {
  width: 100%; border-collapse: collapse;
  font-family: "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums; font-size: 12.5px;
}
.data-table th, .data-table td {
  padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: right;
}
.data-table th:first-child, .data-table td:first-child { text-align: left; }
.data-table th {
  color: var(--muted); font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; font-weight: 500;
}
.elo-cols { display: grid; grid-template-columns: 1fr; gap: 20px; }
@media (min-width: 900px) { .elo-cols { grid-template-columns: 1fr 1fr; } }

.pred-correct { color: var(--good); font-weight: 600; }
.pred-wrong { color: var(--muted-2); }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      {{ nav|safe }}
    </div>
    <div class="controls">
      <a class="btn" href="?refresh=1">Refit</a>
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero-block">
    <h1>{{ state.league_name }} &middot; 12-season backtest</h1>
    <p class="sub">
      3-way Elo model (home / draw / away) with margin-of-victory damping and a soccer-grade home-field
      advantage (+{{ state.hyperparams.HFA|int }} Elo &approx; ~0.4 goals). Fit chronologically across
      {{ state.total_matches }} matches spanning <strong>{{ state.first_season }}/{{ state.first_season + 1 }}&ndash;{{ state.last_season }}/{{ state.last_season + 1 }}</strong>
      ({{ state.num_seasons }} seasons). Team Elo regresses 1/3 toward 1500 at each season boundary.
      The first {{ state.warmup_seasons }} seasons are pure warm-up; the remaining
      <strong>{{ state.last_season - state.scored_from_season + 1 }} seasons are scored</strong> below.
      Data from football-data.co.uk.
    </p>
  </div>

  <div class="grid-metrics">
    <div class="metric">
      <div class="label">Accuracy</div>
      <div class="value accent">{{ '%.1f'|format(m.accuracy * 100) }}%</div>
      <div class="foot">home-only baseline {{ '%.1f'|format(m.home_baseline_accuracy * 100) }}%</div>
    </div>
    <div class="metric">
      <div class="label">Log loss</div>
      <div class="value">{{ '%.4f'|format(m.log_loss) }}</div>
      <div class="foot">baseline {{ '%.4f'|format(m.home_baseline_log_loss) }} &middot; uniform 1.0986</div>
    </div>
    <div class="metric">
      <div class="label">Brier</div>
      <div class="value">{{ '%.4f'|format(m.brier_score) }}</div>
      <div class="foot">3-way per-outcome</div>
    </div>
    <div class="metric">
      <div class="label">Scored matches</div>
      <div class="value">{{ m.n }}</div>
      <div class="foot">from {{ state.scored_from_season }}/{{ state.scored_from_season + 1 }}</div>
    </div>
    <div class="metric">
      <div class="label">Home / Draw / Away</div>
      <div class="value" style="font-size:15px;line-height:1.4">
        {{ (m.home_rate * 100)|round(1) }}% / {{ (m.draw_rate * 100)|round(1) }}% / {{ (m.away_rate * 100)|round(1) }}%
      </div>
      <div class="foot">empirical this window</div>
    </div>
    <div class="metric">
      <div class="label">Hyperparams</div>
      <div class="value" style="font-size:14px;line-height:1.4">
        K={{ state.hyperparams.K|int }} &middot; HFA=+{{ state.hyperparams.HFA|int }}
      </div>
      <div class="foot">draw factor {{ state.hyperparams.DRAW_FACTOR }}</div>
    </div>
  </div>

  <div class="section">
    <h2>Per-season breakdown</h2>
    <p class="lead">
      3-way soccer prediction is harder than 2-way sports &mdash; random guessing is 33%, home-only baseline sits
      around 45%. A sharp Elo model lands 50-55% depending on the league's parity.
      <strong>DC</strong> columns show Dixon-Coles (joint-pmf goal model with low-score correction) run
      in parallel &mdash; typically a touch better on calibration (log loss), close on accuracy.
      Elo stays the primary 3-way predictor; DC powers BTTS, totals, and the Monte Carlo tab.
    </p>
    <div class="table-scroll">
      <table class="data-table">
        <thead><tr>
          <th>Season</th><th>N</th><th>Elo Acc</th><th>DC Acc</th>
          <th>Elo LL</th><th>DC LL</th><th>Brier</th>
        </tr></thead>
        <tbody>
          {% for r in per_season_rows %}
          <tr>
            <td>{{ r.season }}/{{ r.season + 1 }}</td>
            <td>{{ r.n }}</td>
            <td>{{ (r.accuracy * 100)|round(2) }}%</td>
            <td>{% if r.accuracy_dc is not none %}{{ (r.accuracy_dc * 100)|round(2) }}%{% else %}&mdash;{% endif %}</td>
            <td>{{ '%.4f'|format(r.log_loss) }}</td>
            <td>{% if r.log_loss_dc is not none %}{{ '%.4f'|format(r.log_loss_dc) }}{% else %}&mdash;{% endif %}</td>
            <td>{{ '%.4f'|format(r.brier) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Final team Elo</h2>
    <p class="lead">End-of-window ratings. Updates after every scored match; regresses 1/3 toward 1500 between seasons.</p>
    <div class="elo-cols">
      <div>
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th>Elo</th></tr></thead>
          <tbody>
            {% for team, elo in top_elo %}
            <tr>
              <td style="color:var(--muted)">{{ loop.index }}</td>
              <td>{{ team }}</td>
              <td>{{ elo|round|int }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      <div>
        <table class="data-table">
          <thead><tr><th>#</th><th>Team</th><th>Elo</th></tr></thead>
          <tbody>
            {% for team, elo in bottom_elo %}
            <tr>
              <td style="color:var(--muted)">{{ loop.index + top_elo|length }}</td>
              <td>{{ team }}</td>
              <td>{{ elo|round|int }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Sample of scored predictions</h2>
    <p class="lead">Last 24 scored matches.</p>
    <table class="data-table">
      <thead><tr>
        <th>Date</th><th>Match</th><th>Model (H / D / A)</th><th>Result</th><th></th>
      </tr></thead>
      <tbody>
      {% for p in sample_preds %}
        <tr>
          <td>{{ p.date }}</td>
          <td>{{ p.home }} vs {{ p.away }}</td>
          <td>{{ (p.p_home * 100)|round|int }}% / {{ (p.p_draw * 100)|round|int }}% / {{ (p.p_away * 100)|round|int }}%</td>
          <td>{{ p.hg }}&ndash;{{ p.ag }} ({{ p.result }})</td>
          <td>
            {% set pred = 'H' if p.p_home >= p.p_draw and p.p_home >= p.p_away else ('D' if p.p_draw >= p.p_away else 'A') %}
            {% if pred == p.result %}<span class="pred-correct">&check;</span>
            {% else %}<span class="pred-wrong">&times;</span>{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <footer>
    <div>Fit on {{ state.total_matches }} matches &middot; data: football-data.co.uk</div>
    <div>last refit {{ state.generated_at }}</div>
  </footer>
</main>
</body>
</html>
"""


_SOCCER_BACKTEST_SLUGS = {"epl", "laliga", "ligamx"}


@app.route("/sport/<slug>/backtest")
def soccer_backtest(slug):
    """Soccer backtest route. NFL has its own route defined separately."""
    if slug == "nfl":
        from flask import redirect
        return redirect("/sport/nfl/backtest", code=307)
    if slug not in _SOCCER_BACKTEST_SLUGS:
        from flask import abort
        return abort(404)

    import soccer_model
    try:
        if request.args.get("refresh"):
            state = soccer_model.get_or_run_backtest(slug, refresh=True)
        else:
            state = soccer_model.get_or_run_backtest(slug)
    except Exception:
        state = None

    if not state:
        from flask import abort
        return abort(500)

    m = state.get("metrics") or {}

    elo_items = sorted(state["final_elo"].items(), key=lambda kv: -kv[1])
    top_n = 10
    top_elo = elo_items[:top_n]
    bottom_elo = elo_items[top_n:top_n * 2]

    per_season_rows = []
    # JSON loading turns year keys into strings — coerce back to int.
    for s_key in sorted(state["per_season"], key=lambda x: int(x)):
        st = state["per_season"][s_key]
        if st.get("n"):
            per_season_rows.append({
                "season": int(s_key), "n": st["n"],
                "accuracy": st["accuracy"],
                "log_loss": st["log_loss"],
                "brier": st["brier"],
                "accuracy_dc": st.get("accuracy_dc"),
                "log_loss_dc": st.get("log_loss_dc"),
            })

    sample_preds = state["predictions"][-24:]

    return render_template_string(
        SOCCER_BACKTEST_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip(slug),
        nav=render_sport_nav(slug, "backtest"),
        state=state, m=m,
        top_elo=top_elo, bottom_elo=bottom_elo,
        per_season_rows=per_season_rows,
        sample_preds=sample_preds,
    )


# ============================================================================
# Soccer Monte Carlo (Dixon-Coles joint-pmf sampler)
# ============================================================================

SOCCER_MC_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; {{ sport_name }} &mdash; Monte Carlo</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}
.hero-block { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero-block h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px); line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px;
}
.hero-block .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }
.sim-card {
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  padding: 18px; margin-bottom: 16px;
}
.sim-head {
  display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  flex-wrap: wrap; margin-bottom: 14px;
}
.sim-matchup { font-family: "Fraunces", Georgia, serif; font-size: 18px; }
.sim-time    { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted); }
.sim-grid {
  display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}
.sim-cell {
  background: var(--bg); border: 1px solid var(--rule); border-radius: 8px;
  padding: 10px 12px;
}
.sim-cell .label {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
}
.sim-cell .value {
  font-family: "JetBrains Mono", monospace;
  font-size: 20px; font-weight: 500; color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.sim-cell .value.good  { color: var(--good);  text-shadow: var(--ev-strong-glow); }
.sim-cell .value.bad   { color: var(--bad); }
.sim-cell .value.accent{ color: var(--accent); }
.scores-row {
  display: grid; gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
  margin-top: 14px;
}
.score-chip {
  background: var(--bg); border: 1px solid var(--rule); border-radius: 6px;
  padding: 8px; text-align: center;
  font-family: "JetBrains Mono", monospace; font-size: 12px;
}
.score-chip .s { font-size: 16px; color: var(--ink); display: block; margin-bottom: 2px; }
.score-chip .p { color: var(--accent); }
.controls { display: flex; gap: 8px; margin-bottom: 20px; align-items: center; }
.controls select, .controls .btn {
  background: var(--card); border: 1px solid var(--rule);
  padding: 6px 10px; border-radius: 6px; color: var(--ink);
  font-family: "JetBrains Mono", monospace; font-size: 12px;
}
</style>
</head>
<body>
{{ sport_strip|safe }}
<main class="container">
  {{ nav|safe }}
  <div class="hero-block">
    <h1>{{ sport_name }} &mdash; Monte Carlo</h1>
    <p class="sub">
      {{ n_sims|int }}-trial Dixon-Coles simulation per match. Goal rates come
      from team season averages (where available); Pinnacle supplies the game list.
      Each match samples from the joint goal-count pmf with low-score correction
      (ρ = {{ rho }}).
      {% if fallback_warning %}
      <br><strong>Note:</strong> this sport has no historical CSV &mdash; results use
      league-prior goal rates as a fallback.
      {% endif %}
    </p>
  </div>

  <form class="controls" method="get" action="{{ page_url }}">
    <label style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted);">
      Trials:
    </label>
    <select name="n" onchange="this.form.submit()">
      {% for opt in [1000, 5000, 10000, 25000] %}
      <option value="{{ opt }}" {{ 'selected' if opt == n_sims else '' }}>{{ opt }}</option>
      {% endfor %}
    </select>
  </form>

  {% if not sims %}
  <div class="sim-card"><h2 style="margin:0">No upcoming {{ sport_name }} games.</h2></div>
  {% endif %}

  {% for s in sims %}
  <div class="sim-card">
    <div class="sim-head">
      <div class="sim-matchup">{{ s.away }} <span style="color:var(--muted)">at</span> {{ s.home }}</div>
      <div class="sim-time">{{ (s.start_time|ct) if s.start_time else '' }}</div>
    </div>
    <div class="sim-grid">
      <div class="sim-cell">
        <div class="label">{{ s.home_short }} win</div>
        <div class="value {{ 'good' if s.sim.home_win_pct >= 60 else '' }}">{{ '%.1f' % s.sim.home_win_pct }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">Draw</div>
        <div class="value">{{ '%.1f' % s.sim.draw_pct }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">{{ s.away_short }} win</div>
        <div class="value {{ 'good' if s.sim.away_win_pct >= 60 else '' }}">{{ '%.1f' % s.sim.away_win_pct }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">BTTS Yes</div>
        <div class="value accent">{{ '%.1f' % s.sim.btts_yes_pct }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">Over 2.5</div>
        <div class="value">{{ '%.1f' % s.sim.totals.over_2_5 }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">Over 3.5</div>
        <div class="value">{{ '%.1f' % s.sim.totals.over_3_5 }}%</div>
      </div>
      <div class="sim-cell">
        <div class="label">Expected Total</div>
        <div class="value">{{ '%.2f' % s.sim.expected_total }}</div>
      </div>
      <div class="sim-cell">
        <div class="label">Projected score</div>
        <div class="value">{{ '%.2f' % s.sim.avg_home_goals }} - {{ '%.2f' % s.sim.avg_away_goals }}</div>
      </div>
    </div>
    <div class="scores-row">
      {% for sc in s.sim.most_likely_scores[:6] %}
      <div class="score-chip">
        <span class="s">{{ sc.score }}</span>
        <span class="p">{{ '%.1f' % sc.pct }}%</span>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</main>
{{ theme_script|safe }}
</body>
</html>
"""


@app.route("/sport/<slug>/montecarlo")
def soccer_montecarlo(slug):
    """Soccer Monte Carlo page — Dixon-Coles joint-pmf sampler per match."""
    sport = sports.by_slug(slug)
    if not sport or sport.get("dedicated") or slug not in _SOCCER_SLUGS_ALL:
        from flask import abort
        return abort(404)

    try:
        n_sims = int(request.args.get("n", 10000))
    except ValueError:
        n_sims = 10000
    n_sims = max(500, min(50000, n_sims))

    import soccer_model
    try:
        # MC page only needs the game list — skip Odds API fetch to save credits.
        games = generic_odds.build_sport_games(sport, use_us_books=False)
    except Exception:
        games = []

    # League slug for the DC model's goal rates — fall back to EPL prior for
    # international/UCL/Europa where we have no historical CSV.
    model_slug = slug if slug in _SOCCER_BACKTEST_SLUGS else "epl"
    sims = []
    for g in games:
        try:
            sim = soccer_model.simulate_match(
                g["home_name"], g["away_name"], model_slug, n=n_sims,
            )
        except Exception:
            continue
        if not sim:
            continue
        sims.append({
            "home": g["home_name"],
            "away": g["away_name"],
            "home_short": g["home_name"].split()[-1][:12],
            "away_short": g["away_name"].split()[-1][:12],
            "start_time": g.get("start_time"),
            "sim": sim,
        })

    return render_template_string(
        SOCCER_MC_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        theme_script=THEME_SCRIPT,
        sport_strip=render_sport_strip(slug),
        nav=render_sport_nav(slug, "montecarlo"),
        sport_name=sport["name"],
        page_url=f"/sport/{slug}/montecarlo",
        n_sims=n_sims,
        rho=soccer_model.DC_RHO_DEFAULT,
        fallback_warning=(slug not in _SOCCER_BACKTEST_SLUGS),
        sims=sims,
    )


# ============================================================================
# Monte Carlo simulator (MLB)
# ============================================================================

MONTECARLO_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Monte Carlo &mdash; {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px; font-variation-settings: "opsz" 144;
}
.hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.summary { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 24px; }
.summary .metric { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; }
.summary .label { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.summary .value { font-family: "JetBrains Mono", monospace; font-size: 22px; font-weight: 500; color: var(--ink); font-variant-numeric: tabular-nums; }
.summary .value.good { color: var(--good); text-shadow: var(--ev-strong-glow); }

.mc-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 1080px) { .mc-grid { grid-template-columns: repeat(2, 1fr); } }

.mc-card {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 22px;
  display: flex; flex-direction: column; gap: 14px;
  transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
}
.mc-card:hover { border-color: var(--rule-strong); transform: translateY(-1px); box-shadow: var(--card-hover-shadow); }

.mc-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.mc-matchup { font-size: 18px; font-weight: 600; color: var(--ink); letter-spacing: -0.01em; line-height: 1.2; }
.mc-time { font-family: "JetBrains Mono", monospace; font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

.lambda-row { display: flex; gap: 20px; font-family: "JetBrains Mono", monospace; font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted); }
.lambda-row b { color: var(--ink); font-weight: 500; }

.mc-section { border-top: 1px dashed var(--rule); padding-top: 12px; }
.mc-section-label { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }

.mc-table {
  width: 100%; border-collapse: collapse;
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  font-variant-numeric: tabular-nums;
}
.mc-table th {
  text-align: right; padding: 4px 6px;
  color: var(--muted); font-weight: 500;
  font-size: 9.5px; letter-spacing: 0.08em; text-transform: uppercase;
  border-bottom: 1px solid var(--rule);
}
.mc-table th:first-child { text-align: left; }
.mc-table td { padding: 6px; text-align: right; color: var(--ink); }
.mc-table td:first-child { text-align: left; color: var(--muted); }
.mc-table td.pick { color: var(--ink); font-weight: 500; }
.mc-table td.edge-pos { color: var(--good); font-weight: 600; text-shadow: var(--ev-strong-glow); }
.mc-table td.edge-neg { color: var(--muted-2); }

.dist-row {
  display: grid; grid-template-columns: repeat(10, 1fr);
  gap: 2px; align-items: end;
  height: 48px; margin-top: 6px;
}
.dist-bar { background: color-mix(in oklab, var(--accent) 60%, transparent); border-radius: 2px 2px 0 0; }
.dist-labels { display: grid; grid-template-columns: repeat(10, 1fr); gap: 2px; font-family: "JetBrains Mono", monospace; font-size: 9px; color: var(--muted); text-align: center; margin-top: 2px; }

.mc-no-sim { font-size: 12.5px; color: var(--muted-2); font-style: italic; }

.sim-controls { display: flex; align-items: center; gap: 10px; margin: 0 0 20px; flex-wrap: wrap; }
.sim-controls label { font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; gap: 8px; }
.sim-controls select {
  height: 36px; padding: 0 10px;
  border: 1px solid var(--rule-strong); background: var(--surface); color: var(--ink);
  border-radius: 8px; font-family: inherit; font-size: 13px;
}

.empty-mc { padding: 60px 24px; text-align: center; color: var(--muted); }
.empty-mc h2 { font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400; font-size: 24px; color: var(--ink); margin: 0 0 8px; }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab active" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab" href="/market">Market</a>
        <a class="nav-tab" href="/backtest">Model</a>
      </nav>
    </div>
    <form class="controls" method="get" action="/montecarlo">
      <a class="btn icon" href="/montecarlo?date={{ prev_date }}&amp;n={{ n_sims }}">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()">
      <a class="btn icon" href="/montecarlo?date={{ next_date }}&amp;n={{ n_sims }}">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/montecarlo?date={{ today }}&amp;n={{ n_sims }}">Today</a>{% endif %}
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>Monte Carlo &middot; {{ date_pretty }}</h1>
    <p class="sub">
      Draw <strong>{{ '{:,}'.format(n_sims) }}</strong> independent Poisson samples per game using
      projected scoring rates (team RPG blended with opposing SP ERA and bullpen quality).
      Compute empirical probabilities for every market and compare them to Pinnacle's devigged fair.
      <br><br>
      <em>Caveat:</em> this is the same scoring model we already use analytically &mdash; the simulation
      gives you the full outcome distribution, not a sharper probability. On run-line and total markets
      the model can show large edges (10%+) when it disagrees with the market; Pinnacle has more
      information (injuries, lineups, sharp money) so treat those as model-view, not guaranteed EV.
      Live and finished games have their edge calcs suppressed because the market has moved for state.
    </p>
  </div>

  <div class="sim-controls">
    <label>
      Simulations
      <select onchange="window.location.href='/montecarlo?date={{ date_str }}&amp;n='+this.value">
        {% for n in [1000, 5000, 10000, 25000, 50000] %}
        <option value="{{ n }}" {% if n == n_sims %}selected{% endif %}>{{ '{:,}'.format(n) }}</option>
        {% endfor %}
      </select>
    </label>
    <span style="color:var(--muted-2); font-size:11px">fit time ~{{ fit_time_ms }} ms total</span>
  </div>

  <div class="summary">
    <div class="metric">
      <div class="label">Games simulated</div>
      <div class="value">{{ games_simulated }}</div>
    </div>
    <div class="metric">
      <div class="label">+Edge markets</div>
      <div class="value {% if positive_count %}good{% endif %}">{{ positive_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Strong (&ge;2%)</div>
      <div class="value {% if strong_count %}good{% endif %}">{{ strong_count }}</div>
    </div>
    <div class="metric">
      <div class="label">Avg total projected</div>
      <div class="value">{{ '%.2f'|format(avg_total) if avg_total is not none else '—' }}</div>
    </div>
  </div>

  {% if games %}
  <div class="mc-grid">
    {% for g in games %}
    <article class="mc-card">
      <div class="mc-head">
        <div class="mc-matchup">{{ g.away }} <span style="font-family:Fraunces,serif;font-style:italic;color:var(--muted)">at</span> {{ g.home }}</div>
        <div class="mc-time">
          {% if g.live %}
          <span class="meta-chip status" style="margin-right:6px">{{ g.status or 'Live' }} &middot; edges off</span>
          {% endif %}
          {{ g.first_pitch }}
        </div>
      </div>
      {% if g.sim %}
      <div class="lambda-row">
        <span>&lambda;<sub>away</sub> <b>{{ '%.2f'|format(g.sim.lambda_away) }}</b></span>
        <span>&lambda;<sub>home</sub> <b>{{ '%.2f'|format(g.sim.lambda_home) }}</b></span>
        <span>exp total <b>{{ '%.2f'|format(g.sim.mean_total) }}</b></span>
        <span>extras <b>{{ (g.sim.extras_pct * 100)|round(1) }}%</b></span>
      </div>

      <div class="mc-section">
        <div class="mc-section-label">Market vs Simulation</div>
        <table class="mc-table">
          <thead>
            <tr>
              <th>Market</th>
              <th>Sim %</th>
              <th>Fair</th>
              <th>Pin price</th>
              <th>Edge</th>
            </tr>
          </thead>
          <tbody>
            {% for row in g.rows %}
            <tr>
              <td class="pick">{{ row.pick }}</td>
              <td>{{ '%.1f'|format(row.sim_pct) }}%</td>
              <td>{{ row.sim_fair_am }}</td>
              <td>
                {% if row.pin_am is not none %}
                  {{ ('+' if row.pin_am > 0 else '') ~ row.pin_am }}
                {% else %}&mdash;{% endif %}
              </td>
              <td class="{% if row.edge_pct is not none and row.edge_pct > 0 %}edge-pos{% else %}edge-neg{% endif %}">
                {% if row.edge_pct is not none %}
                  {{ '%+.2f'|format(row.edge_pct) }}%
                {% else %}&mdash;{% endif %}
              </td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>

      <div class="mc-section">
        <div class="mc-section-label">Simulated Total Runs Distribution</div>
        <div class="dist-row">
          {% for bar in g.total_hist %}
          <div class="dist-bar" style="height: {{ bar.height }}%" title="Total {{ bar.value }}: {{ bar.pct }}%"></div>
          {% endfor %}
        </div>
        <div class="dist-labels">
          {% for bar in g.total_hist %}<span>{{ bar.value }}</span>{% endfor %}
        </div>
      </div>
      {% else %}
      <div class="mc-no-sim">Missing team or pitcher stats &mdash; can't project &lambda; for this matchup.</div>
      {% endif %}
    </article>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-mc">
    <h2>No MLB games to simulate.</h2>
    <p>Try a different date.</p>
  </div>
  {% endif %}

  <footer>
    <div>{{ games_simulated }} games &middot; {{ '{:,}'.format(n_sims) }} sims each &middot; projection: team RPG &times; opp SP ERA + bullpen</div>
    <div>fit {{ fit_time_ms }} ms</div>
  </footer>
</main>
</body>
</html>
"""


def _american_from_prob(p):
    if p is None or p <= 0 or p >= 1:
        return None
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


@app.route("/mlb/montecarlo")
def mlb_montecarlo():
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()
    try:
        n_sims = int(request.args.get("n") or 10000)
    except ValueError:
        n_sims = 10000
    n_sims = max(1000, min(50000, n_sims))

    try:
        games = get_games(date_str)
    except Exception:
        games = []

    today = datetime.now(CENTRAL).date().isoformat()

    t_start = time.time()
    out_games = []
    positive_count = 0
    strong_count = 0
    totals_mean_sum = 0.0
    totals_mean_n = 0

    IN_PROGRESS_STATUSES = {"In Progress", "Delayed", "Delayed Start", "Suspended", "Final",
                             "Game Over", "Postponed", "Completed Early"}

    for g in games:
        ctx = g.get("game_ctx") or {}
        sim = mlb_model.simulate_from_game_ctx(ctx, n_sims=n_sims,
                                                seed=g.get("game_pk"))

        # Pinnacle odds for a live/finished game won't match a full-game sim.
        # Keep the simulation (useful preview) but suppress edge calcs there.
        game_live = g.get("status") in IN_PROGRESS_STATUSES

        rows = []
        total_hist = []
        if sim:
            totals_mean_sum += sim["mean_total"]
            totals_mean_n += 1

            pin = (g.get("odds") or {}).get("pinnacle") or {}
            # Devig Pinnacle probabilities for fair comparison
            def _pin_devig_2(am_h, am_a):
                if am_h is None or am_a is None:
                    return None, None
                ph = 1.0 / mlb_odds.american_to_decimal(am_h) if mlb_odds.american_to_decimal(am_h) else None
                pa = 1.0 / mlb_odds.american_to_decimal(am_a) if mlb_odds.american_to_decimal(am_a) else None
                if ph is None or pa is None:
                    return None, None
                s = ph + pa
                return ph / s, pa / s

            # ---- MONEYLINE ----
            ml = pin.get("moneyline") or {}
            pin_home_fair, pin_away_fair = _pin_devig_2(ml.get("home_am"), ml.get("away_am"))
            for side, sim_p, pin_am, pin_fair, label in [
                ("home", sim["p_home"], ml.get("home_am"), pin_home_fair, g["home"]["team"]),
                ("away", sim["p_away"], ml.get("away_am"), pin_away_fair, g["away"]["team"]),
            ]:
                edge = None
                if pin_am and sim_p and not game_live:
                    dec = mlb_odds.american_to_decimal(pin_am)
                    if dec:
                        edge = (sim_p * dec - 1) * 100
                rows.append({
                    "pick": label,
                    "sim_pct": sim_p * 100,
                    "sim_fair_am": _signed(_american_from_prob(sim_p)),
                    "pin_am": pin_am,
                    "edge_pct": edge,
                })

            # ---- TOTAL ----
            tot = pin.get("total") or {}
            if tot.get("line") is not None:
                line = tot["line"]
                p_over = sim["p_over_fn"](line)
                p_under = 1 - p_over
                pin_over_fair, pin_under_fair = _pin_devig_2(tot.get("over_am"), tot.get("under_am"))
                for side, sim_p, pin_am, label in [
                    ("over", p_over, tot.get("over_am"), f"Over {line}"),
                    ("under", p_under, tot.get("under_am"), f"Under {line}"),
                ]:
                    edge = None
                    if pin_am and sim_p:
                        dec = mlb_odds.american_to_decimal(pin_am)
                        if dec:
                            edge = (sim_p * dec - 1) * 100
                    rows.append({
                        "pick": label,
                        "sim_pct": sim_p * 100,
                        "sim_fair_am": _signed(_american_from_prob(sim_p)),
                        "pin_am": pin_am,
                        "edge_pct": edge,
                    })

            # ---- RUN LINE -1.5 ----
            rl = pin.get("run_line") or {}
            if rl.get("home_am") is not None or rl.get("away_am") is not None:
                p_home_cov = sim["p_margin_ge_fn"](2)
                p_away_cov = 1 - p_home_cov
                for side, sim_p, pin_am, label in [
                    ("home", p_home_cov, rl.get("home_am"), f"{g['home']['team']} -1.5"),
                    ("away", p_away_cov, rl.get("away_am"), f"{g['away']['team']} +1.5"),
                ]:
                    edge = None
                    if pin_am and sim_p:
                        dec = mlb_odds.american_to_decimal(pin_am)
                        if dec:
                            edge = (sim_p * dec - 1) * 100
                    rows.append({
                        "pick": label,
                        "sim_pct": sim_p * 100,
                        "sim_fair_am": _signed(_american_from_prob(sim_p)),
                        "pin_am": pin_am,
                        "edge_pct": edge,
                    })

            for r in rows:
                if r["edge_pct"] is not None:
                    if r["edge_pct"] > 0:
                        positive_count += 1
                    if r["edge_pct"] >= 2:
                        strong_count += 1

            # Histogram of totals (bins 4..13)
            lo, hi = 4, 13
            counts = [0] * (hi - lo + 1)
            for t in sim["totals"]:
                if t < lo:
                    counts[0] += 1
                elif t > hi:
                    counts[-1] += 1
                else:
                    counts[t - lo] += 1
            max_count = max(counts) or 1
            for i, c in enumerate(counts):
                pct = c / sim["n_sims"] * 100
                total_hist.append({
                    "value": lo + i,
                    "pct": round(pct, 1),
                    "height": int(round(c / max_count * 100)),
                })

        out_games.append({
            "away": g["away"]["team"],
            "home": g["home"]["team"],
            "first_pitch": g["first_pitch"],
            "status": g.get("status"),
            "live": game_live,
            "sim": sim,
            "rows": rows,
            "total_hist": total_hist,
        })

    fit_time_ms = int((time.time() - t_start) * 1000)
    avg_total = (totals_mean_sum / totals_mean_n) if totals_mean_n else None
    date_pretty = d.strftime("%A, %B %d").replace(" 0", " ")

    return render_template_string(
        MONTECARLO_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        date_str=date_str,
        date_pretty=date_pretty,
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        games=out_games,
        n_sims=n_sims,
        games_simulated=sum(1 for g in out_games if g["sim"]),
        positive_count=positive_count,
        strong_count=strong_count,
        avg_total=avg_total,
        fit_time_ms=fit_time_ms,
    )


def _signed(am):
    if am is None:
        return "—"
    return f"+{am}" if am > 0 else str(am)


# ============================================================================
# ============================================================================
# AI Analyst (/analyst) — Claude-powered structured analysis per game
# ============================================================================

ANALYST_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; AI Analyst &mdash; {{ date_pretty }}</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}

.hero { padding-bottom: 20px; margin-bottom: 24px; border-bottom: 1px solid var(--rule); }
.hero h1 { font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400; font-size: clamp(32px, 5vw, 48px); line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 8px; font-variation-settings: "opsz" 144; }
.hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.6; }

.needs-key {
  background: color-mix(in oklab, var(--warn) 10%, var(--card));
  border: 1px solid color-mix(in oklab, var(--warn) 40%, var(--rule));
  border-radius: 10px; padding: 16px 20px; margin-bottom: 24px;
}
.needs-key strong { color: var(--ink); }
.needs-key code { background: var(--surface); padding: 2px 6px; border-radius: 3px; font-size: 11.5px; }

.game-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 900px)  { .game-grid { grid-template-columns: repeat(2, 1fr); } }

.ag-card {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 12px; padding: 20px 22px;
  display: flex; flex-direction: column; gap: 14px;
}
.ag-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.ag-matchup { font-size: 18px; font-weight: 600; color: var(--ink); letter-spacing: -0.01em; line-height: 1.2; }
.ag-time { font-family: "JetBrains Mono", monospace; font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

.ag-stats {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); font-variant-numeric: tabular-nums;
  display: flex; flex-wrap: wrap; gap: 12px;
}
.ag-stats b { color: var(--ink); font-weight: 500; }

.ag-btn {
  align-self: flex-start;
  padding: 8px 16px; border-radius: 8px;
  border: 1px solid var(--accent);
  background: color-mix(in oklab, var(--accent) 10%, var(--surface));
  color: var(--accent); font-weight: 600;
  font-size: 13px; cursor: pointer;
  transition: background 120ms ease, box-shadow 120ms ease;
}
.ag-btn:hover {
  background: color-mix(in oklab, var(--accent) 20%, var(--surface));
  box-shadow: 0 0 14px var(--accent-glow);
}
.ag-btn:disabled { opacity: 0.5; cursor: wait; }

.ag-analysis {
  border-top: 1px dashed var(--rule); padding-top: 14px;
  font-size: 13px; line-height: 1.55;
  display: none;
}
.ag-analysis.visible { display: block; }
.ag-analysis h4 {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); margin: 10px 0 6px; font-weight: 500;
}
.ag-analysis ul { margin: 0 0 8px; padding-left: 20px; }
.ag-analysis li { margin-bottom: 4px; color: var(--ink); }
.ag-pick {
  background: color-mix(in oklab, var(--good) 15%, transparent);
  box-shadow: inset 3px 0 0 var(--good);
  padding: 10px 14px; border-radius: 6px; margin-top: 10px;
}
.ag-pick .label { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.ag-pick .pick-text { font-weight: 600; font-size: 15px; color: var(--ink); margin-top: 4px; }
.ag-pick .pick-rationale { font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height: 1.4; }
.ag-pick .conf { font-family: "JetBrains Mono", monospace; color: var(--good); text-shadow: var(--ev-strong-glow); }
.ag-error { color: var(--accent); font-style: italic; font-size: 12.5px; }

.loading { color: var(--muted); font-style: italic; }
.gen-foot { font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--muted-2); margin-top: 6px; letter-spacing: 0.08em; text-transform: uppercase; }
</style>
</head>
<body>
<header>
  <div class="wrap header-row">
    <div class="brand" style="display: flex; align-items: baseline;">
      <span class="brand-mark" aria-hidden="true"></span>
      <span class="brand-name">Betting Tools</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/mlb/schedule">Schedule</a>
        <a class="nav-tab" href="/montecarlo">Monte Carlo</a>
        <a class="nav-tab active" href="/analyst">AI Analyst</a>
      </nav>
    </div>
    <form class="controls" method="get" action="/analyst">
      <a class="btn icon" href="/analyst?date={{ prev_date }}">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()">
      <a class="btn icon" href="/analyst?date={{ next_date }}">&rsaquo;</a>
      {% if not is_today %}<a class="btn" href="/analyst?date={{ today }}">Today</a>{% endif %}
    </form>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>AI analyst &middot; {{ date_pretty }}</h1>
    <p class="sub">
      Claude reviews each matchup against the model, the Pinnacle market, pitcher lines, and context,
      then returns a structured take: stat read, matchup factors, risk flags, and a pick with confidence (1&ndash;3 stars).
      Each analysis is cached for 6 hours. One Claude call per matchup; costs about $0.01&ndash;0.03 per game at Sonnet pricing.
    </p>
  </div>

  {% if not key_available %}
  <div class="needs-key">
    <strong>Needs setup:</strong> this page calls Anthropic's API. Set <code>ANTHROPIC_API_KEY</code> in
    Render Settings &rarr; Environment, then redeploy. You can get a key at
    <a href="https://console.anthropic.com" style="color:var(--accent)">console.anthropic.com</a>
    &mdash; usage is pay-as-you-go, no monthly minimum.
  </div>
  {% endif %}

  {% if not games %}
  <div class="empty-mc">
    <h2>No MLB games on this date.</h2>
    <p>Try a different date.</p>
  </div>
  {% else %}
  <div class="game-grid">
    {% for g in games %}
    <article class="ag-card" data-game-pk="{{ g.game_pk }}">
      <div class="ag-head">
        <div class="ag-matchup">{{ g.away }} <span style="font-family:Fraunces,serif;font-style:italic;color:var(--muted)">at</span> {{ g.home }}</div>
        <div class="ag-time">{{ g.first_pitch }}</div>
      </div>
      <div class="ag-stats">
        <span>Model: <b>{{ (g.p_home * 100)|round|int if g.p_home else '?' }}%</b> home</span>
        <span>{{ g.away }} {{ g.away_rec }}</span>
        <span>{{ g.home }} {{ g.home_rec }}</span>
      </div>
      <button class="ag-btn" onclick="analyzeGame({{ g.game_pk }}, this)">Analyze</button>
      <div class="ag-analysis" id="analysis-{{ g.game_pk }}"></div>
    </article>
    {% endfor %}
  </div>
  {% endif %}

  <footer>
    <div>Analyses generated by Claude Sonnet &middot; cached 6h per game</div>
    <div>updated {{ now }}</div>
  </footer>
</main>

<script>
const CURRENT_DATE = {{ date_str|tojson }};
async function analyzeGame(pk, btn) {
  const target = document.getElementById('analysis-' + pk);
  if (btn) { btn.disabled = true; btn.textContent = 'Thinking…'; }
  target.classList.add('visible');
  target.innerHTML = '<div class="loading">Claude is reviewing the matchup…</div>';
  try {
    const res = await fetch('/analyst/api/game?pk=' + pk + '&date=' + encodeURIComponent(CURRENT_DATE));
    const data = await res.json();
    target.innerHTML = renderAnalysis(data);
  } catch (e) {
    target.innerHTML = '<div class="ag-error">Fetch error: ' + e + '</div>';
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Re-analyze'; }
  }
}
function esc(s){ return (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function renderAnalysis(d) {
  if (d.error) {
    return '<div class="ag-error">' + esc(d.error) + '</div>'
      + (d.raw ? '<pre style="white-space:pre-wrap;color:var(--muted-2);font-size:11px">' + esc(d.raw) + '</pre>' : '');
  }
  if (d.narrative) {
    return '<div style="font-size:12.5px;line-height:1.55;white-space:pre-wrap;color:var(--ink)">'
         + esc(d.narrative) + '</div>'
         + (d.partial ? '<div style="color:var(--muted);font-size:10px;margin-top:8px;letter-spacing:0.08em;text-transform:uppercase">partial response — re-run for a full structured read</div>' : '');
  }
  const pick = d.pick || {};
  const stars = '★'.repeat(pick.confidence || 1) + '☆'.repeat(3 - (pick.confidence || 1));
  const sideLabel = (pick.side || '').replace(/^./, c => c.toUpperCase());
  return `
    <h4>Stat Read</h4>
    <ul>${(d.stat_read||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <h4>Matchup Factors</h4>
    <ul>${(d.matchup_factors||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <h4>Risk Flags</h4>
    <ul>${(d.risk_flags||[]).map(b => '<li>' + esc(b) + '</li>').join('')}</ul>
    <div class="ag-pick">
      <div class="label">Pick · <span class="conf">${stars}</span></div>
      <div class="pick-text">${esc(pick.market || '?')} — ${esc(sideLabel)}</div>
      <div class="pick-rationale">${esc(pick.rationale || '')}</div>
    </div>
    <div class="gen-foot">${esc(d.model||'claude')} · ${esc(d.generated_at||'now')}</div>
  `;
}
</script>
</body>
</html>
"""


@app.route("/mlb/analyst")
def mlb_analyst_page():
    import analyst as analyst_mod
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(CENTRAL).date()
        date_str = d.isoformat()
    try:
        games = get_games(date_str)
    except Exception:
        games = []

    def _rec(side):
        w, l = side.get("wins"), side.get("losses")
        if w is None or l is None:
            return ""
        return f"{w}-{l}"

    simple = [{
        "game_pk": g["game_pk"],
        "away": g["away"]["team"],
        "home": g["home"]["team"],
        "first_pitch": g["first_pitch"],
        "p_home": g.get("p_home"),
        "away_rec": _rec(g["away"]),
        "home_rec": _rec(g["home"]),
    } for g in games if g.get("game_pk")]

    today = datetime.now(CENTRAL).date().isoformat()
    return render_template_string(
        ANALYST_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        games=simple,
        key_available=analyst_mod.is_available(),
        date_str=date_str,
        date_pretty=d.strftime("%A, %B %d").replace(" 0", " "),
        prev_date=(d - timedelta(days=1)).isoformat(),
        next_date=(d + timedelta(days=1)).isoformat(),
        today=today,
        is_today=(date_str == today),
        now=datetime.now(CENTRAL).strftime("%I:%M %p CT").lstrip("0"),
    )


@app.route("/analyst/api/game")
def analyst_api_game():
    import analyst as analyst_mod
    pk_str = request.args.get("pk")
    date_str = request.args.get("date") or datetime.now(CENTRAL).date().isoformat()
    try:
        pk = int(pk_str) if pk_str else None
    except ValueError:
        pk = None
    if not pk:
        return Response(json.dumps({"error": "missing pk"}), mimetype="application/json")
    try:
        games = get_games(date_str)
    except Exception:
        games = []
    game = next((g for g in games if g.get("game_pk") == pk), None)
    if not game:
        return Response(json.dumps({"error": "game not found on this date"}),
                        mimetype="application/json")
    try:
        result = analyst_mod.analyze_game(game)
    except Exception as e:
        result = {"error": f"analyst error: {e}"}
    return Response(json.dumps(result), mimetype="application/json")


# ============================================================================
# Unified Monte Carlo + AI Analysis (three primary nav sections)
# ============================================================================

def _mc_sports_options():
    """Return list of sports available in the unified MC/AI pages."""
    return [{"slug": s["slug"], "name": s["name"]} for s in sports.SPORTS]


def _in_central_date(iso_str, date_str):
    """True if the UTC iso timestamp falls on date_str in Central time."""
    if not iso_str:
        return False
    try:
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(CENTRAL).date().isoformat() == date_str
    except Exception:
        return False


# Back-compat alias so any lingering call sites still work during the switch.
_in_eastern_date = _in_central_date


_MC_GAMES_CACHE = {}
_MC_GAMES_TTL_S = 300
_MC_GAMES_CACHE_MAX = 20  # cap to prevent unbounded growth on long-running workers


def _mc_fetch_games(sport_slug, date_str):
    """Return [{id, home, away, start_time, ctx}] for sport's games on date."""
    key = (sport_slug, date_str)
    now = time.time()
    hit = _MC_GAMES_CACHE.get(key)
    if hit and now - hit[1] < _MC_GAMES_TTL_S:
        return hit[0]
    # Evict expired or oldest entries to keep the cache bounded.
    if len(_MC_GAMES_CACHE) >= _MC_GAMES_CACHE_MAX:
        stale = [k for k, v in _MC_GAMES_CACHE.items() if now - v[1] > _MC_GAMES_TTL_S]
        for k in stale:
            _MC_GAMES_CACHE.pop(k, None)
        while len(_MC_GAMES_CACHE) >= _MC_GAMES_CACHE_MAX:
            # drop oldest
            oldest = min(_MC_GAMES_CACHE, key=lambda k: _MC_GAMES_CACHE[k][1])
            _MC_GAMES_CACHE.pop(oldest, None)

    out = []
    if sport_slug == "mlb":
        try:
            games = get_games(date_str)
        except Exception:
            games = []
        for g in games:
            if not g.get("game_pk"):
                continue
            out.append({
                "id": str(g["game_pk"]),
                "home": g["home"]["team"],
                "away": g["away"]["team"],
                "start_time": g.get("first_pitch"),
                "ctx": g,
            })
    else:
        sport = sports.by_slug(sport_slug)
        if not sport or sport.get("dedicated"):
            _MC_GAMES_CACHE[key] = (out, now)
            return out
        try:
            # MC/AI unified dropdown — skip Odds API; the Pinnacle game list
            # is all we need to render the sport's games for the date.
            games = generic_odds.build_sport_games(sport, use_us_books=False)
        except Exception:
            games = []
        for g in games:
            st = g.get("start_time") or ""
            if not _in_eastern_date(st, date_str):
                continue
            gid = g.get("matchup_id") or f"{g.get('home_name','')}vs{g.get('away_name','')}"
            out.append({
                "id": str(gid),
                "home": g.get("home_name", ""),
                "away": g.get("away_name", ""),
                "start_time": st,
                "ctx": g,
            })
    _MC_GAMES_CACHE[key] = (out, now)
    return out


def _mc_team_stats(sport_slug, game_ctx):
    """Return {home_name, away_name, rows:[{label, home, away, group?}]} for the stats panel.

    Deeper per-sport stats so the panel matches the depth of
    bettingtools.ai's reference layout. Rows are grouped by `group` label so
    the template can render section headings.
    """
    if sport_slug == "mlb":
        g = game_ctx
        h = g["home"]; a = g["away"]
        hp = g.get("home_pitcher") or {}
        ap = g.get("away_pitcher") or {}
        return {
            "home_name": h["team"],
            "away_name": a["team"],
            "rows": [
                # Record
                {"group": "Record", "label": "Overall",   "home": f"{h.get('wins',0)}-{h.get('losses',0)}",
                                                           "away": f"{a.get('wins',0)}-{a.get('losses',0)}"},
                {"group": "Record", "label": "Streak",    "home": h.get('streak_code','—'), "away": a.get('streak_code','—')},
                {"group": "Record", "label": "Run diff",  "home": h.get('run_diff','—'),    "away": a.get('run_diff','—')},
                # Scoring
                {"group": "Scoring",    "label": "Runs / Game",
                 "home": f"{h.get('rpg'):.2f}" if h.get('rpg') is not None else "—",
                 "away": f"{a.get('rpg'):.2f}" if a.get('rpg') is not None else "—"},
                {"group": "Scoring",    "label": "OPS",            "home": h.get('ops','—'),      "away": a.get('ops','—')},
                # Pitching
                {"group": "Pitching",   "label": "Team ERA",       "home": h.get('team_era','—'), "away": a.get('team_era','—')},
                {"group": "Pitching",   "label": "Bullpen ERA",    "home": h.get('bullpen_era','—'), "away": a.get('bullpen_era','—')},
                # Starting pitcher
                {"group": "Starting Pitcher", "label": "SP",       "home": hp.get('name','—'),    "away": ap.get('name','—')},
                {"group": "Starting Pitcher", "label": "Record",   "home": hp.get('wl','—'),      "away": ap.get('wl','—')},
                {"group": "Starting Pitcher", "label": "ERA",      "home": hp.get('era','—'),     "away": ap.get('era','—')},
                {"group": "Starting Pitcher", "label": "WHIP",     "home": hp.get('whip','—'),    "away": ap.get('whip','—')},
                {"group": "Starting Pitcher", "label": "K / 9",    "home": hp.get('k9','—'),      "away": ap.get('k9','—')},
                {"group": "Starting Pitcher", "label": "IP",       "home": hp.get('ip','—'),      "away": ap.get('ip','—')},
            ],
        }
    if sport_slug in {"epl", "laliga", "ligamx", "ucl", "europa", "international"}:
        import soccer_model
        model_slug = sport_slug if sport_slug in {"epl","laliga","ligamx"} else "epl"
        try:
            rates = soccer_model.get_team_goal_rates(model_slug)
            # Lightweight final-Elo lookup (reads ~5KB side-file, not the ~1.4MB
            # full backtest cache) to keep memory off the Render free tier floor.
            elo = soccer_model.get_final_elo(model_slug) or {}
        except Exception:
            rates, elo = {}, {}
        h_name = game_ctx.get("home_name", "")
        a_name = game_ctx.get("away_name", "")
        hr = rates.get(h_name) or {}
        ar = rates.get(a_name) or {}
        h_elo = round(elo.get(h_name, 1500), 1)
        a_elo = round(elo.get(a_name, 1500), 1)
        h_gs = hr.get('gs_per_match', 0) or 0
        h_ga = hr.get('ga_per_match', 0) or 0
        a_gs = ar.get('gs_per_match', 0) or 0
        a_ga = ar.get('ga_per_match', 0) or 0
        h_gd = round(h_gs - h_ga, 2); a_gd = round(a_gs - a_ga, 2)
        # Fetch Pinnacle fair probabilities if a 3-way ML is present
        fair = game_ctx.get("fair") or {}
        return {
            "home_name": h_name,
            "away_name": a_name,
            "rows": [
                {"group": "Rating",   "label": "Elo rating",       "home": h_elo, "away": a_elo},
                {"group": "Rating",   "label": "Elo diff vs opp",  "home": round(h_elo - a_elo, 1),
                                                                     "away": round(a_elo - h_elo, 1)},
                {"group": "Scoring",  "label": "Goals / match",    "home": round(h_gs, 2), "away": round(a_gs, 2)},
                {"group": "Scoring",  "label": "Conceded / match", "home": round(h_ga, 2), "away": round(a_ga, 2)},
                {"group": "Scoring",  "label": "Goal differential","home": h_gd, "away": a_gd},
                {"group": "Season",   "label": "Matches played",   "home": hr.get('matches', 0), "away": ar.get('matches', 0)},
                {"group": "Market",   "label": "Pinnacle fair",
                 "home": f"{round((fair.get('home') or 0) * 100, 1)}%" if fair.get('home') else "—",
                 "away": f"{round((fair.get('away') or 0) * 100, 1)}%" if fair.get('away') else "—"},
            ],
        }
    if sport_slug == "nfl":
        try:
            import nfl_model
            # Lightweight final-state lookup; avoids loading the full
            # multi-season predictions list (hundreds of KB).
            state = nfl_model.get_final_state()
            final_elo = state.get("final_elo", {}) if state else {}
            final_qb = state.get("final_qb_elo", {}) if state else {}
        except Exception:
            final_elo, final_qb = {}, {}
        h_name = game_ctx.get("home_name", "")
        a_name = game_ctx.get("away_name", "")
        try:
            import nfl_model
            h = nfl_model.abbr_from_name(h_name) or h_name
            a = nfl_model.abbr_from_name(a_name) or a_name
        except Exception:
            h, a = h_name, a_name
        h_elo = round(final_elo.get(h, 1500), 1)
        a_elo = round(final_elo.get(a, 1500), 1)
        diff = (h_elo + 65) - a_elo
        edge = diff / 25.0
        proj_h = round(22.5 + edge / 2, 1)
        proj_a = round(22.5 - edge / 2, 1)
        fair = game_ctx.get("fair") or {}
        return {
            "home_name": h_name,
            "away_name": a_name,
            "rows": [
                {"group": "Rating",   "label": "Team Elo",         "home": h_elo, "away": a_elo},
                {"group": "Rating",   "label": "QB Elo",
                 "home": round(final_qb.get(h, 1500), 1) if final_qb else "—",
                 "away": round(final_qb.get(a, 1500), 1) if final_qb else "—"},
                {"group": "Rating",   "label": "Elo diff (incl. HFA)", "home": round(diff, 1), "away": round(-diff, 1)},
                {"group": "Scoring",  "label": "Projected points",  "home": proj_h, "away": proj_a},
                {"group": "Scoring",  "label": "Projected margin",  "home": round(proj_h - proj_a, 1),
                                                                     "away": round(proj_a - proj_h, 1)},
                {"group": "Market",   "label": "Pinnacle fair",
                 "home": f"{round((fair.get('home') or 0) * 100, 1)}%" if fair.get('home') else "—",
                 "away": f"{round((fair.get('away') or 0) * 100, 1)}%" if fair.get('away') else "—"},
            ],
        }
    if sport_slug == "ncaaf":
        try:
            import cfb_model
            elo = cfb_model.get_or_fit_team_elo()
        except Exception:
            elo = {}
        h_name = game_ctx.get("home_name", "")
        a_name = game_ctx.get("away_name", "")
        h_elo = round(elo.get(h_name, cfb_model.INITIAL_ELO), 1) if elo else 1500.0
        a_elo = round(elo.get(a_name, cfb_model.INITIAL_ELO), 1) if elo else 1500.0
        try:
            import cfb_model
            proj_h, proj_a = cfb_model.project_points(h_name, a_name)
        except Exception:
            proj_h = proj_a = 28.0
        fair = game_ctx.get("fair") or {}
        return {
            "home_name": h_name,
            "away_name": a_name,
            "rows": [
                {"group": "Rating",   "label": "Team Elo",         "home": h_elo, "away": a_elo},
                {"group": "Rating",   "label": "Elo diff (incl. HFA)",
                 "home": round(h_elo + 65 - a_elo, 1), "away": round(a_elo - h_elo - 65, 1)},
                {"group": "Scoring",  "label": "Projected points", "home": round(proj_h, 1), "away": round(proj_a, 1)},
                {"group": "Scoring",  "label": "Projected margin", "home": round(proj_h - proj_a, 1),
                                                                    "away": round(proj_a - proj_h, 1)},
                {"group": "Market",   "label": "Pinnacle fair",
                 "home": f"{round((fair.get('home') or 0) * 100, 1)}%" if fair.get('home') else "—",
                 "away": f"{round((fair.get('away') or 0) * 100, 1)}%" if fair.get('away') else "—"},
            ],
        }
    if sport_slug == "nhl":
        try:
            import nhl_model
            stats = nhl_model.get_or_fetch_team_stats()
            elo = nhl_model.get_or_fit_final_elo()
        except Exception:
            stats, elo = {}, {}
        h_name = game_ctx.get("home_name", "")
        a_name = game_ctx.get("away_name", "")
        hs = stats.get(h_name) or {}
        as_ = stats.get(a_name) or {}
        try:
            import nhl_goalies
            goalie_adj = nhl_goalies.project_goalie_adjustments(h_name, a_name)
        except Exception:
            goalie_adj = {}
        try:
            import nhl_model
            lam_h, lam_a = nhl_model.project_lambdas(h_name, a_name, stats, elo,
                                                      goalie_adj=goalie_adj)
        except Exception:
            lam_h = lam_a = 2.95
        h_elo = round(elo.get(h_name, 1500.0), 1) if elo else 1500.0
        a_elo = round(elo.get(a_name, 1500.0), 1) if elo else 1500.0
        fair = game_ctx.get("fair") or {}
        def _f(v, d=2):
            try: return round(float(v), d)
            except (TypeError, ValueError): return "—"
        return {
            "home_name": h_name,
            "away_name": a_name,
            "rows": [
                {"group": "Rating",     "label": "Team Elo",
                 "home": h_elo, "away": a_elo},
                {"group": "Rating",     "label": "Elo diff (incl. HFA)",
                 "home": round(h_elo + 35 - a_elo, 1), "away": round(a_elo - h_elo - 35, 1)},
                {"group": "Record",     "label": "Record (W-L-OTL)",
                 "home": f"{hs.get('wins',0)}-{hs.get('losses',0)}-{hs.get('ot_losses',0)}",
                 "away": f"{as_.get('wins',0)}-{as_.get('losses',0)}-{as_.get('ot_losses',0)}"},
                {"group": "Record",     "label": "Points %",
                 "home": _f(hs.get('points_pct'), 3), "away": _f(as_.get('points_pct'), 3)},
                {"group": "Scoring",    "label": "Goals / game",
                 "home": _f(hs.get('gf_per')), "away": _f(as_.get('gf_per'))},
                {"group": "Scoring",    "label": "Goals against / game",
                 "home": _f(hs.get('ga_per')), "away": _f(as_.get('ga_per'))},
                {"group": "Scoring",    "label": "Goal differential",
                 "home": _f((hs.get('gf',0) - hs.get('ga',0)), 0),
                 "away": _f((as_.get('gf',0) - as_.get('ga',0)), 0)},
                {"group": "Special Tms","label": "Power play %",
                 "home": _f((hs.get('pp_pct') or 0) * 100, 1), "away": _f((as_.get('pp_pct') or 0) * 100, 1)},
                {"group": "Special Tms","label": "Penalty kill %",
                 "home": _f((hs.get('pk_pct') or 0) * 100, 1), "away": _f((as_.get('pk_pct') or 0) * 100, 1)},
                {"group": "Special Tms","label": "Faceoff win %",
                 "home": _f((hs.get('faceoff_pct') or 0) * 100, 1), "away": _f((as_.get('faceoff_pct') or 0) * 100, 1)},
                {"group": "Goalie",     "label": "Projected starter",
                 "home": goalie_adj.get("home_goalie") or "—",
                 "away": goalie_adj.get("away_goalie") or "—"},
                {"group": "Goalie",     "label": "Save % (season)",
                 "home": _f((goalie_adj.get("home_save_pct") or 0) * 100, 1) if goalie_adj.get("home_save_pct") else "—",
                 "away": _f((goalie_adj.get("away_save_pct") or 0) * 100, 1) if goalie_adj.get("away_save_pct") else "—"},
                {"group": "Goalie",     "label": "Opp λ scaling",
                 "home": f"{(goalie_adj.get('home_factor') or 1.0):.2f}×",
                 "away": f"{(goalie_adj.get('away_factor') or 1.0):.2f}×"},
                {"group": "Projection", "label": "Projected goals (λ)",
                 "home": _f(lam_h), "away": _f(lam_a)},
                {"group": "Market",     "label": "Pinnacle fair",
                 "home": f"{round((fair.get('home') or 0) * 100, 1)}%" if fair.get('home') else "—",
                 "away": f"{round((fair.get('away') or 0) * 100, 1)}%" if fair.get('away') else "—"},
            ],
        }
    if sport_slug == "ufc":
        import ufc_model
        h_name = game_ctx.get("home_name", "")
        a_name = game_ctx.get("away_name", "")
        a = ufc_model.get_fighter(h_name)
        b = ufc_model.get_fighter(a_name)
        p_a = ufc_model.win_prob_glicko(a, b) * 100
        return {
            "home_name": a.get("canonical_name", h_name),
            "away_name": b.get("canonical_name", a_name),
            "rows": [
                {"group": "Rating", "label": "Glicko rating",
                 "home": round(a.get("rating", 1500), 0),
                 "away": round(b.get("rating", 1500), 0)},
                {"group": "Rating", "label": "Rating deviation",
                 "home": f"±{a.get('rd', 350):.0f}",
                 "away": f"±{b.get('rd', 350):.0f}"},
                {"group": "Rating", "label": "Glicko win-prob",
                 "home": f"{p_a:.1f}%",
                 "away": f"{100 - p_a:.1f}%"},
                {"group": "Record", "label": "Pro record",
                 "home": f"{a.get('wins', 0)}-{a.get('losses', 0)}",
                 "away": f"{b.get('wins', 0)}-{b.get('losses', 0)}"},
                {"group": "Striking", "label": "Strikes landed / min",
                 "home": a.get("slpm", "—"), "away": b.get("slpm", "—")},
                {"group": "Striking", "label": "Strikes absorbed / min",
                 "home": a.get("sapm", "—"), "away": b.get("sapm", "—")},
                {"group": "Grappling", "label": "Takedown accuracy",
                 "home": f"{a.get('td_acc', 0)*100:.0f}%",
                 "away": f"{b.get('td_acc', 0)*100:.0f}%"},
                {"group": "Grappling", "label": "Takedown defense",
                 "home": f"{a.get('td_def', 0)*100:.0f}%",
                 "away": f"{b.get('td_def', 0)*100:.0f}%"},
                {"group": "Grappling", "label": "Submissions / 15 min",
                 "home": a.get("sub_avg", "—"), "away": b.get("sub_avg", "—")},
            ],
        }
    return {
        "home_name": game_ctx.get("home_name", game_ctx.get("home", "Home")),
        "away_name": game_ctx.get("away_name", game_ctx.get("away", "Away")),
        "rows": [],
    }


def _mc_edge_table(sport_slug, game_ctx, sim):
    """Build the edge-detection rows: sim prob vs book prob per market.

    Returns [{market, sim_prob, book_prob, decimal, american, book, edge_pct, kelly_pct}].
    Uses the best US book price available (via game_ctx['bets']) and falls back
    to Pinnacle. Kelly is quartered to match the rest of the site. Shares the
    same Asian-line filter the picks board uses so we don't recommend edges on
    lines that don't exist on US books (+0.0, 0.25-step, 1.75 totals, etc.).
    """
    from mlb_odds import decimal_to_american
    import picks as _picks_mod

    rows = []

    def _add(label, sim_prob_pct, decimal, book_name, book_prob_pct=None,
             market_type="ML", pick_label=None):
        """Add an edge-table row. `label` is what's displayed (e.g.
        "Finland -0.25"); `market_type` is the market family used by the
        line-validity filter (e.g. "Spread", "Total", "ML", "BTTS").
        """
        if decimal is None or decimal <= 1.0 or sim_prob_pct is None:
            return
        # Reject lines the picks board also skips (+0.0, 0.25-step Asians,
        # 1.75/2.25 totals, etc.) so the edge table only shows plays that
        # exist at DK/FD/BetMGM.
        if not _picks_mod._is_valid_pick_line({
            "market": market_type,
            "pick": pick_label or label,
            "sport_slug": sport_slug,
        }):
            return
        p = sim_prob_pct / 100.0
        edge_pts = sim_prob_pct - (book_prob_pct if book_prob_pct is not None
                                   else (100.0 / decimal))
        b = decimal - 1.0
        q = 1.0 - p
        kelly = ((p * b - q) / b) * 0.25 * 100.0 if b > 0 else 0.0
        kelly = max(0.0, kelly)
        rows.append({
            "market": label,
            "sim_prob": sim_prob_pct,
            "book_prob": book_prob_pct if book_prob_pct is not None else (100.0 / decimal),
            "decimal": decimal,
            "american": (lambda am: f"+{am}" if am and am > 0 else str(am) if am else "—")(decimal_to_american(decimal)),
            "book": book_name or "pinnacle",
            "edge_pct": edge_pts,
            "kelly_pct": kelly,
        })

    sport = sports.by_slug(sport_slug)

    # MLB uses its own odds structure
    if sport_slug == "mlb":
        o = game_ctx.get("odds") or {}
        bets = o.get("bets") or []
        best_dec = o.get("best_decimal") or {}
        best_book = o.get("best_book") or {}
        home_name = game_ctx["home"]["team"]
        away_name = game_ctx["away"]["team"]
        # ML
        for b in bets:
            if b.get("market") == "ML":
                side = b["side"]
                sim_prob = sim["p_home"] if side == "home" else sim["p_away"]
                dec = best_dec.get(side) or b.get("decimal")
                bk = best_book.get(side) or "pinnacle"
                label = f"{home_name if side == 'home' else away_name} ML"
                _add(label, sim_prob, dec, bk, market_type="ML",
                     pick_label=b.get("pick") or label)
        # Total — need to count sims over book line
        for b in bets:
            if b.get("market") == "Total":
                dec = best_dec.get(b["side"]) or b.get("decimal")
                bk = best_book.get(b["side"]) or "pinnacle"
                # Find a totals line
                tot = (o.get("pinnacle") or {}).get("total") or {}
                line = tot.get("line")
                if line is None: continue
                over_pct = sum(1 for m, g in zip(sim.get("margins") or [], sim.get("margins") or [])
                               for _ in [None]) if False else None
                # Compute over pct from margins + mean_total heuristic
                # simpler: use "p_over_fn" if provided, else approximate
                margins = sim.get("margins") or []
                if not margins:
                    continue
                # Reconstruct totals from margins is incorrect. We used mean_total.
                # Use the raw totals list stored elsewhere if available.
        return rows

    # Soccer / NFL: Pinnacle lines live under game_ctx keys (ml/spread/total)
    bets = game_ctx.get("bets") or []
    home_name = game_ctx.get("home_name", "")
    away_name = game_ctx.get("away_name", "")

    # ML (3-way for soccer, 2-way for NFL / NHL)
    ml = game_ctx.get("ml") or {}
    for b in bets:
        if b.get("market") == "ML":
            side = b["side"]
            if side == "home":
                sim_prob = sim["p_home"]; label = f"{home_name} ML"
            elif side == "away":
                sim_prob = sim["p_away"]; label = f"{away_name} ML"
            elif side == "draw":
                sim_prob = sim["p_draw"]; label = "Draw"
            else:
                continue
            dec = b.get("book_decimal") or b.get("pin_decimal")
            bk  = b.get("book") or "pinnacle"
            _add(label, sim_prob, dec, bk, market_type="ML",
                 pick_label=b.get("pick") or label)

    # Spread / Run Line / Puck Line — estimate cover probability from the sim's margin list.
    sp = game_ctx.get("spread") or {}
    if sp.get("line_home") is not None and sim.get("margins"):
        hpt = sp["line_home"]
        margins = sim["margins"]
        n = len(margins)
        if n:
            home_cover = sum(1 for m in margins if m + hpt > 0) / n * 100
            away_cover = 100 - home_cover
            for b in bets:
                mkt = b.get("market")
                if mkt in ("Spread", "Run Line", "Puck Line"):
                    side = b["side"]
                    sim_prob = home_cover if side == "home" else away_cover
                    dec = b.get("book_decimal") or b.get("pin_decimal")
                    bk  = b.get("book") or "pinnacle"
                    pt = hpt if side == "home" else -hpt
                    sign = "+" if pt >= 0 else ""
                    name = home_name if side == "home" else away_name
                    label_text = f"{name} {sign}{pt}"
                    _add(label_text, sim_prob, dec, bk, market_type=mkt,
                         pick_label=b.get("pick") or label_text)

    # Total — need totals; derive from projected mean + margin distribution
    tot = game_ctx.get("total") or {}
    if tot.get("line") is not None and sim.get("margins"):
        line = tot["line"]
        # Reconstruct per-sim total using projected means + each margin's share.
        # Approximation: total ≈ proj_home + proj_away for most games; use mean + stochastic.
        # Since we don't have the raw per-sim totals, use expected total:
        import random
        rng = random.Random(42)
        margins = sim["margins"]
        n = len(margins)
        proj_total = sim.get("mean_total") or (sim.get("proj_home", 0) + sim.get("proj_away", 0))
        # Fall back: assume ~Normal(proj_total, sqrt(proj_total)*1.4) — rough
        import math as _m
        sigma = max(1.2, _m.sqrt(abs(proj_total)) * 1.4)
        if sport_slug == "nfl":
            sigma = 18.0  # NFL total std ≈ 18 pts
        over = sum(1 for _ in range(n) if rng.gauss(proj_total, sigma) > line) / n * 100
        under = 100 - over
        for b in bets:
            if b.get("market") == "Total":
                side = b["side"]
                sim_prob = over if side == "over" else under
                dec = b.get("book_decimal") or b.get("pin_decimal")
                bk  = b.get("book") or "pinnacle"
                label_text = f"{'Over' if side == 'over' else 'Under'} {line}"
                _add(label_text, sim_prob, dec, bk, market_type="Total",
                     pick_label=b.get("pick") or label_text)

    # BTTS (soccer)
    if sim.get("btts_yes_pct") is not None:
        for b in bets:
            if b.get("market") == "BTTS":
                side = b["side"]
                sim_prob = sim["btts_yes_pct"] if side == "yes" else (100 - sim["btts_yes_pct"])
                dec = b.get("book_decimal")
                bk  = b.get("book") or "pinnacle"
                label_text = f"BTTS {'Yes' if side == 'yes' else 'No'}"
                _add(label_text, sim_prob, dec, bk, market_type="BTTS",
                     pick_label=b.get("pick") or label_text)

    return rows


def _mc_run_simulation(sport_slug, game_ctx, n_sims):
    """Run the per-sport simulator and return a normalized result dict."""
    if sport_slug == "mlb":
        # Prefer the new plate-appearance Markov sim; fall back to the old
        # Poisson engine if the PA path errors (e.g. missing hitting_raw).
        sim = None
        engine_label = ""
        try:
            import mlb_pa_model
            sim = mlb_pa_model.simulate_from_game_ctx(game_ctx, n=n_sims)
            engine_label = sim.get("notes") or "PA Markov sim"
        except Exception:
            sim = None
        if sim is None:
            try:
                import mlb_model
                sim = mlb_model.simulate_from_game_ctx(game_ctx, n_sims=n_sims)
                engine_label = f"Poisson scoring (extras {sim['extras_pct']*100:.0f}%)"
            except Exception:
                return None
        if not sim:
            return None
        return {
            "n_sims": n_sims,
            "home_team": game_ctx["home"]["team"],
            "away_team": game_ctx["away"]["team"],
            "p_home": sim["p_home"] * 100,
            "p_away": sim["p_away"] * 100,
            "p_draw": 0.0,
            "proj_home": round(sim["lambda_home"], 2),
            "proj_away": round(sim["lambda_away"], 2),
            "mean_total": round(sim["mean_total"], 1),
            "margins": sim["margins"],
            "has_draw": False,
            "notes": engine_label,
            "sport_name": "MLB",
        }
    if sport_slug in {"epl", "laliga", "ligamx", "ucl", "europa", "international"}:
        import soccer_model
        model_slug = sport_slug if sport_slug in {"epl","laliga","ligamx"} else "epl"
        sim = None
        used_national_elo = False
        if sport_slug == "international":
            try:
                sim = soccer_model.predict_international_match(
                    game_ctx["home_name"], game_ctx["away_name"],
                    is_friendly=False,
                )
                if sim:
                    sim["home_team"] = game_ctx["home_name"]
                    sim["away_team"] = game_ctx["away_name"]
                    used_national_elo = True
            except Exception:
                sim = None
        if not sim:
            try:
                sim = soccer_model.simulate_match(
                    game_ctx["home_name"], game_ctx["away_name"], model_slug, n=n_sims
                )
            except Exception:
                sim = None
        if not sim:
            return None
        # Build a margin array from most-likely scores for histogram
        margins = []
        for sc in sim["most_likely_scores"]:
            try:
                hg, ag = sc["score"].split("-")
                k = int(sc["pct"] * n_sims / 100)
                margins.extend([int(hg) - int(ag)] * max(1, k))
            except Exception: pass
        return {
            "n_sims": n_sims,
            "home_team": sim["home_team"],
            "away_team": sim["away_team"],
            "p_home": sim["home_win_pct"],
            "p_away": sim["away_win_pct"],
            "p_draw": sim["draw_pct"],
            "proj_home": round(sim["avg_home_goals"], 2),
            "proj_away": round(sim["avg_away_goals"], 2),
            "mean_total": round(sim["expected_total"], 2),
            "btts_yes_pct": sim["btts_yes_pct"],
            "most_likely_scores": sim["most_likely_scores"],
            "margins": margins,
            "has_draw": True,
            "notes": (f"National-team Elo + Dixon-Coles (home {sim.get('home_elo'):.0f} vs away {sim.get('away_elo'):.0f}, HFA {sim.get('hfa_used'):.0f})"
                      if used_national_elo
                      else f"Dixon-Coles sampler (ρ = {soccer_model.DC_RHO_DEFAULT})"),
            "sport_name": sports.by_slug(sport_slug)["name"] if sports.by_slug(sport_slug) else sport_slug,
        }
    if sport_slug == "nfl":
        try:
            import nfl_model, nfl_margin_dist
            state = nfl_model.get_final_state()
            final_elo = state.get("final_elo", {}) if state else {}
            h_name = game_ctx.get("home_name", "")
            a_name = game_ctx.get("away_name", "")
            h = nfl_model.abbr_from_name(h_name) or h_name
            a = nfl_model.abbr_from_name(a_name) or a_name
            h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
            a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
            diff = (h_elo + 65) - a_elo
            edge = diff / 25.0
            proj_margin = edge  # expected margin, home − away
            # Totals stay Gaussian for V1 — the empirical margin distribution
            # is the piece that mattered for key-number pricing on spreads.
            proj_total = 45.0
            import random
            rng = random.Random()
            margins = []
            h_wins = a_wins = ties = 0
            for _ in range(n_sims):
                m = nfl_margin_dist.sample_margin(proj_margin, rng, sport="nfl")
                margins.append(m)
                if m == 0:
                    ties += 1
                elif m > 0:
                    h_wins += 1
                else:
                    a_wins += 1
            proj_h = (proj_total + proj_margin) / 2
            proj_a = (proj_total - proj_margin) / 2
            return {
                "n_sims": n_sims,
                "home_team": h_name,
                "away_team": a_name,
                "p_home": h_wins / n_sims * 100,
                "p_away": a_wins / n_sims * 100,
                "p_draw": ties / n_sims * 100,
                "proj_home": round(proj_h, 1),
                "proj_away": round(proj_a, 1),
                "mean_total": round(proj_total, 1),
                "margins": margins,
                "has_draw": False,
                "notes": "Empirical margin distribution (2600+ historical NFL games) shifted by Elo-projected margin",
                "sport_name": "NFL",
            }
        except Exception:
            return None
    if sport_slug == "ncaaf":
        try:
            import cfb_model
            sim = cfb_model.simulate_match(
                game_ctx.get("home_name", ""), game_ctx.get("away_name", ""), n=n_sims
            )
        except Exception:
            sim = None
        if not sim:
            return None
        note = ("Normal-distribution scoring from CFBD-fit Elo "
                "(σ ≈ 15 pts)" if sim.get("fitted") else
                "Normal-distribution scoring; CFBD_API_KEY not set — "
                "teams default to 1500 Elo (flat fallback)")
        return {
            "n_sims": n_sims,
            "home_team": sim["home_team"],
            "away_team": sim["away_team"],
            "p_home": sim["home_win_pct"],
            "p_away": sim["away_win_pct"],
            "p_draw": sim["draw_pct"],
            "proj_home": round(sim["avg_home_points"], 1),
            "proj_away": round(sim["avg_away_points"], 1),
            "mean_total": round(sim["expected_total"], 1),
            "margins": sim["margins"],
            "has_draw": False,
            "notes": note,
            "sport_name": "NCAAF",
        }
    if sport_slug == "nhl":
        try:
            import nhl_model
            sim = nhl_model.simulate_match(
                game_ctx.get("home_name", ""), game_ctx.get("away_name", ""), n=n_sims
            )
        except Exception:
            sim = None
        if not sim:
            return None
        return {
            "n_sims": n_sims,
            "home_team": sim["home_team"],
            "away_team": sim["away_team"],
            "p_home": sim["home_win_pct"],
            "p_away": sim["away_win_pct"],
            "p_draw": 0.0,
            "proj_home": round(sim["avg_home_goals"], 2),
            "proj_away": round(sim["avg_away_goals"], 2),
            "mean_total": round(sim["mean_total"], 2),
            "margins": sim["margins"],
            "most_likely_scores": sim["most_likely_scores"],
            "has_draw": False,
            "notes": (f"Poisson goals from NHL.com season stats "
                      f"(home λ={sim['lam_home']:.2f}, away λ={sim['lam_away']:.2f}, "
                      f"OT/SO rate {sim['ot_pct']:.0f}%)"),
            "sport_name": "NHL",
        }
    if sport_slug == "ufc":
        try:
            import ufc_model
            # UFC games use "home_name" / "away_name" for the fighters
            h_name = game_ctx.get("home_name", "")
            a_name = game_ctx.get("away_name", "")
            ufc_sim = ufc_model.simulate_fight(h_name, a_name, n=n_sims)
        except Exception:
            return None
        if not ufc_sim:
            return None
        return {
            "n_sims":     n_sims,
            "home_team":  ufc_sim["fighter_a"],
            "away_team":  ufc_sim["fighter_b"],
            "p_home":     ufc_sim["p_a_wins"] * 100,
            "p_away":     ufc_sim["p_b_wins"] * 100,
            "p_draw":     0.0,
            "proj_home":  "—",   # fights don't have points
            "proj_away":  "—",
            "mean_total": "—",
            "margins":    [],
            "has_draw":   False,
            "method_breakdown": {k: v * 100 for k, v in ufc_sim["method"].items()},
            "notes": (f"Glicko ratings (A {ufc_sim['a_stats'].get('rating',1500):.0f}±{ufc_sim['a_stats'].get('rd',350):.0f}, "
                      f"B {ufc_sim['b_stats'].get('rating',1500):.0f}±{ufc_sim['b_stats'].get('rd',350):.0f}) "
                      f"with striking/grappling blend (style adj {ufc_sim['style_adj']*100:+.1f}pp)"),
            "sport_name": "UFC",
        }
    return None


def _histogram(values, bucket_width=1, max_buckets=25):
    """Return [(center, pct)] bucket list from a list of numeric values."""
    if not values:
        return []
    import math as _m
    lo = min(values); hi = max(values)
    span = hi - lo if hi > lo else 1
    bw = max(1, _m.ceil(span / max_buckets)) if bucket_width is None else bucket_width
    # Centered around 0 for margin histograms — bucket by integer division
    buckets = {}
    for v in values:
        b = int(round(v / bw)) * bw
        buckets[b] = buckets.get(b, 0) + 1
    total = sum(buckets.values())
    rows = sorted(buckets.items())
    return [{"center": k, "pct": v / total * 100} for k, v in rows]


MC_UNIFIED_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Monte Carlo</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}
.mc-hero {
  padding: 24px 0 20px; margin-bottom: 10px; border-bottom: 1px solid var(--rule);
}
.mc-hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 52px);
  line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px;
}
.mc-hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

/* Narrow MC + AI pages so they breathe like the picks page */
main.mc-page, main.ai-page { max-width: 1040px; }

/* Date toggle (prev / date-input / next / Today / pretty-date) */
.date-toggle-wrap { margin: 14px 0 18px; }
.date-form {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 10px 14px;
}
.date-btn {
  background: var(--surface); border: 1px solid var(--rule-strong);
  color: var(--ink); padding: 6px 12px; border-radius: 6px;
  text-decoration: none; font-size: 12.5px;
  font-family: "JetBrains Mono", monospace;
}
.date-btn:hover { background: var(--card); border-color: var(--accent, var(--ink)); }
.date-btn.icon { padding: 6px 10px; font-size: 15px; line-height: 1; }
.date-input {
  background: var(--surface); border: 1px solid var(--rule-strong);
  color: var(--ink); padding: 6px 10px; border-radius: 6px;
  font-family: "JetBrains Mono", monospace; font-size: 12.5px;
}
.date-pretty {
  color: var(--muted); font-size: 12.5px; margin-left: 6px;
  font-family: "JetBrains Mono", monospace;
}

/* Mobile tightening across MC + AI */
@media (max-width: 680px) {
  main.mc-page, main.ai-page { padding-left: 14px; padding-right: 14px; }
  .stats-panel { grid-template-columns: 1fr !important; }
  .probs-grid { grid-template-columns: repeat(2, 1fr) !important; }
  .proj-row { grid-template-columns: 1fr !important; gap: 14px !important; }
  .scores-chips { gap: 6px; }
  .edge-table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .edge-table { min-width: 560px; }
  .date-pretty { display: none; }
  .mc-hero h1, .ai-hero h1 { font-size: 32px !important; }
  .howitworks li { font-size: 12px; line-height: 1.5; }
  .mc-form select, .mc-form input[type="number"] { max-width: 100%; }
}

.howitworks {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 16px 20px; margin: 20px 0;
}
.howitworks h3 {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 10px;
}
.howitworks ol { padding-left: 20px; margin: 0; }
.howitworks li { color: var(--ink); font-size: 13px; line-height: 1.6; margin-bottom: 6px; }
.howitworks li strong { color: var(--accent, var(--ink)); font-weight: 600; }

.edge-table {
  width: 100%; border-collapse: collapse; margin-top: 10px;
  font-family: "JetBrains Mono", monospace; font-size: 12.5px;
}
.edge-table thead th {
  text-align: left; padding: 8px 10px;
  color: var(--muted); font-weight: 500; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  border-bottom: 1px solid var(--rule-strong);
}
.edge-table td {
  padding: 10px; border-bottom: 1px solid var(--rule);
  color: var(--ink); font-variant-numeric: tabular-nums;
}
.edge-table td.market { color: var(--ink); font-weight: 500; }
.edge-table td.good   { color: var(--good); text-shadow: var(--ev-strong-glow); font-weight: 600; }
.edge-table td.bad    { color: #ef4444; }
.edge-table tr:last-child td { border-bottom: none; }

.mc-form { margin-top: 20px; }
.mc-form label {
  display: block;
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 8px;
}
.mc-form select, .mc-form input[type="number"] {
  width: 100%; max-width: 520px;
  background: var(--card); border: 1px solid var(--rule-strong);
  color: var(--ink); padding: 10px 12px; border-radius: 8px;
  font-family: "Public Sans", system-ui, sans-serif; font-size: 14px;
}
.mc-form .controls-row {
  display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; margin-top: 14px;
}
.mc-form .trials-input { width: 140px; }

.btn-run {
  background: var(--good, #22c55e); color: #052311;
  border: none; padding: 12px 24px; border-radius: 8px;
  font-family: "Public Sans", system-ui, sans-serif;
  font-weight: 600; font-size: 14px; cursor: pointer;
  margin-top: 18px;
}
.btn-run:hover { filter: brightness(1.08); }

.stats-panel {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  margin-top: 22px;
}
.stats-col {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 16px 18px;
}
.stats-col h3 {
  font-family: "Fraunces", Georgia, serif; font-style: italic;
  font-weight: 500; font-size: 20px; margin: 0 0 10px;
}
.stats-col.away { border-left: 3px solid var(--accent, #60a5fa); }
.stats-col.home { border-left: 3px solid #ef4444; }
.stats-col .col-tag {
  color: var(--muted); font-size: 10.5px;
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.14em; font-weight: 400; margin-left: 6px;
}
.stats-group {
  font-family: "JetBrains Mono", monospace;
  font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted);
  padding: 12px 0 4px; border-top: 1px solid var(--rule);
  margin-top: 6px;
}
.stats-group:first-of-type { border-top: none; padding-top: 4px; margin-top: 0; }
.stats-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 6px 0;
  font-family: "JetBrains Mono", monospace; font-size: 12.5px;
}
.stats-row .label { color: var(--muted); }
.stats-row .value { color: var(--ink); font-weight: 500; text-align: right; }

.results-section { margin-top: 32px; }
.results-section h2 {
  font-family: "Fraunces", Georgia, serif; font-style: italic;
  font-weight: 400; font-size: 28px; margin: 0 0 16px;
}

.probs-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.prob-tile {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.prob-tile .label {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px;
}
.prob-tile .value {
  font-family: "JetBrains Mono", monospace; font-size: 24px;
  color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 500;
}
.prob-tile .value.good { color: var(--good); text-shadow: var(--ev-strong-glow); }

.proj-row {
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
  text-align: center; margin: 24px 0 20px;
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 18px;
}
.proj-row .cell .val {
  font-family: "JetBrains Mono", monospace; font-size: 32px;
  font-weight: 500; color: var(--ink); display: block;
}
.proj-row .cell .lbl {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
  margin-top: 4px;
}

.hist-wrap {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 10px; padding: 20px; margin-top: 10px;
}
.hist-wrap h3 {
  font-family: "Fraunces", Georgia, serif; font-style: italic;
  font-weight: 500; font-size: 18px; margin: 0 0 4px;
}
.hist-wrap .sub {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); margin-bottom: 14px;
}
.hist-bars {
  display: flex; align-items: flex-end; gap: 2px; height: 160px; margin-bottom: 8px;
}
.hist-bar {
  flex: 1; min-width: 6px; border-radius: 3px 3px 0 0;
  background: var(--good);
}
.hist-bar.neg { background: #ef4444; }
.hist-x {
  display: flex; justify-content: space-between;
  font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--muted);
}

.scores-chips { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.score-chip {
  background: var(--bg); border: 1px solid var(--rule); border-radius: 6px;
  padding: 8px 10px; min-width: 68px; text-align: center;
  font-family: "JetBrains Mono", monospace; font-size: 11px;
}
.score-chip .s { color: var(--ink); font-size: 15px; display: block; }
.score-chip .p { color: var(--accent); }

.notes-line {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); margin-top: 16px;
}

@media (max-width: 680px) {
  .stats-panel { grid-template-columns: 1fr; }
  .proj-row { grid-template-columns: 1fr; gap: 18px; }
}
</style>
</head>
<body>
{{ sport_strip|safe }}
<main class="wrap mc-page">
  <section class="mc-hero">
    <h1>Monte Carlo Simulator</h1>
    <p class="sub">Pick a sport, choose today's game, run thousands of simulated games.
    MLB uses Poisson run-scoring from team RPG blended with starting-pitcher ERA.
    Soccer uses the Dixon-Coles joint-pmf goal model with low-score correction.
    NFL + NCAAF use normal-distribution scoring from team Elo (QB Elo blended in for NFL;
    CFBD-fit Elo for NCAAF when CFBD_API_KEY is present).
    NHL uses Poisson goals from NHL.com season team stats with OT/shootout coin-flip.</p>
  </section>

  <div class="howitworks">
    <h3>How this works</h3>
    <ol>
      <li><strong>Pick a sport</strong> below &mdash; the dropdown fills with every game
      Pinnacle has posted for today in that league.</li>
      <li><strong>Pick a game</strong> &mdash; season stats for both teams load in a two-column
      panel so you can eyeball the matchup before you simulate.</li>
      <li><strong>Hit Run Simulation</strong> &mdash; the sport-specific engine draws N
      independent games from the scoring distribution (Poisson for runs, Dixon-Coles for
      goals, normal for points). Each draw rolls a score, we tally win / draw / loss, totals,
      spreads, and BTTS, then show the empirical probabilities, projected averages, a score
      differential histogram, the most-likely scorelines, and an edge table comparing our
      sim probability against the live book price with a quarter-Kelly stake.</li>
    </ol>
  </div>

  {{ sport_chip_bar|safe }}

  <div class="date-toggle-wrap">
    <form method="get" action="/montecarlo" class="date-form">
      <input type="hidden" name="sport" value="{{ sport_slug or '' }}">
      <a class="date-btn icon" href="/montecarlo?date={{ prev_date }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}" aria-label="Previous day">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()" class="date-input">
      <a class="date-btn icon" href="/montecarlo?date={{ next_date }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}" aria-label="Next day">&rsaquo;</a>
      {% if not is_today %}<a class="date-btn" href="/montecarlo?date={{ today_str }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}">Today</a>{% endif %}
      <span class="date-pretty">{{ date_pretty }}</span>
    </form>
  </div>

  {% if not sport_slug %}
  <div class="hist-wrap"><h3 style="margin:0">Select a sport above to see games for {{ date_pretty }}.</h3></div>
  {% elif not games %}
  <div class="hist-wrap">
    <h3 style="margin:0">No {{ sport_name }} games on {{ date_pretty }}.</h3>
    <div class="sub">Pinnacle has no upcoming matchups for {{ sport_name }} on that date — try prev / next or Today.</div>
  </div>
  {% else %}
  <form class="mc-form" method="get" action="/montecarlo">
    <input type="hidden" name="sport" value="{{ sport_slug }}">
    <input type="hidden" name="date" value="{{ date_str }}">
    <label for="game-select">Select {{ sport_name }} game ({{ games|length }} on {{ date_pretty }})</label>
    <select id="game-select" name="game" onchange="this.form.submit()">
      <option value="">— choose a game —</option>
      {% for g in games %}
      <option value="{{ g.id }}" {% if game_id == g.id %}selected{% endif %}>
        {{ g.away }} at {{ g.home }}{% if g.start_time %} · {{ g.start_time|ct }}{% endif %}
      </option>
      {% endfor %}
    </select>
  </form>
  {% endif %}

  {% if team_stats and selected_game %}
  <div class="stats-panel">
    <div class="stats-col away">
      <h3>{{ team_stats.away_name }} <span class="col-tag">AWAY</span></h3>
      {% for grp in team_stats_groups %}
        {% if grp.name %}<div class="stats-group">{{ grp.name }}</div>{% endif %}
        {% for r in grp.rows %}
        <div class="stats-row"><span class="label">{{ r.label }}</span><span class="value">{{ r.away }}</span></div>
        {% endfor %}
      {% endfor %}
    </div>
    <div class="stats-col home">
      <h3>{{ team_stats.home_name }} <span class="col-tag">HOME</span></h3>
      {% for grp in team_stats_groups %}
        {% if grp.name %}<div class="stats-group">{{ grp.name }}</div>{% endif %}
        {% for r in grp.rows %}
        <div class="stats-row"><span class="label">{{ r.label }}</span><span class="value">{{ r.home }}</span></div>
        {% endfor %}
      {% endfor %}
    </div>
  </div>

  <form method="post" action="/montecarlo">
    <input type="hidden" name="sport" value="{{ sport_slug }}">
    <input type="hidden" name="game" value="{{ game_id }}">
    <input type="hidden" name="date" value="{{ date_str }}">
    <div class="controls-row">
      <div>
        <label for="n-sims">Trials</label>
        <select class="trials-input" id="n-sims" name="n_sims">
          {% for opt in [1000, 2500, 5000, 10000, 15000] %}
          <option value="{{ opt }}" {% if opt == n_sims %}selected{% endif %}>{{ '{:,}'.format(opt) }}</option>
          {% endfor %}
        </select>
      </div>
    </div>
    <button class="btn-run" type="submit">Run Simulation ({{ '{:,}'.format(n_sims) }} sims)</button>
  </form>
  {% endif %}

  {% if sim %}
  <section class="results-section">
    <h2>Simulation Results</h2>
    <div class="probs-grid">
      <div class="prob-tile">
        <div class="label">{{ sim.home_team }}{% if not sim.has_draw %} ML{% else %} win{% endif %}</div>
        <div class="value {% if sim.p_home >= 55 %}good{% endif %}">{{ '%.1f' % sim.p_home }}%</div>
      </div>
      {% if sim.has_draw %}
      <div class="prob-tile">
        <div class="label">Draw</div>
        <div class="value">{{ '%.1f' % sim.p_draw }}%</div>
      </div>
      {% endif %}
      <div class="prob-tile">
        <div class="label">{{ sim.away_team }}{% if not sim.has_draw %} ML{% else %} win{% endif %}</div>
        <div class="value {% if sim.p_away >= 55 %}good{% endif %}">{{ '%.1f' % sim.p_away }}%</div>
      </div>
      {% if sim.btts_yes_pct is defined %}
      <div class="prob-tile">
        <div class="label">BTTS Yes</div>
        <div class="value">{{ '%.1f' % sim.btts_yes_pct }}%</div>
      </div>
      {% endif %}
      <div class="prob-tile">
        <div class="label">Mean total</div>
        <div class="value">{{ sim.mean_total }}</div>
      </div>
    </div>

    <div class="proj-row">
      <div class="cell"><span class="val">{{ sim.proj_away }}</span><span class="lbl">{{ sim.away_team }} proj</span></div>
      <div class="cell"><span class="val">{{ sim.proj_home }}</span><span class="lbl">{{ sim.home_team }} proj</span></div>
      <div class="cell"><span class="val">{{ sim.mean_total }}</span><span class="lbl">Projected total</span></div>
    </div>

    {% if hist %}
    <div class="hist-wrap">
      <h3>Score Differential Distribution</h3>
      <div class="sub">{{ sim.away_team }} wins ← → {{ sim.home_team }} wins · {{ '{:,}'.format(sim.n_sims) }} trials</div>
      <div class="hist-bars">
        {% for b in hist %}
        <div class="hist-bar {% if b.center < 0 %}neg{% endif %}"
             style="height: {{ (b.pct / hist_max * 100) }}%"
             title="margin {{ b.center }}: {{ '%.1f' % b.pct }}%"></div>
        {% endfor %}
      </div>
      <div class="hist-x">
        <span>{{ hist[0].center }}</span>
        <span>0</span>
        <span>+{{ hist[-1].center }}</span>
      </div>
    </div>
    {% endif %}

    {% if sim.most_likely_scores %}
    <div class="hist-wrap">
      <h3>Most-likely scorelines</h3>
      <div class="scores-chips">
        {% for sc in sim.most_likely_scores[:8] %}
        <div class="score-chip"><span class="s">{{ sc.score }}</span><span class="p">{{ '%.1f' % sc.pct }}%</span></div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if edges %}
    <div class="hist-wrap">
      <h3>Edge Detection &amp; Kelly Criterion</h3>
      <div class="sub">Sim probability vs the best available book price. Edge is sim − book (percentage points). Kelly is quarter-Kelly.</div>
      <div class="edge-table-wrap">
      <table class="edge-table">
        <thead>
          <tr>
            <th>Market</th><th>Sim Prob</th><th>Book Prob</th>
            <th>Odds</th><th>Book</th><th>Edge</th><th>Kelly %</th>
          </tr>
        </thead>
        <tbody>
        {% for e in edges %}
          <tr>
            <td class="market">{{ e.market }}</td>
            <td>{{ '%.1f' % e.sim_prob }}%</td>
            <td>{{ '%.1f' % e.book_prob }}%</td>
            <td>{{ e.american }}</td>
            <td>{{ e.book }}</td>
            <td class="{{ 'good' if e.edge_pct > 0.5 else ('bad' if e.edge_pct < -0.5 else '') }}">
              {{ '+' if e.edge_pct > 0 else '' }}{{ '%.1f' % e.edge_pct }}%
            </td>
            <td>{% if e.kelly_pct > 0 %}{{ '%.1f' % e.kelly_pct }}%{% else %}&mdash;{% endif %}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
    {% endif %}

    <div class="notes-line">{{ sim.notes }}</div>
  </section>
  {% endif %}
</main>
{{ theme_script|safe }}
</body>
</html>
"""


@app.route("/montecarlo", methods=["GET", "POST"])
def montecarlo_unified():
    """Unified cross-sport Monte Carlo page."""
    sport_slug = (request.values.get("sport") or "").lower() or None
    game_id = request.values.get("game") or None
    try:
        n_sims = int(request.values.get("n_sims") or 10000)
    except ValueError:
        n_sims = 10000
    # Cap at 15k on prod — 50k sims produce margin lists that are MB-scale
    # per request and tip the free-tier instance over its memory limit when
    # several users hit the page at once.
    n_sims = max(500, min(15000, n_sims))
    run_sim = request.method == "POST"

    today_str = datetime.now(CENTRAL).date().isoformat()
    date_str = request.values.get("date") or today_str
    try:
        d_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d_obj = datetime.now(CENTRAL).date()
        date_str = d_obj.isoformat()
    prev_date = (d_obj - timedelta(days=1)).isoformat()
    next_date = (d_obj + timedelta(days=1)).isoformat()

    sport = sports.by_slug(sport_slug) if sport_slug else None
    sport_name = sport["name"] if sport else ""

    games = []
    selected_game = None
    team_stats = None
    team_stats_groups = None
    sim = None
    hist = None
    hist_max = 1
    edges = None

    if sport_slug:
        games = _mc_fetch_games(sport_slug, date_str)
        if game_id:
            selected_game = next((g for g in games if g["id"] == game_id), None)
            if selected_game:
                team_stats = _mc_team_stats(sport_slug, selected_game["ctx"])
                team_stats_groups = _group_stats_rows(team_stats.get("rows") or [])
                if run_sim:
                    sim = _mc_run_simulation(sport_slug, selected_game["ctx"], n_sims)
                    if sim and sim.get("margins"):
                        bw = 1 if sport_slug not in ("nfl", "ncaaf") else 3
                        hist = _histogram(sim["margins"], bucket_width=bw, max_buckets=25)
                        if hist:
                            hist_max = max(b["pct"] for b in hist) or 1
                    if sim:
                        try:
                            edges = _mc_edge_table(sport_slug, selected_game["ctx"], sim)
                        except Exception:
                            edges = None
                        # Drop the raw per-sim arrays — the template renders
                        # histograms from `hist` and the edge table from
                        # `edges`; keeping the 15k-element lists around just
                        # bloats the request's heap for the response render.
                        sim.pop("margins", None)
                        sim.pop("totals", None)
                        import gc
                        gc.collect()

    return render_template_string(
        MC_UNIFIED_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        theme_script=THEME_SCRIPT,
        sport_strip=render_sport_strip("montecarlo"),
        sport_chip_bar=render_sport_chip_bar(sport_slug, "/montecarlo", date_str),
        sport_slug=sport_slug,
        sport_name=sport_name,
        games=games,
        game_id=game_id,
        selected_game=selected_game,
        team_stats=team_stats,
        team_stats_groups=team_stats_groups,
        sim=sim,
        hist=hist,
        hist_max=hist_max,
        edges=edges,
        n_sims=n_sims,
        date_str=date_str,
        date_pretty=d_obj.strftime("%A, %B %d").replace(" 0", " "),
        today_str=today_str,
        prev_date=prev_date,
        next_date=next_date,
        is_today=(date_str == today_str),
    )


def _group_stats_rows(rows):
    """Group stat rows by their 'group' key, preserving first-seen group order."""
    grouped = {}
    order = []
    for r in rows:
        g = r.get("group") or ""
        if g not in grouped:
            grouped[g] = []
            order.append(g)
        grouped[g].append(r)
    return [{"name": g, "rows": grouped[g]} for g in order]


# ---------------------------------------------------------------------------
# AI Analysis — unified
# ---------------------------------------------------------------------------

AI_UNIFIED_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; AI Analysis</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}
.ai-hero { padding: 24px 0 20px; margin-bottom: 10px; border-bottom: 1px solid var(--rule); }
.ai-hero h1 {
  font-family: "Fraunces", Georgia, serif;
  font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 52px); line-height: 1.05;
  letter-spacing: -0.02em; margin: 0 0 8px;
}
.ai-hero .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.ai-form label {
  display: block;
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 20px 0 8px;
}
.ai-form select {
  width: 100%; max-width: 520px;
  background: var(--card); border: 1px solid var(--rule-strong);
  color: var(--ink); padding: 10px 12px; border-radius: 8px;
  font-size: 14px;
}
.btn-ai {
  background: var(--accent, #60a5fa); color: #072034;
  border: none; padding: 12px 24px; border-radius: 8px;
  font-weight: 600; font-size: 14px; cursor: pointer; margin-top: 18px;
}

.ai-result {
  background: var(--card); border: 1px solid var(--rule); border-radius: 12px;
  padding: 22px 24px; margin-top: 24px; line-height: 1.55;
}
.ai-result h2, .ai-result h3 {
  font-family: "Fraunces", Georgia, serif; font-weight: 500;
  margin-top: 18px; margin-bottom: 8px;
}
.ai-result h2 { font-size: 20px; }
.ai-result h3 { font-size: 16px; }
.ai-result ul { padding-left: 22px; margin: 6px 0 10px; }
.ai-result li { margin: 4px 0; }
.ai-foot {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); margin-top: 16px;
}
.ai-err {
  background: color-mix(in oklab, #ef4444 20%, var(--card));
  border-color: #ef4444; color: var(--ink);
}
</style>
</head>
<body>
{{ sport_strip|safe }}
<main class="wrap ai-page">
  <section class="ai-hero">
    <h1>AI Analysis</h1>
    <p class="sub">Pick a sport and today's game, Claude Sonnet 5.5 writes
    an in-depth read on the matchup &mdash; stat angle, matchup factors, risk
    flags and a lean, using live model + market data.</p>
  </section>

  <div class="howitworks">
    <h3>How this works</h3>
    <ol>
      <li><strong>Pick a sport</strong> below &mdash; the dropdown shows every game posted
      for today in that league.</li>
      <li><strong>Pick a game</strong> and hit <strong>Run Analysis</strong>. The server
      builds a sport-specific context block: team records and season stats, starting-pitcher
      line for MLB, Elo and Dixon-Coles projections for soccer, team Elo plus QB Elo for NFL,
      plus the current Pinnacle fair probabilities.</li>
      <li><strong>Claude Sonnet 5.5</strong> receives that context with a structured prompt
      and returns a 4-section markdown write-up: <em>Stat Read</em>, <em>Matchup Factors</em>,
      <em>Risk Flags</em>, <em>Lean &amp; Pick</em>. The response is rendered in-page below.</li>
    </ol>
    <p style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--muted);margin:10px 0 0">
      model: {{ 'claude-sonnet-5-5' }} &middot; requires ANTHROPIC_API_KEY on the host &middot;
      runs on-demand, not cached across games
    </p>
  </div>

  {{ sport_chip_bar|safe }}

  <div class="date-toggle-wrap">
    <form method="get" action="/ai-analysis" class="date-form">
      <input type="hidden" name="sport" value="{{ sport_slug or '' }}">
      <a class="date-btn icon" href="/ai-analysis?date={{ prev_date }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}" aria-label="Previous day">&lsaquo;</a>
      <input type="date" name="date" value="{{ date_str }}" onchange="this.form.submit()" class="date-input">
      <a class="date-btn icon" href="/ai-analysis?date={{ next_date }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}" aria-label="Next day">&rsaquo;</a>
      {% if not is_today %}<a class="date-btn" href="/ai-analysis?date={{ today_str }}{% if sport_slug %}&amp;sport={{ sport_slug }}{% endif %}">Today</a>{% endif %}
      <span class="date-pretty">{{ date_pretty }}</span>
    </form>
  </div>

  {% if not key_available %}
  <div class="ai-result ai-err">
    <h3 style="margin-top:0">ANTHROPIC_API_KEY not set</h3>
    <p>Set the environment variable on this host (or in your Render dashboard)
    and reload — this page needs it to call Claude.</p>
  </div>
  {% endif %}

  {% if sport_slug and not games %}
  <div class="ai-result"><h3 style="margin:0">No {{ sport_name }} games on {{ date_pretty }}.</h3></div>
  {% endif %}

  {% if sport_slug and games %}
  <form class="ai-form" method="post" action="/ai-analysis">
    <input type="hidden" name="sport" value="{{ sport_slug }}">
    <input type="hidden" name="date" value="{{ date_str }}">
    <label for="g">Select {{ sport_name }} game ({{ games|length }} on {{ date_pretty }})</label>
    <select id="g" name="game">
      <option value="">— choose a game —</option>
      {% for g in games %}
      <option value="{{ g.id }}" {% if game_id == g.id %}selected{% endif %}>
        {{ g.away }} at {{ g.home }}{% if g.start_time %} · {{ g.start_time|ct }}{% endif %}
      </option>
      {% endfor %}
    </select>
    <br>
    <button class="btn-ai" type="submit" {% if not key_available %}disabled{% endif %}>Run Analysis</button>
  </form>
  {% endif %}

  {% if analysis %}
  <section class="ai-result">
    {% if analysis.error %}
      <h3 style="margin-top:0;color:#ef4444">Analysis error</h3>
      <p>{{ analysis.error }}</p>
    {% else %}
      {{ analysis.html|safe }}
      <div class="ai-foot">model: {{ analysis.model }} · generated: {{ analysis.generated_at }}</div>
    {% endif %}
  </section>
  {% endif %}
</main>
{{ theme_script|safe }}
</body>
</html>
"""


def _ai_render_markdown(text):
    """Very lightweight markdown → HTML (headings, lists, paragraphs, bold)."""
    import html as _html, re as _re
    out = []
    lines = text.split("\n")
    in_list = False
    for ln in lines:
        s = ln.rstrip()
        if not s:
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        esc = _html.escape(s)
        # bold **text**
        esc = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
        esc = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", esc)
        if s.startswith("### "):
            if in_list: out.append("</ul>"); in_list = False
            out.append(f"<h3>{esc[4:]}</h3>")
        elif s.startswith("## "):
            if in_list: out.append("</ul>"); in_list = False
            out.append(f"<h2>{esc[3:]}</h2>")
        elif s.startswith("# "):
            if in_list: out.append("</ul>"); in_list = False
            out.append(f"<h2>{esc[2:]}</h2>")
        elif s.lstrip().startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{esc.lstrip()[2:]}</li>")
        else:
            if in_list: out.append("</ul>"); in_list = False
            out.append(f"<p>{esc}</p>")
    if in_list: out.append("</ul>")
    return "\n".join(out)


def _ai_analyze_matchup(sport_slug, game_ctx):
    """Call Claude for a sport-aware game analysis. Returns {html, model, generated_at}."""
    import analyst as analyst_mod
    if not analyst_mod.is_available():
        return {"error": "ANTHROPIC_API_KEY is not set in this environment."}
    try:
        import os
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        return {"error": f"anthropic SDK not available: {e}"}

    # Build sport-specific context block
    ctx_lines = []
    if sport_slug == "mlb":
        g = game_ctx
        a = g["away"]; h = g["home"]
        ap = g.get("away_pitcher") or {}; hp = g.get("home_pitcher") or {}
        ctx_lines.append(f"MLB matchup: {a['team']} at {h['team']} · first pitch {g.get('first_pitch','?')}")
        ctx_lines.append(f"- {a['team']}: {a.get('wins','?')}-{a.get('losses','?')}, OPS {a.get('ops','?')}, team ERA {a.get('team_era','?')}")
        ctx_lines.append(f"- {h['team']}: {h.get('wins','?')}-{h.get('losses','?')}, OPS {h.get('ops','?')}, team ERA {h.get('team_era','?')}")
        ctx_lines.append(f"- Away SP: {ap.get('name','?')} ({ap.get('era','?')} ERA, {ap.get('whip','?')} WHIP)")
        ctx_lines.append(f"- Home SP: {hp.get('name','?')} ({hp.get('era','?')} ERA, {hp.get('whip','?')} WHIP)")
        if g.get("p_home") is not None:
            ctx_lines.append(f"- Model prob: home {int(round(g['p_home']*100))}%")
    elif sport_slug in {"epl","laliga","ligamx","ucl","europa","international"}:
        try:
            import soccer_model
            model_slug = sport_slug if sport_slug in {"epl","laliga","ligamx"} else "epl"
            elo = soccer_model.get_final_elo(model_slug) or {}
            h_name = game_ctx.get("home_name",""); a_name = game_ctx.get("away_name","")
            # Smaller sim for the prompt context — the full MC page already
            # offers 15k; the AI prompt doesn't need that many trials.
            sim = soccer_model.simulate_match(h_name, a_name, model_slug, n=2000)
            ctx_lines.append(f"Soccer matchup: {a_name} at {h_name} · kickoff {game_ctx.get('start_time','?')}")
            ctx_lines.append(f"- Elo: home {round(elo.get(h_name,1500),1)}, away {round(elo.get(a_name,1500),1)}")
            ctx_lines.append(f"- Dixon-Coles MC: home {sim['home_win_pct']:.1f}% / draw {sim['draw_pct']:.1f}% / away {sim['away_win_pct']:.1f}%")
            ctx_lines.append(f"- Projected goals: {sim['avg_home_goals']:.2f} - {sim['avg_away_goals']:.2f} (BTTS {sim['btts_yes_pct']:.1f}%)")
        except Exception as e:
            ctx_lines.append(f"Soccer context (limited data): {game_ctx.get('home_name','?')} vs {game_ctx.get('away_name','?')}")
    elif sport_slug == "nfl":
        try:
            import nfl_model
            state = nfl_model.get_final_state()
            elo = state.get("final_elo", {}) if state else {}
            qb_elo = state.get("final_qb_elo", {}) if state else {}
            h_name = game_ctx.get("home_name",""); a_name = game_ctx.get("away_name","")
            h = nfl_model.abbr_from_name(h_name) or h_name
            a = nfl_model.abbr_from_name(a_name) or a_name
            ctx_lines.append(f"NFL matchup: {a_name} at {h_name} · kickoff {game_ctx.get('start_time','?')}")
            ctx_lines.append(f"- Team Elo: home {round(elo.get(h,1500),1)}, away {round(elo.get(a,1500),1)}")
            if qb_elo:
                ctx_lines.append(f"- QB Elo: home {round(qb_elo.get(h,1500),1)}, away {round(qb_elo.get(a,1500),1)}")
        except Exception:
            ctx_lines.append(f"NFL matchup: {game_ctx.get('away_name','?')} at {game_ctx.get('home_name','?')}")
    elif sport_slug == "ncaaf":
        try:
            import cfb_model
            elo = cfb_model.get_or_fit_team_elo()
            h_name = game_ctx.get("home_name",""); a_name = game_ctx.get("away_name","")
            proj_h, proj_a = cfb_model.project_points(h_name, a_name)
            ctx_lines.append(f"NCAAF matchup: {a_name} at {h_name} · kickoff {game_ctx.get('start_time','?')}")
            ctx_lines.append(f"- Team Elo: home {round(elo.get(h_name,1500),1)}, away {round(elo.get(a_name,1500),1)}")
            ctx_lines.append(f"- Projected points: {proj_h:.1f} - {proj_a:.1f}")
            if not cfb_model.is_fitted():
                ctx_lines.append("- NOTE: CFBD_API_KEY not set; Elo defaults to 1500 for all teams.")
        except Exception:
            ctx_lines.append(f"NCAAF matchup: {game_ctx.get('away_name','?')} at {game_ctx.get('home_name','?')}")
    elif sport_slug == "nhl":
        try:
            import nhl_model
            stats = nhl_model.get_or_fetch_team_stats()
            h_name = game_ctx.get("home_name",""); a_name = game_ctx.get("away_name","")
            hs = stats.get(h_name) or {}; as_ = stats.get(a_name) or {}
            sim = nhl_model.simulate_match(h_name, a_name, n=3000)
            ctx_lines.append(f"NHL matchup: {a_name} at {h_name} · puck drop {game_ctx.get('start_time','?')}")
            ctx_lines.append(f"- Record: home {hs.get('wins',0)}-{hs.get('losses',0)}-{hs.get('ot_losses',0)}, "
                             f"away {as_.get('wins',0)}-{as_.get('losses',0)}-{as_.get('ot_losses',0)}")
            ctx_lines.append(f"- Goals/G: home GF {hs.get('gf_per',0):.2f} GA {hs.get('ga_per',0):.2f}, "
                             f"away GF {as_.get('gf_per',0):.2f} GA {as_.get('ga_per',0):.2f}")
            ctx_lines.append(f"- PP%/PK%: home {(hs.get('pp_pct') or 0)*100:.1f}/{(hs.get('pk_pct') or 0)*100:.1f}, "
                             f"away {(as_.get('pp_pct') or 0)*100:.1f}/{(as_.get('pk_pct') or 0)*100:.1f}")
            ctx_lines.append(f"- Sim: home {sim['home_win_pct']:.1f}% / away {sim['away_win_pct']:.1f}% "
                             f"(λ home {sim['lam_home']:.2f}, away {sim['lam_away']:.2f})")
        except Exception:
            ctx_lines.append(f"NHL matchup: {game_ctx.get('away_name','?')} at {game_ctx.get('home_name','?')}")
    else:
        ctx_lines.append(f"{sport_slug.upper()} matchup: {game_ctx.get('away_name','?')} vs {game_ctx.get('home_name','?')}")

    context = "\n".join(ctx_lines)
    prompt = f"""You are an expert sports betting analyst with access to the web.
Produce the sharpest, most CURRENT read possible on this matchup. Use web search
to pull in any news from today or the last few days — injury updates, lineup /
starter changes, suspensions, weather, late line movement, public betting
splits, and recent form. Ground your analysis in what the search returns.

{context}

Write 6 sections with markdown headings (## Section):
## News & Injury Report
  — the most recent news items (reporter + source + date if you can).
  Called out injuries, questionable players, suspensions, weather, notable
  late scratches, lineup / rotation / goalie / starting-pitcher news.
## Head-to-Head History
  — recent H2H results between these two sides, home/away splits, any
  stylistic pattern (ex: team A has owned team B at home, totals trend etc).
## Play Styles & Tactical Matchup
  — how each side plays, where they create their edge, and specifically
  how those styles interact (ex: run-heavy offense vs rush defense;
  high-press team vs long-ball; power play vs penalty kill).
## Stat Read
  — pull the key numeric angle(s) from the context block above (model
  Elo / Dixon-Coles projection / Pinnacle fair) and interpret them.
## Risk Flags
  — anything that could blow up your lean: short rest, travel, trap-game
  spots, injury uncertainty, umpire / referee tendencies, late news.
## Lean & Pick
  — one clear recommended play (or "pass" if nothing is actionable).
  Give the market, the price range you'd take, and a confidence tier
  (Lean / Solid / Strong). Avoid filler.

450-650 words. Be specific, cite sources when you use them. If search returns
nothing useful, say so and work from training knowledge rather than inventing."""

    def _attempt(use_search):
        """One API call. Returns (text, stop_reason, used_search, err)."""
        k = dict(
            model=analyst_mod.CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        if use_search:
            # Anthropic server-side web search tool. If the account doesn't
            # have it enabled we catch the error and fall back.
            k["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }]
        try:
            resp = client.messages.create(**k)
        except Exception as e:
            return "", None, False, str(e)
        parts = []
        used_search = False
        for block in (resp.content or []):
            btype = getattr(block, "type", None)
            if btype == "text":
                t = getattr(block, "text", "") or ""
                if t:
                    parts.append(t)
            elif btype in ("server_tool_use", "web_search_tool_result", "tool_use"):
                used_search = True
        return "\n\n".join(parts), getattr(resp, "stop_reason", None), used_search, None

    try:
        # First try WITH web search for current news; fall back to no-tool
        # if the account can't use it, or if the response had no text block
        # (e.g. Claude only emitted tool_use/intermediate blocks before stopping).
        text, stop_reason, used_search, err = _attempt(use_search=True)
        tool_err = err
        if not text:
            text2, stop_reason2, _, err2 = _attempt(use_search=False)
            if text2:
                text = text2
                stop_reason = stop_reason2
                used_search = False
            elif err2:
                # Both attempts failed outright
                reason = tool_err or err2
                return {"error": f"Claude call failed: {reason}"}
        if not text:
            return {"error": f"Claude returned no text (stop_reason={stop_reason}, "
                              f"tool_err={tool_err or 'none'})."}
        if used_search:
            text = text + "\n\n*(web search used for current news / injuries)*"
        return {
            "html": _ai_render_markdown(text),
            "model": analyst_mod.CLAUDE_MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        return {"error": f"Claude call failed: {e}"}


@app.route("/ai-analysis", methods=["GET", "POST"])
def ai_analysis_unified():
    """Unified cross-sport AI analysis page."""
    import analyst as analyst_mod
    sport_slug = (request.values.get("sport") or "").lower() or None
    game_id = request.values.get("game") or None

    today_str = datetime.now(CENTRAL).date().isoformat()
    date_str = request.values.get("date") or today_str
    try:
        d_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d_obj = datetime.now(CENTRAL).date()
        date_str = d_obj.isoformat()
    prev_date = (d_obj - timedelta(days=1)).isoformat()
    next_date = (d_obj + timedelta(days=1)).isoformat()

    sport = sports.by_slug(sport_slug) if sport_slug else None
    sport_name = sport["name"] if sport else ""

    games = []
    analysis = None
    if sport_slug:
        games = _mc_fetch_games(sport_slug, date_str)
        if game_id and request.method == "POST":
            selected = next((g for g in games if g["id"] == game_id), None)
            if selected:
                analysis = _ai_analyze_matchup(sport_slug, selected["ctx"])

    return render_template_string(
        AI_UNIFIED_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        theme_script=THEME_SCRIPT,
        sport_strip=render_sport_strip("analysis"),
        sport_chip_bar=render_sport_chip_bar(sport_slug, "/ai-analysis", date_str),
        sport_slug=sport_slug,
        sport_name=sport_name,
        games=games,
        game_id=game_id,
        analysis=analysis,
        key_available=analyst_mod.is_available(),
        date_str=date_str,
        date_pretty=d_obj.strftime("%A, %B %d").replace(" 0", " "),
        today_str=today_str,
        prev_date=prev_date,
        next_date=next_date,
        is_today=(date_str == today_str),
    )


# ============================================================================
# Logged Plays — W/L dashboard for past picks
# ============================================================================

LOGGED_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betting Tools &mdash; Logged Plays</title>
{{ fonts_link|safe }}
<style>
{{ shared_style|safe }}
main.logged-page { max-width: 1040px; }

.hero-block { padding-bottom: 18px; margin-bottom: 20px; border-bottom: 1px solid var(--rule); }
.hero-block h1 {
  font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 400;
  font-size: clamp(32px, 5vw, 48px); line-height: 1.05; letter-spacing: -0.02em;
  margin: 0 0 8px;
}
.hero-block .sub { color: var(--muted); max-width: 760px; font-size: 13.5px; line-height: 1.55; }

.records-section { margin-bottom: 32px; }
.records-section h2 {
  font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 500;
  font-size: 22px; margin: 0 0 10px;
}
.records-section .sub {
  color: var(--muted); font-size: 12.5px; margin-bottom: 14px;
  font-family: "JetBrains Mono", monospace;
}
.records-grid {
  display: grid; gap: 10px;
  grid-template-columns: repeat(3, 1fr);
  margin-bottom: 10px;
}
.rec-tile {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 14px 16px;
}
.rec-tile.strong-tile { border-left: 3px solid var(--good); }
.rec-tile .lbl {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 6px;
}
.rec-tile .wl {
  font-family: "JetBrains Mono", monospace; font-size: 26px;
  color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 500;
}
.rec-tile .pct { font-size: 14px; color: var(--muted); margin-left: 8px; }
.rec-tile.good .wl, .rec-tile.good .units { color: var(--good); text-shadow: var(--ev-strong-glow); }
.rec-tile.bad  .wl, .rec-tile.bad  .units { color: #ef4444; }
.rec-tile .meta {
  font-family: "JetBrains Mono", monospace; font-size: 11px;
  color: var(--muted); margin-top: 6px;
}
.rec-tile .units {
  display: block; margin-top: 4px;
  font-family: "JetBrains Mono", monospace; font-size: 15px;
  color: var(--ink); font-variant-numeric: tabular-nums;
}

.persist-status {
  padding: 10px 14px; border-radius: 8px; margin-bottom: 20px;
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  line-height: 1.5; border: 1px solid var(--rule);
}
.persist-status.ok {
  background: color-mix(in oklab, var(--good) 10%, var(--card));
  border-color: var(--good); color: var(--ink);
}
.persist-status.warn {
  background: color-mix(in oklab, #f59e0b 15%, var(--card));
  border-color: #f59e0b; color: var(--ink);
}
.persist-status code {
  background: var(--surface); padding: 1px 5px; border-radius: 3px;
  font-size: 11px;
}

.section-sub {
  font-family: "JetBrains Mono", monospace;
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 20px 0 8px;
}
.insights { display: flex; flex-direction: column; gap: 6px; margin: 10px 0 18px; }
.insight {
  padding: 10px 12px; border-radius: 6px;
  font-family: "JetBrains Mono", monospace; font-size: 12px;
  line-height: 1.45; border-left: 3px solid var(--rule-strong);
  background: var(--card);
}
.insight.good  { border-left-color: var(--good); color: var(--ink); }
.insight.bad   { border-left-color: #ef4444; color: var(--ink); }
.insight.warn  { border-left-color: #f59e0b; color: var(--ink); }
.insight.info  { border-left-color: var(--accent, #60a5fa); color: var(--muted); }

.active-adj {
  background: color-mix(in oklab, var(--good) 10%, var(--card));
  border: 1px solid var(--good); border-radius: 8px;
  padding: 10px 14px; margin: 10px 0 20px;
  font-family: "JetBrains Mono", monospace; font-size: 11.5px;
  line-height: 1.6; color: var(--ink);
}
.active-adj .adj-chip {
  display: inline-block; margin: 2px 4px 2px 0;
  padding: 2px 8px; border-radius: 999px;
  background: var(--surface); border: 1px solid var(--rule-strong);
  font-size: 10.5px; color: var(--ink);
}
.active-adj .adj-chip.bad { border-color: #ef4444; color: #ef4444; }

.breakdown-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}
.breakdown-col {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 14px 16px;
}
.breakdown-col h3 {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 8px;
}
.brk-table {
  width: 100%; border-collapse: collapse;
  font-family: "JetBrains Mono", monospace; font-size: 12px;
}
.brk-table thead th {
  text-align: left; padding: 6px 4px;
  color: var(--muted); font-weight: 500; font-size: 10px;
  letter-spacing: 0.1em; text-transform: uppercase;
  border-bottom: 1px solid var(--rule-strong);
}
.brk-table td {
  padding: 6px 4px; border-bottom: 1px solid var(--rule);
  color: var(--ink); font-variant-numeric: tabular-nums; text-align: right;
}
.brk-table td.brk-key { text-align: left; color: var(--ink); font-weight: 500; }
.brk-table td.good { color: var(--good); text-shadow: var(--ev-strong-glow); }
.brk-table td.bad  { color: #ef4444; }
.brk-table tr:last-child td { border-bottom: none; }
@media (max-width: 680px) {
  .breakdown-grid { grid-template-columns: 1fr; }
}

.date-group {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 12px;
}
.date-group h3 {
  font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 500;
  font-size: 18px; margin: 0 0 10px;
}
.pick-row {
  display: grid; grid-template-columns: 68px 1fr 60px 60px 70px;
  gap: 10px; align-items: center;
  padding: 8px 0; border-top: 1px solid var(--rule);
  font-family: "JetBrains Mono", monospace; font-size: 12.5px;
}
.pick-row .clv {
  text-align: right; font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.pick-row .clv.pos { color: var(--good); }
.pick-row .clv.neg { color: #ef4444; }
.pick-row .clv .clv-lbl {
  display: inline-block; font-size: 9px; letter-spacing: 0.1em;
  color: var(--muted); margin-right: 3px;
}
.pick-row:first-of-type { border-top: none; }
.pick-row .tag {
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted);
}
.pick-row .strong-badge {
  display: inline-block; padding: 1px 6px; border-radius: 4px;
  background: var(--good); color: #052311;
  font-size: 9px; letter-spacing: 0.1em; font-weight: 700;
  margin-right: 6px;
}
.pick-row .pick-cell { color: var(--ink); }
.pick-row .pick-cell .matchup { color: var(--muted); font-size: 11px; margin-top: 2px; }
.pick-row .odd { color: var(--muted); text-align: right; }
.pick-row .res {
  text-align: center; font-weight: 600; letter-spacing: 0.1em;
  font-size: 11px; padding: 4px; border-radius: 4px;
}
.pick-row .res.W { color: var(--good); background: color-mix(in oklab, var(--good) 15%, transparent); }
.pick-row .res.L { color: #ef4444; background: color-mix(in oklab, #ef4444 15%, transparent); }
.pick-row .res.P { color: var(--muted); background: var(--surface); }
.pick-row .res.pending { color: var(--muted); font-size: 10px; }

.strong-section .pick-row { background: color-mix(in oklab, var(--good) 4%, transparent); }
.date-group .section-label {
  font-family: "JetBrains Mono", monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 14px 0 4px; display: flex; gap: 8px; align-items: baseline;
}
.date-group .section-label .count { color: var(--ink); font-weight: 500; }

@media (max-width: 680px) {
  .records-grid { grid-template-columns: 1fr; }
  .pick-row { grid-template-columns: 50px 1fr 58px; }
  .pick-row .odd, .pick-row .clv { display: none; }
  main.logged-page { padding-left: 14px; padding-right: 14px; }
}
</style>
</head>
<body>
{{ sport_strip|safe }}
<main class="wrap logged-page">
  <div class="hero-block">
    <h1>Logged Plays</h1>
    <p class="sub">Every pick the board has recommended since this feature went live,
    graded against final scores. Strong (model &geq; 60%) plays are tracked separately
    so you can see how the high-conviction tier is actually hitting versus the broader
    board. Pushes don't count toward W-L; units P/L assumes a flat 1-unit stake per
    pick at the odds shown on the board at pick time.</p>
  </div>

  <div class="persist-status {{ 'ok' if persist_status.enabled else 'warn' }}">
    {% if persist_status.enabled %}
      <strong>Persistent storage:</strong> enabled &middot;
      {{ persist_status.picks_files or 0 }} picks files &middot;
      {{ persist_status.graded_files or 0 }} graded files in
      <code>{{ persist_status.repo }}</code> (branch <code>{{ persist_status.branch }}</code>).
      Snapshots survive Render restarts.
    {% else %}
      <strong>Persistent storage disabled.</strong>
      {{ persist_status.reason or 'GITHUB_TOKEN not set.' }}
      Set <code>GITHUB_TOKEN</code> on Render (fine-grained PAT with Contents: Read &amp; Write
      on this repo) so logs survive the next spin-down.
    {% endif %}
  </div>

  <div class="persist-status {{ 'ok' if grader_status.odds_api_enabled else 'warn' }}">
    {% if grader_status.odds_api_enabled %}
      <strong>Grading:</strong> enabled &middot; Odds API scores (all sports) + MLB statsapi (free fallback).
    {% else %}
      <strong>Grading:</strong> MLB only (statsapi fallback is live).
      <code>ODDS_API_KEY</code> is not set, so NFL / NCAAF / NHL / soccer / UFC picks will stay pending
      until the key is added on Render.
    {% endif %}
  </div>

  {% if odds_usage and odds_usage.key_set %}
  <div class="persist-status ok">
    <strong>Odds API usage:</strong>
    {{ odds_usage.hour.credits or 0 }} credits in last hour &middot;
    {{ odds_usage.day.credits or 0 }} in last 24h &middot;
    {{ odds_usage.week.credits or 0 }} in last week &middot;
    cache TTL {{ odds_usage.ttl_minutes }} min &middot; markets: h2h only.
    MC + AI dropdowns do not hit the Odds API — Pinnacle covers them.
  </div>
  {% endif %}

  <section class="records-section">
    <h2>Overall Record</h2>
    <div class="sub">All graded picks across every sport. Pending = game not yet final (Odds API lags by a few minutes after games end).</div>
    <div class="records-grid">
      {% for key, label in [('all_7d','Last 7 days'),('all_30d','Last 30 days'),('all_90d','Last 90 days')] %}
      {% set r = summary[key] %}
      <div class="rec-tile {{ 'good' if r.units > 0 else ('bad' if r.units < 0 else '') }}">
        <div class="lbl">{{ label }}</div>
        {% if r.settled %}
        <div class="wl">{{ r.wins }}-{{ r.losses }}{% if r.pushes %}-{{ r.pushes }}{% endif %}<span class="pct">{{ '%.1f' % r.win_pct }}%</span></div>
        <span class="units">{{ '%+.2f' % r.units }}u ({{ '%+.1f' % r.roi_pct }}% ROI)</span>
        {% else %}
        <div class="wl" style="color:var(--muted)">&mdash;</div>
        <span class="units" style="color:var(--muted)">&mdash;</span>
        {% endif %}
        <div class="meta">{{ r.settled }} settled{% if r.pending %} &middot; {{ r.pending }} pending{% endif %}</div>
      </div>
      {% endfor %}
    </div>
  </section>

  <section class="records-section">
    <h2>Strong Picks Only <span style="color:var(--good);text-shadow:var(--ev-strong-glow);font-size:16px">★</span></h2>
    <div class="sub">Both &ge; 60% consensus AND &ge; +4% EV. Separated so you can tell if the strong tier is the real signal (the way the reference pickers package their best plays).</div>
    <div class="records-grid">
      {% for key, label in [('strong_7d','Last 7 days'),('strong_30d','Last 30 days'),('strong_90d','Last 90 days')] %}
      {% set r = summary[key] %}
      <div class="rec-tile strong-tile {{ 'good' if r.units > 0 else ('bad' if r.units < 0 else '') }}">
        <div class="lbl">{{ label }}</div>
        {% if r.settled %}
        <div class="wl">{{ r.wins }}-{{ r.losses }}{% if r.pushes %}-{{ r.pushes }}{% endif %}<span class="pct">{{ '%.1f' % r.win_pct }}%</span></div>
        <span class="units">{{ '%+.2f' % r.units }}u ({{ '%+.1f' % r.roi_pct }}% ROI)</span>
        {% else %}
        <div class="wl" style="color:var(--muted)">&mdash;</div>
        <span class="units" style="color:var(--muted)">&mdash;</span>
        {% endif %}
        <div class="meta">{{ r.settled }} settled{% if r.pending %} &middot; {{ r.pending }} pending{% endif %}</div>
      </div>
      {% endfor %}
    </div>
  </section>

  {% if summary.all_30d.clv_n or summary.all_90d.clv_n %}
  <section class="records-section">
    <h2>Closing-Line Value (CLV)</h2>
    <div class="sub">
      Change in Pinnacle's devigged probability between our FIRST snapshot of
      a pick and its latest (closing-ish) value. Positive = the sharp side of
      the market moved toward our pick after we locked in &mdash; the single
      strongest leading indicator of long-run bettor skill. Only settled picks
      count.
    </div>
    <div class="records-grid">
      {% for key, label in [('all_7d','Last 7 days'),('all_30d','Last 30 days'),('all_90d','Last 90 days')] %}
      {% set r = summary[key] %}
      <div class="rec-tile {{ 'good' if r.clv_avg_pp > 0.3 else ('bad' if r.clv_avg_pp < -0.3 else '') }}">
        <div class="lbl">{{ label }}</div>
        {% if r.clv_n %}
        <div class="wl">{{ '%+.2f' % r.clv_avg_pp }}<span class="pct">pp avg</span></div>
        <span class="units">{{ '%.0f' % r.clv_pos_pct }}% beat close &middot; {{ '%+.2f' % r.clv_avg_ev_pct }}% EV</span>
        {% else %}
        <div class="wl" style="color:var(--muted)">&mdash;</div>
        <span class="units" style="color:var(--muted)">&mdash;</span>
        {% endif %}
        <div class="meta">{{ r.clv_n }} CLV samples</div>
      </div>
      {% endfor %}
    </div>
  </section>
  {% endif %}

  {% if poly_summary and poly_summary.total_trades %}
  <section class="records-section">
    <h2>Polymarket History
      <span style="color:var(--muted);font-size:14px">
        {{ poly_summary.total_trades }} trades &middot;
        {{ poly_summary.wins }}-{{ poly_summary.losses }}
        ({{ '%.1f' % poly_summary.win_pct }}% hit) &middot;
        <span class="{{ 'good' if poly_summary.pnl > 0 else ('bad' if poly_summary.pnl < 0 else '') }}">${{ '%+.2f' % poly_summary.pnl }}</span>
      </span>
    </h2>
    <div class="sub">
      Your settled Polymarket trades, bucketed by sport + market + side.
      Daily picks that match a strong winning bucket get a green "Pattern
      match" badge; picks matching a chronically losing bucket get a red
      warning. Signals need ≥ 3 settled trades in the bucket to fire.
    </div>
    <div class="breakdown-grid">
      <div class="breakdown-col">
        <h3>Winning patterns</h3>
        <table class="brk-table">
          <thead><tr><th>Pattern</th><th>Record</th><th>Win %</th><th>PnL</th></tr></thead>
          <tbody>
          {% for r in poly_summary.top_winning %}
            <tr>
              <td class="brk-key">{{ r.key }}</td>
              <td>{{ r.wins }}-{{ r.losses }}</td>
              <td class="{{ 'good' if r.win_pct >= 55 else '' }}">{{ '%.0f' % r.win_pct }}%</td>
              <td class="good">+${{ '%.2f' % r.pnl }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="breakdown-col">
        <h3>Losing patterns</h3>
        <table class="brk-table">
          <thead><tr><th>Pattern</th><th>Record</th><th>Win %</th><th>PnL</th></tr></thead>
          <tbody>
          {% for r in poly_summary.top_losing %}
            <tr>
              <td class="brk-key">{{ r.key }}</td>
              <td>{{ r.wins }}-{{ r.losses }}</td>
              <td class="{{ 'bad' if r.win_pct < 35 else '' }}">{{ '%.0f' % r.win_pct }}%</td>
              <td class="bad">${{ '%.2f' % r.pnl }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </section>
  {% endif %}

  {% if active_adj and active_adj.any %}
  <div class="active-adj">
    <strong>Analyzer-driven adjustments applied to today's board:</strong>
    {% for sport, w in active_adj.weights.items() %}
      <span class="adj-chip">{{ sport }} weights {{ (w.pricing*100)|int }}/{{ (w.mc*100)|int }}</span>
    {% endfor %}
    {% for sport, f in active_adj.floors.items() %}
      <span class="adj-chip">{{ sport }} min prob {{ (f*100)|int }}%</span>
    {% endfor %}
    {% for m in active_adj.blacklist %}
      <span class="adj-chip bad">drop {{ m }}</span>
    {% endfor %}
  </div>
  {% endif %}

  {% if analysis %}
  <section class="records-section">
    <h2>Daily Deep Analysis
      <span style="color:var(--muted);font-size:14px">
        {{ analysis.lookback_days or 90 }}-day lookback &middot;
        {{ analysis.settled_picks or 0 }} settled picks
      </span>
    </h2>
    <div class="sub">
      Automated calibration + per-model-signal breakdown of logged picks.
      Any time pricing-model and Monte Carlo disagree on which signal is
      more honest for a sport, the analyzer suggests a weight shift and the
      picks pipeline applies it on the next refresh — self-tuning without
      a code change. Regenerates once per day; last run
      {{ analysis.generated_at or '—' }}.
    </div>

    {% if not (analysis.insights or analysis.calibration or analysis.recommended_weights) %}
    <div class="insight info">
      No settled picks yet — the analyzer is live but has nothing to score
      against. Once games finish and the logged plays grade in, this
      section will populate with calibration bins, per-model Brier scores,
      and recommended consensus weight shifts.
    </div>
    {% endif %}

    {% if analysis.insights %}
    <div class="insights">
      {% for i in analysis.insights %}
      <div class="insight {{ i.tone }}">{{ i.text }}</div>
      {% endfor %}
    </div>
    {% endif %}

    {% if analysis.calibration %}
    <h3 class="section-sub">Consensus calibration</h3>
    <table class="brk-table">
      <thead><tr><th>Bin</th><th>N</th><th>Expected</th><th>Actual</th><th>Gap</th></tr></thead>
      <tbody>
      {% for b in analysis.calibration %}
        <tr>
          <td class="brk-key">{{ (b.bin_lo * 100)|int }}&ndash;{{ (b.bin_hi * 100)|int }}%</td>
          <td>{{ b.n }}</td>
          <td>{{ '%.1f' % b.expected }}%</td>
          <td>{{ '%.1f' % b.actual }}%</td>
          <td class="{{ 'good' if b.gap_pp >= 1 else ('bad' if b.gap_pp <= -1 else '') }}">
            {{ '%+.1f' % b.gap_pp }}pp
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}

    {% if analysis.recommended_weights %}
    <h3 class="section-sub" style="margin-top:16px">Recommended consensus weights</h3>
    <table class="brk-table">
      <thead><tr><th>Sport</th><th>N</th><th>Pricing w.</th><th>MC w.</th>
                 <th>Default Brier</th><th>Rec. Brier</th><th>Δ bp</th></tr></thead>
      <tbody>
      {% for sport, w in analysis.recommended_weights.items() %}
        <tr>
          <td class="brk-key">{{ sport }}</td>
          <td>{{ w.n }}</td>
          <td>{{ (w.pricing_w * 100)|int }}%</td>
          <td>{{ (w.mc_w * 100)|int }}%</td>
          <td>{{ '%.4f' % w.brier_at_default }}</td>
          <td>{{ '%.4f' % w.brier_at_rec }}</td>
          <td class="{{ 'good' if w.improvement_bp >= 5 else '' }}">+{{ '%.1f' % w.improvement_bp }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </section>
  {% endif %}

  {% if by_sport or by_market %}
  <section class="records-section">
    <h2>Model Health <span style="color:var(--muted);font-size:14px">last 90 days</span></h2>
    <div class="sub">Which sports and markets our models are actually winning on. A sport under 48% or negative ROI over a meaningful sample is a signal to tune or drop that model.</div>
    <div class="breakdown-grid">
      <div class="breakdown-col">
        <h3>By sport</h3>
        <table class="brk-table">
          <thead><tr><th>Sport</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr></thead>
          <tbody>
          {% for r in by_sport %}
            <tr>
              <td class="brk-key">{{ r.key }}</td>
              <td>{% if r.settled %}{{ r.wins }}-{{ r.losses }}{% if r.pushes %}-{{ r.pushes }}{% endif %}{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.win_pct >= 55 and r.settled else ('bad' if r.win_pct < 48 and r.settled >= 10 else '') }}">{% if r.settled %}{{ '%.1f' % r.win_pct }}%{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.units > 0 else ('bad' if r.units < 0 else '') }}">{% if r.settled %}{{ '%+.2f' % r.units }}u{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.roi_pct > 0 else ('bad' if r.roi_pct < 0 else '') }}">{% if r.settled %}{{ '%+.1f' % r.roi_pct }}%{% else %}&mdash;{% endif %}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
      <div class="breakdown-col">
        <h3>By market</h3>
        <table class="brk-table">
          <thead><tr><th>Market</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr></thead>
          <tbody>
          {% for r in by_market %}
            <tr>
              <td class="brk-key">{{ r.key }}</td>
              <td>{% if r.settled %}{{ r.wins }}-{{ r.losses }}{% if r.pushes %}-{{ r.pushes }}{% endif %}{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.win_pct >= 55 and r.settled else ('bad' if r.win_pct < 48 and r.settled >= 10 else '') }}">{% if r.settled %}{{ '%.1f' % r.win_pct }}%{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.units > 0 else ('bad' if r.units < 0 else '') }}">{% if r.settled %}{{ '%+.2f' % r.units }}u{% else %}&mdash;{% endif %}</td>
              <td class="{{ 'good' if r.roi_pct > 0 else ('bad' if r.roi_pct < 0 else '') }}">{% if r.settled %}{{ '%+.1f' % r.roi_pct }}%{% else %}&mdash;{% endif %}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </section>
  {% endif %}

  {% if not dates %}
  <div class="date-group">
    <h3 style="margin:0">No logged picks yet.</h3>
    <p style="color:var(--muted);margin:8px 0 0;font-size:13px">
      Visit the <a href="/">Picks</a> page on any day — the day's board auto-snapshots the
      first time it's viewed, and this page starts tracking results as soon as the games finish.
    </p>
  </div>
  {% endif %}

  {% for ds in dates %}
  {% set picks = graded_by_date[ds] %}
  {% set strongs = picks | selectattr('strong') | list %}
  {% set others  = picks | rejectattr('strong')  | list %}
  <div class="date-group">
    <h3>{{ ds }}</h3>
    {% if strongs %}
    <div class="section-label">Strong picks <span class="count">({{ strongs|length }})</span></div>
    <div class="strong-section">
      {% for p in strongs %}
      <div class="pick-row">
        <span class="tag">{{ p.sport }}</span>
        <div class="pick-cell">
          <span class="strong-badge">STRONG</span><strong>{{ p.pick }}</strong>
          <div class="matchup">{{ p.away_team }} at {{ p.home_team }} &middot; {{ p.market }}
            {% if p.home_score is not none and p.away_score is not none %}
              &middot; {{ p.away_score|int }}-{{ p.home_score|int }}
            {% endif %}
          </div>
        </div>
        <span class="odd">{{ p.american }}</span>
        {% set clv = p.get('clv_pp') %}
        {% if clv is not none %}
          <span class="clv {{ 'pos' if clv > 0 else ('neg' if clv < 0 else '') }}" title="Line movement from first-save to close (pp of Pinnacle devig). Positive = sharp signal.">
            <span class="clv-lbl">CLV</span>{{ '%+.1f' % clv }}
          </span>
        {% else %}
          <span class="clv">&mdash;</span>
        {% endif %}
        <span class="res {{ p.result }}">{{ p.result or 'pending' }}</span>
      </div>
      {% endfor %}
    </div>
    {% endif %}
    {% if others %}
    <div class="section-label">Other picks <span class="count">({{ others|length }})</span></div>
    {% for p in others %}
      <div class="pick-row">
        <span class="tag">{{ p.sport }}</span>
        <div class="pick-cell">
          <strong>{{ p.pick }}</strong>
          <div class="matchup">{{ p.away_team }} at {{ p.home_team }} &middot; {{ p.market }}
            {% if p.home_score is not none and p.away_score is not none %}
              &middot; {{ p.away_score|int }}-{{ p.home_score|int }}
            {% endif %}
          </div>
        </div>
        <span class="odd">{{ p.american }}</span>
        {% set clv = p.get('clv_pp') %}
        {% if clv is not none %}
          <span class="clv {{ 'pos' if clv > 0 else ('neg' if clv < 0 else '') }}" title="Line movement from first-save to close (pp of Pinnacle devig). Positive = sharp signal.">
            <span class="clv-lbl">CLV</span>{{ '%+.1f' % clv }}
          </span>
        {% else %}
          <span class="clv">&mdash;</span>
        {% endif %}
        <span class="res {{ p.result }}">{{ p.result or 'pending' }}</span>
      </div>
    {% endfor %}
    {% endif %}
  </div>
  {% endfor %}
</main>
{{ theme_script|safe }}
</body>
</html>
"""


@app.route("/logged")
def logged_plays():
    """Dashboard of past picks with W/L records (7d / 30d / 90d, strong vs all).

    Also kicks off the daily deep analysis of logged picks (model_analytics)
    the first time it's hit each day — insights appear in-page and the picks
    pipeline reads back any recommended per-sport consensus weight shifts.
    """
    import plays_log, log_persist, model_analytics
    try:
        graded_by_date = plays_log.grade_all(limit_dates=120)
    except Exception:
        graded_by_date = {}
    summary = plays_log.summary(graded_by_date)
    by_sport  = plays_log.breakdown_by_key(graded_by_date, lambda p: p.get("sport"))
    by_market = plays_log.breakdown_by_key(graded_by_date, lambda p: p.get("market"))
    dates = sorted(graded_by_date.keys(), reverse=True)
    try:
        persist_status = log_persist.status_summary()
    except Exception as e:
        persist_status = {"enabled": False, "reason": f"status check failed: {e}"}
    try:
        grader_status = plays_log.grader_status()
    except Exception:
        grader_status = {"odds_api_enabled": False, "mlb_statsapi": True}
    try:
        import generic_odds
        odds_usage = generic_odds.odds_api_usage_summary()
    except Exception:
        odds_usage = None
    try:
        import polymarket_history
        poly_summary = polymarket_history.summary()
    except Exception:
        poly_summary = None
    # Trigger the daily analysis in a background thread (fire-and-forget) so
    # /logged never blocks on it. We still read the LATEST available analysis
    # synchronously — on first-ever visit this is None and the UI shows the
    # empty-state card; subsequent visits see yesterday's analysis until the
    # background job finishes writing today's.
    try:
        model_analytics.start_background_analysis()
        analysis = model_analytics.read_latest_analysis() or {
            "insights": [], "lookback_days": 90, "settled_picks": 0,
            "generated_at": None,
        }
    except Exception as e:
        analysis = {"error": str(e), "insights": []}
    try:
        active_adj = model_analytics.active_adjustments_summary()
    except Exception:
        active_adj = {"any": False}
    return render_template_string(
        LOGGED_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        theme_script=THEME_SCRIPT,
        sport_strip=render_sport_strip("logged"),
        summary=summary,
        by_sport=by_sport,
        by_market=by_market,
        dates=dates,
        graded_by_date=graded_by_date,
        persist_status=persist_status,
        analysis=analysis,
        active_adj=active_adj,
        grader_status=grader_status,
        odds_usage=odds_usage,
        poly_summary=poly_summary,
    )


if __name__ == "__main__":
    import os as _os
    import socket
    # Honor $PORT when it's set (Render, Heroku, Fly, Railway all inject it);
    # fall back to 5000 for local dev. Without this, a Render service that
    # uses `python mlb_ui.py` as its start command would bind to 5000 while
    # Render's port scanner probes $PORT (usually 10000+), producing the
    # "Port scan timeout, no open ports detected" deploy failure.
    port = int(_os.environ.get("PORT") or 5000)
    hostname = socket.gethostname()
    try:
        lan_ip = socket.gethostbyname(hostname)
    except OSError:
        lan_ip = None
    print("Betting Tools - cross-sport Monte Carlo + AI analysis")
    print(f"  On this PC : http://127.0.0.1:{port}")
    if lan_ip and not lan_ip.startswith("127."):
        print(f"  On phone   : http://{lan_ip}:{port}   (same Wi-Fi network)")
    print("  Backtest   : /backtest")
    app.run(debug=False, host="0.0.0.0", port=port)
