#!/usr/bin/env python3
"""First Pitch — daily MLB matchups UI + model predictions + backtest report.

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

EASTERN = ZoneInfo("America/New_York")

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
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(EASTERN)


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
    """Load fitted model state (Elo ratings + backtest results)."""
    key = ("model_state",)
    def _f():
        try:
            return mlb_model.get_or_run_backtest()
        except Exception as e:
            print(f"[warn] model unavailable: {e}", flush=True)
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
        first_pitch = dt.strftime("%I:%M %p ET").lstrip("0") if dt else "TBD"

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
            "sort_key": dt or datetime.max.replace(tzinfo=EASTERN),
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
        })
    games.sort(key=lambda g: g["sort_key"])
    return games


# ============================================================================
# HTML — shared styles + layout
# ============================================================================

SHARED_STYLE = r"""
:root {
  --bg: #F1ECDF; --surface: #FBF8F0; --card: #FFFFFF;
  --ink: #1A1613; --muted: #7A7167; --muted-2: #9A9186;
  --rule: #DED7C6; --rule-strong: #C9C1AE; --chip-bg: #F1ECDF;
  --accent: #A63329; --good: #3B7350; --warn: #B0801E; --poor: #A63329;
  --focus: #A63329; --card-shadow: 0 1px 0 rgba(26,22,19,0.04);
  --header-bg: rgba(241,236,223,0.88);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14110E; --surface: #1B1815; --card: #23201C;
    --ink: #ECE5D8; --muted: #8F8676; --muted-2: #6A6357;
    --rule: #2A2621; --rule-strong: #3B362E; --chip-bg: #1B1815;
    --accent: #D06A5F; --good: #7BB893; --warn: #D9B25E; --poor: #D06A5F;
    --focus: #D06A5F; --card-shadow: 0 1px 0 rgba(0,0,0,0.25);
    --header-bg: rgba(20,17,14,0.85);
  }
}
:root[data-theme="dark"] {
  --bg: #14110E; --surface: #1B1815; --card: #23201C;
  --ink: #ECE5D8; --muted: #8F8676; --muted-2: #6A6357;
  --rule: #2A2621; --rule-strong: #3B362E; --chip-bg: #1B1815;
  --accent: #D06A5F; --good: #7BB893; --warn: #D9B25E; --poor: #D06A5F;
  --focus: #D06A5F; --card-shadow: 0 1px 0 rgba(0,0,0,0.25);
  --header-bg: rgba(20,17,14,0.85);
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
"""

FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;1,9..144,400'
    '&family=Public+Sans:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500&display=swap">'
)


def render_sport_strip(active_slug):
    parts = ['<div class="sport-strip"><div class="wrap sport-strip-inner">']
    for sp in sports.SPORTS:
        cls = "sport-pill active" if sp["slug"] == active_slug else "sport-pill"
        href = "/" if sp["slug"] == "mlb" else f"/sport/{sp['slug']}"
        parts.append(f'<a class="{cls}" href="{href}">{sp["name"]}</a>')
    parts.append('</div></div>')
    return "".join(parts)


# ============================================================================
# main page template
# ============================================================================

INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Pitch — {{ date_pretty }}</title>
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
      <span class="brand-name">First Pitch</span>
      <nav class="nav-tabs">
        <a class="nav-tab active" href="/">Schedule</a>
        <a class="nav-tab" href="/edges">Edges</a>
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
# backtest report template
# ============================================================================

BACKTEST_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Pitch &mdash; Model &amp; Backtest</title>
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
      <span class="brand-name">First Pitch</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/">Schedule</a>
        <a class="nav-tab" href="/edges">Edges</a>
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
    <h1>The model, and how it did.</h1>
    <p class="sub">
      An Elo rating model, warm-started across the full {{ state.season }} regular season, with a
      pregame adjustment for each starting pitcher's season-to-date ERA and WHIP.
      Predictions are made <em>before</em> each game using only stats that would have been available at first pitch.
      The final {{ state.score_last_n_days }} days are scored against actual outcomes below.
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
      <div class="foot">{{ state.score_from }} &rarr; {{ state.score_to }}</div>
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

@app.route("/")
def index():
    date_str = request.args.get("date") or datetime.now(EASTERN).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(EASTERN).date()
        date_str = d.isoformat()

    error = None
    games = []
    try:
        games = get_games(date_str)
    except URLError as e:
        error = f"Could not reach MLB statsapi ({e.reason})."
    except Exception as e:
        error = f"Unexpected error: {e}"

    today = datetime.now(EASTERN).date().isoformat()
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
        now=datetime.now(EASTERN).strftime("%I:%M %p ET").lstrip("0"),
    )


@app.route("/backtest")
def backtest():
    if request.args.get("refresh"):
        _cache.pop(("model_state",), None)
        try:
            state = mlb_model.get_or_run_backtest(refresh=True)
            _cache[("model_state",)] = (state, time.time())
        except Exception as e:
            state = None
    else:
        state = get_model_state()

    if not state:
        return render_template_string(
            BACKTEST_TEMPLATE,
            fonts_link=FONTS_LINK,
            shared_style=SHARED_STYLE,
            state=None, m=None,
            top_elo=[], bottom_elo=[],
            sample_preds=[],
            calibration_svg="",
        )

    m = state.get("metrics") or mlb_model.compute_metrics(state["predictions"])

    elo_items = sorted(
        state["final_elo"].items(),
        key=lambda kv: -kv[1],
    )
    top_elo = [(tid, r, state["final_wl"].get(tid, [0, 0])) for tid, r in elo_items[:15]]
    bottom_elo = [(tid, r, state["final_wl"].get(tid, [0, 0])) for tid, r in elo_items[15:]]

    sample_preds = state["predictions"][-30:]

    return render_template_string(
        BACKTEST_TEMPLATE,
        fonts_link=FONTS_LINK,
        shared_style=SHARED_STYLE,
        sport_strip=render_sport_strip("mlb"),
        state=state,
        m=m,
        top_elo=top_elo,
        bottom_elo=bottom_elo,
        sample_preds=sample_preds,
        calibration_svg=render_calibration_svg(m.get("calibration") or []),
    )


@app.route("/export.csv")
def export_csv():
    date_str = request.args.get("date") or datetime.now(EASTERN).date().isoformat()
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        date_str = datetime.now(EASTERN).date().isoformat()

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
<title>First Pitch &mdash; Edges &mdash; {{ date_pretty }}</title>
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
      <span class="brand-name">First Pitch</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/">Schedule</a>
        <a class="nav-tab active" href="/edges">Edges</a>
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
    date_str = request.args.get("date") or datetime.now(EASTERN).date().isoformat()
    market_filter = (request.args.get("market") or "all").upper()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(EASTERN).date()
        date_str = d.isoformat()

    try:
        games = get_games(date_str)
    except Exception:
        games = []

    today = datetime.now(EASTERN).date().isoformat()
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
        now=datetime.now(EASTERN).strftime("%I:%M %p ET").lstrip("0"),
    )


# ============================================================================
# /market — Pinnacle limits + Polymarket futures + splits context
# ============================================================================

MARKET_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Pitch &mdash; Market &mdash; {{ date_pretty }}</title>
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
      <span class="brand-name">First Pitch</span>
      <nav class="nav-tabs">
        <a class="nav-tab" href="/">Schedule</a>
        <a class="nav-tab" href="/edges">Edges</a>
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
    date_str = request.args.get("date") or datetime.now(EASTERN).date().isoformat()
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        d = datetime.now(EASTERN).date()
        date_str = d.isoformat()

    try:
        games = get_games(date_str)
    except Exception:
        games = []

    today = datetime.now(EASTERN).date().isoformat()

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
        now=datetime.now(EASTERN).strftime("%I:%M %p ET").lstrip("0"),
    )


# ============================================================================
# Generic sport page (NFL, UFC, EPL, La Liga, Liga MX, UCL, Europa, Int'l)
# ============================================================================

SPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>First Pitch &mdash; {{ sport.name }}</title>
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
.bets tr.pos { background: color-mix(in oklab, var(--good) 6%, transparent); }
.bets tr.pos.strong { background: color-mix(in oklab, var(--good) 14%, transparent); }
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
      <span class="brand-name">First Pitch</span>
    </div>
  </div>
</header>

{{ sport_strip|safe }}

<main class="wrap reveal">
  <div class="hero">
    <h1>{{ sport.name }} &middot; model-free edges</h1>
    <p class="sub">
      Positive-EV plays identified by comparing each US sportsbook's price to
      <strong>Pinnacle's devigged fair probability</strong> &mdash; the industry
      consensus "true" market line. No per-sport model yet; this is pure line-shopping
      against the sharpest book.
    </p>
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
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(EASTERN)
        return dt.strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")
    except (ValueError, TypeError):
        return iso_utc


@app.route("/sport/<slug>")
def sport_page(slug):
    sport = sports.by_slug(slug)
    if not sport or sport.get("dedicated"):
        from flask import redirect
        return redirect("/", code=302)

    market_filter = (request.args.get("market") or "all").upper()
    show = request.args.get("show") or "all"

    try:
        games = generic_odds.build_sport_games(sport)
    except Exception as e:
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
        sport=sport,
        rows=rows,
        game_count=len(games),
        positive_count=positive_count,
        strong_count=strong_count,
        odds_api_available=generic_odds.odds_api_available(),
        market_filter=market_filter,
        available_markets=markets_meta,
        show=show,
        now=datetime.now(EASTERN).strftime("%I:%M %p ET").lstrip("0"),
    )


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        lan_ip = socket.gethostbyname(hostname)
    except OSError:
        lan_ip = None
    print("First Pitch - daily MLB scoreboard + prediction model")
    print("  On this PC : http://127.0.0.1:5000")
    if lan_ip and not lan_ip.startswith("127."):
        print(f"  On phone   : http://{lan_ip}:5000   (same Wi-Fi network)")
    print("  Backtest   : /backtest")
    app.run(debug=False, host="0.0.0.0", port=5000)
