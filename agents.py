"""AI agents — scheduled Claude calls that produce prose outputs.

Each agent is a plain function that:
  1. Pulls the data it needs from plays_log / board state
  2. Compacts into a Sonnet or Haiku prompt
  3. Returns a dict with the parsed response
  4. Records API usage via mlb_ui._record_ai_usage() so the MTD
     spend pill on TODAY stays accurate

Agents are called from /cron/<agent-name> routes (hit by Render
cron or UptimeRobot) rather than running on a background thread,
so each one is observable, idempotent per hour-of-day, and the
operator can trigger manually when debugging.

Shipped agents (2026-10-24):
  model_tune_weekly()  — reads last 7d of logged picks, asks Sonnet
                          for per-sport/market under-performance +
                          suggested consensus weight nudges
  compose_eod_recap()  — reads today's bet-able picks + results,
                          asks Haiku for a 2-3 paragraph human-
                          readable day summary suitable for Discord
"""
import json
import os
import time
from datetime import datetime, timedelta


# Lazy accessor so cross-module imports don't break if agents.py
# gets imported before mlb_ui's _record_ai_usage is wired up.
def _record_usage(model, in_toks, out_toks):
    try:
        import mlb_ui
        mlb_ui._record_ai_usage(model, in_toks, out_toks)
    except Exception:
        pass


def _resolve_discord_webhook(settings):
    """Resolve the Discord webhook the moderator / recap agents post
    through. Preference order:
      1. daily_picks_webhook_url (dedicated picks channel, legacy name)
      2. discord_webhook_url     (general picks channel, now the single
                                   webhook most users set)
    Returns an empty string if neither is configured. Lets the user
    get away with configuring just one webhook URL in /settings instead
    of two identical ones."""
    url = (settings.get("daily_picks_webhook_url") or "").strip()
    if url:
        return url
    return (settings.get("discord_webhook_url") or "").strip()


def _call_claude(prompt, model="claude-sonnet-5-5", max_tokens=2000, system=None):
    """Thin wrapper around Anthropic's SDK that returns parsed text +
    records usage. Falls back to None if the SDK isn't configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
    except Exception:
        return None
    try:
        client = Anthropic(api_key=api_key)
        kwargs = {"model": model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        # Record token usage for the MTD spend pill
        try:
            u = getattr(resp, "usage", None)
            _record_usage(model,
                           int(getattr(u, "input_tokens", 0) or 0),
                           int(getattr(u, "output_tokens", 0) or 0))
        except Exception:
            pass
        # Claude 5 returns ThinkingBlock + TextBlock; take the first
        # TextBlock with non-empty text (same pattern as analyst.py).
        txt = ""
        for block in (resp.content or []):
            if getattr(block, "type", None) == "text":
                txt = (getattr(block, "text", "") or "").strip()
                if txt:
                    break
        if not txt:
            for block in (resp.content or []):
                t = (getattr(block, "text", "") or "").strip()
                if t:
                    txt = t
                    break
        return txt
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {str(e)[:200]}"


# ============================================================
# Agent #7 — Self-critiquing model tuner (weekly)
# ============================================================

MODEL_TUNE_PROMPT = """You are the auto-analyst for a sports-betting
model. Below is the last 7 days of our Signal 3+ bet record, bucketed
by sport and market. For each bucket, compare:
  - Hit rate vs expectation (50% ML, 50% spread, 50% total baseline)
  - Units P&L vs expectation (0u baseline on 1u flat stakes)
  - CLV (closing line value — model's prob vs Pinnacle closing devig)

Return a tight, structured analysis:

OVERALL: <1 sentence read of the week>

OUTPERFORMERS:
- <sport+market>: W-L, units, 1-line why
- <optional second bucket>

UNDERPERFORMERS:
- <sport+market>: W-L, units, 1-line hypothesis for the miss
- <optional second bucket>

SUGGESTED NUDGES:
- <sport>: <action — "shrink model weight 5pp", "raise prob floor to 0.60", "pause market X">
- <optional second nudge>

CONFIDENCE: <low/medium/high — how comfortable are you with the suggestions given only 7d of data>

Only suggest nudges where the sample size is at least 10 bets AND
the signal exceeds expected noise. For buckets under 10 bets, say
"insufficient sample" and skip. Do NOT suggest paper-trading changes
that would require model code rewrites — stick to tunable weights,
thresholds, and bucket pauses.

======================
7-DAY RECORD DATA:
{record_data}
======================
"""


def _collect_tune_data():
    """Pull last 7 days of logged picks + compute per-sport/market
    buckets. Returns a compact string suitable for the LLM."""
    try:
        import plays_log
        graded = plays_log.grade_all(limit_dates=10) or {}
    except Exception:
        graded = {}
    cutoff = (datetime.now().date() - timedelta(days=7)).isoformat()
    picks = []
    for day, day_picks in graded.items():
        if day < cutoff:
            continue
        for p in day_picks or []:
            if p.get("do_not_bet") or p.get("model_only"):
                continue
            if (p.get("signal_score") or 0) < 3:
                continue
            picks.append(p)
    # Bucket by (sport, market_category)
    from collections import defaultdict
    buckets = defaultdict(lambda: {"W": 0, "L": 0, "P": 0, "u": 0.0, "n": 0,
                                     "clv_sum": 0.0, "clv_n": 0})
    for p in picks:
        key = (p.get("sport") or "?", (p.get("market") or "?").lower())
        b = buckets[key]
        b["n"] += 1
        r = p.get("result")
        if r in ("W", "L", "P"):
            b[r] += 1
            try:
                b["u"] += float(p.get("profit_u") or 0.0)
            except Exception:
                pass
        # CLV: pinnacle_prob at log time vs close_prob (if captured)
        try:
            clv = p.get("clv_pp")
            if clv is not None:
                b["clv_sum"] += float(clv)
                b["clv_n"]   += 1
        except Exception:
            pass
    # Format as text for the prompt
    lines = []
    for (sport, market), b in sorted(buckets.items()):
        if b["n"] < 3:
            continue  # skip trivially small buckets
        wlp = f"{b['W']}-{b['L']}" + (f"-{b['P']}" if b["P"] else "")
        hit = b["W"] / max(1, b["W"] + b["L"]) * 100 if (b["W"] + b["L"]) else 0
        clv_str = ""
        if b["clv_n"] > 0:
            clv_str = f" | CLV {b['clv_sum']/b['clv_n']:+.1f}pp"
        lines.append(f"  {sport:14s} {market[:20]:20s}  {wlp:9s}  "
                     f"{b['n']:3d} bets  {hit:4.0f}% hit  "
                     f"{b['u']:+6.2f}u{clv_str}")
    return "\n".join(lines) or "  (no Signal 3+ bets in last 7 days)"


def model_tune_weekly():
    """Weekly agent. Reads last 7d record, asks Sonnet for analysis.
    Writes result to logs/analyzer_suggestions.json. Returns the
    analysis + the prompt data for operator inspection."""
    data = _collect_tune_data()
    prompt = MODEL_TUNE_PROMPT.format(record_data=data)
    system = ("You are an experienced sports-model analyst. Be "
              "concise, data-driven, and honest about small-sample "
              "limitations.")
    reply = _call_claude(prompt, model="claude-sonnet-5-5",
                           max_tokens=1500, system=system)
    out = {
        "ran_at":      datetime.utcnow().isoformat() + "Z",
        "record_data": data,
        "analysis":    reply or "(no response)",
    }
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "logs", "analyzer_suggestions.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Append to a running list of weekly runs (last 12 kept)
        try:
            with open(path) as f:
                history = json.load(f) or []
        except Exception:
            history = []
        if not isinstance(history, list):
            history = []
        history.append(out)
        history = history[-12:]
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        # Mirror via log_persist so deploys don't wipe
        try:
            import log_persist
            log_persist.write_file("logs/analyzer_suggestions.json",
                                     json.dumps(history, indent=2))
        except Exception:
            pass
    except Exception:
        pass
    return out


# ============================================================
# Agent #4 — Daily EOD recap composer
# ============================================================

EOD_RECAP_PROMPT = """You are a sharp sports-model analyst writing a
day-in-review for the Discord subscribers. Below is today's record.
Write 2-3 short paragraphs:

- Lead with the overall record + units for the day. Call out the
  biggest win (specific pick, sport, result) and the biggest bleed.
- Note which filter/tier performed best (Strongest, Filter 2.0,
  Whale, AI).
- Close with one honest line about what the day says about the
  model (hot streak, variance, specific sport trending). No fluff.

Tone: dry, confident, no exclamation points, no emoji, no "thanks
for being part of the community" filler. Treat the reader as a
peer, not a customer.

======================
TODAY: {date_label}
Record: {wlp}  Units: {units}  ROI: {roi}

Per-tier breakdown:
{tier_table}

Picks (sorted by profit):
{pick_list}
======================
"""


def _collect_eod_data(date_str=None):
    """Returns dict with per-tier record + top gains/losses for Haiku."""
    try:
        import plays_log
        import mlb_ui
    except Exception:
        return None
    try:
        day = date_str or datetime.now(mlb_ui.CENTRAL).date().isoformat()
    except Exception:
        day = date_str or datetime.now().date().isoformat()
    try:
        graded = plays_log.grade_all(limit_dates=3) or {}
    except Exception:
        graded = {}
    picks = [p for p in (graded.get(day) or [])
             if not p.get("do_not_bet") and not p.get("model_only")]
    if not picks:
        return None

    # Overall
    w = sum(1 for p in picks if p.get("result") == "W")
    l = sum(1 for p in picks if p.get("result") == "L")
    push = sum(1 for p in picks if p.get("result") == "P")
    units = sum(float(p.get("profit_u") or 0.0) for p in picks)
    settled = w + l + push
    hit = (w / max(1, w + l) * 100) if (w + l) else 0
    roi_pct = (units / settled * 100) if settled else 0

    # Per-tier
    def _tier_of(p):
        sig = int(p.get("signal_score") or 0)
        if p.get("whale"): return "Whale"
        if sig >= 5: return "5/5"
        if sig >= 4: return "4/5"
        if sig >= 3: return "3/5"
        return "under"
    tiers = {}
    for p in picks:
        t = _tier_of(p)
        b = tiers.setdefault(t, {"n": 0, "w": 0, "l": 0, "u": 0.0})
        b["n"] += 1
        if p.get("result") == "W": b["w"] += 1
        elif p.get("result") == "L": b["l"] += 1
        try: b["u"] += float(p.get("profit_u") or 0.0)
        except Exception: pass

    tier_lines = []
    for t in ("Whale", "5/5", "4/5", "3/5"):
        if t not in tiers:
            continue
        b = tiers[t]
        tier_lines.append(f"  {t:6s} {b['w']}-{b['l']}  n={b['n']}  {b['u']:+.2f}u")

    # Top gain + biggest bleed
    settled_picks = [p for p in picks if p.get("result") in ("W", "L")]
    settled_picks.sort(key=lambda p: float(p.get("profit_u") or 0.0), reverse=True)
    pick_lines = []
    for p in settled_picks[:3]:
        pr = p.get("profit_u") or 0.0
        pick_lines.append(f"  {p.get('result','?')} {p.get('sport',''):8s} "
                           f"{(p.get('pick') or '')[:40]:40s} {float(pr):+.2f}u")
    for p in settled_picks[-2:]:
        if p in settled_picks[:3]: continue
        pr = p.get("profit_u") or 0.0
        pick_lines.append(f"  {p.get('result','?')} {p.get('sport',''):8s} "
                           f"{(p.get('pick') or '')[:40]:40s} {float(pr):+.2f}u")

    return {
        "date_label": datetime.strptime(day, "%Y-%m-%d").strftime("%A, %B %d"),
        "wlp":  f"{w}-{l}" + (f"-{push}" if push else ""),
        "units": f"{units:+.2f}u",
        "roi":   f"{roi_pct:+.1f}%",
        "tier_table": "\n".join(tier_lines) or "  (none)",
        "pick_list":  "\n".join(pick_lines) or "  (no settled picks)",
    }


def compose_eod_recap(date_str=None, post_to_discord=False):
    """Build the recap prose. If post_to_discord=True, also sends it
    to the daily-picks webhook. Returns {ok, prose, data}."""
    data = _collect_eod_data(date_str)
    if not data:
        return {"ok": False, "error": "no bet-able picks on this date"}
    prompt = EOD_RECAP_PROMPT.format(**data)
    system = ("You write short day-in-review recaps for betting-model "
              "subscribers. Dry, honest, zero fluff.")
    reply = _call_claude(prompt, model="claude-haiku-5-5",
                           max_tokens=800, system=system)
    if not reply or reply.startswith("__ERROR__"):
        return {"ok": False, "error": reply or "no response", "data": data}
    if post_to_discord:
        try:
            import mlb_ui
            settings = mlb_ui._load_settings()
            webhook = _resolve_discord_webhook(settings)
            if webhook:
                import discord_bot
                discord_bot.post_text(webhook,
                    f":bar_chart: **EOD Recap — {data['date_label']}**\n\n{reply}")
        except Exception:
            pass
    return {"ok": True, "prose": reply, "data": data}


# ============================================================
# Agent #1 — News/injury watcher (cost-optimized cadence)
# ============================================================
# Design: the user flagged that polling every 10 min was expensive.
# Solution: separate the DETECTION step (free HTTP) from the
# ANALYSIS step (LLM).
#
#   Detection (every tick): poll ESPN injury API per active sport.
#   Diff against local status cache to find NEW items (first-time
#   seen) and CHANGED items (status flipped e.g. Questionable → Out).
#   This is pure HTTP + dict comparison. $0.
#
#   Analysis (on-demand): only when a NEW/CHANGED item is both
#   (a) involving a player on a team playing today AND
#   (b) status just worsened,
#   fire ONE Haiku call to summarize impact on affected picks.
#   Hard daily cap: 15 LLM calls/day to protect the bill.

_NEWS_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "logs", "news_watcher_state.json")
_NEWS_LLM_DAILY_CAP = 15

ESPN_INJURY_URLS = {
    "nba":   "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "nfl":   "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
    "mlb":   "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries",
    "nhl":   "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries",
}

# Status order from least-to-most-severe. Any flip to a higher rank
# counts as "worsened" and may trigger an LLM analysis call.
_STATUS_RANK = {
    "healthy": 0,
    "day-to-day": 1,
    "questionable": 2,
    "doubtful": 3,
    "out": 4,
    "injured reserve": 4,
    "il": 4,
    "suspended": 4,
}


def _load_news_state():
    try:
        with open(_NEWS_STATE_PATH) as f:
            return json.load(f) or {}
    except Exception:
        try:
            import log_persist
            text, _sha = log_persist.read_file("logs/news_watcher_state.json")
            if text:
                return json.loads(text) or {}
        except Exception:
            pass
        return {}


def _save_news_state(state):
    try:
        os.makedirs(os.path.dirname(_NEWS_STATE_PATH), exist_ok=True)
        with open(_NEWS_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass
    try:
        import log_persist
        log_persist.write_file("logs/news_watcher_state.json",
                                 json.dumps(state))
    except Exception:
        pass


def _fetch_espn_injuries(sport_slug):
    """Return list of {team, player, status, date, description} from
    ESPN injury API. Pure HTTP, no LLM."""
    url = ESPN_INJURY_URLS.get(sport_slug)
    if not url:
        return []
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "cheeks-model/1.0 (news-watcher)"})
        with urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    out = []
    try:
        for team_block in (data.get("injuries") or []):
            team = (team_block.get("displayName") or team_block.get("abbreviation")
                    or team_block.get("team", {}).get("displayName") or "?")
            for inj in (team_block.get("injuries") or []):
                player = (inj.get("athlete") or {}).get("displayName", "?")
                status = (inj.get("status") or inj.get("type", {}).get("description")
                          or "unknown")
                desc = (inj.get("details") or {}).get("detail") or inj.get("longComment") or ""
                out.append({
                    "team":   team,
                    "player": player,
                    "status": status,
                    "desc":   desc[:300],
                    "date":   inj.get("date") or "",
                })
    except Exception:
        pass
    return out


def _todays_sports():
    """Return sport slugs that have games today (from board_rows)."""
    try:
        import mlb_ui
        today = datetime.now(mlb_ui.CENTRAL).date()
        board, _ = mlb_ui._build_board_rows(today.isoformat(), today, "all",
                                              strong_only=False)
        return sorted({r.get("sport_slug") for r in board if r.get("sport_slug")})
    except Exception:
        return list(ESPN_INJURY_URLS.keys())


# Pre-game window (shipped 2026-10-24): LLM analysis fires only when
# the player's team has a game within this many minutes. Late news is
# what actually moves lines — anything 2+ hours out is already priced
# into Pinnacle. The user specifically asked for this cut because
# lineup/scratch news often drops close to tip.
_PREGAME_WINDOW_MIN = 120     # 2 hours — safe default across sports
_PREGAME_URGENT_MIN = 30      # inside 30 min, treat as very high priority


def _upcoming_games_by_team(sport_slugs=None):
    """Return {sport|team-slug: minutes_to_tipoff} for games on today's
    board that are UPCOMING (not live/final). Used by the news watcher
    to decide whether a given injury item is worth an LLM call."""
    try:
        import mlb_ui
        today = datetime.now(mlb_ui.CENTRAL).date()
        now = datetime.now(mlb_ui.CENTRAL)
        board, _ = mlb_ui._build_board_rows(today.isoformat(), today, "all",
                                              strong_only=False)
    except Exception:
        return {}
    out = {}
    for r in board:
        if sport_slugs and r.get("sport_slug") not in sport_slugs:
            continue
        if (r.get("status") or "") != "upcoming":
            continue
        try:
            from datetime import datetime as _dt
            start = _dt.fromisoformat(
                (r.get("start_time") or "").replace("Z", "+00:00")
            ).astimezone(mlb_ui.CENTRAL)
        except Exception:
            continue
        mins = max(0, int((start - now).total_seconds() / 60))
        for team_name in (r.get("home") or "", r.get("away") or ""):
            if not team_name:
                continue
            key = f"{r.get('sport_slug')}|{team_name.lower()}"
            prev = out.get(key)
            if prev is None or mins < prev:
                out[key] = mins
    return out


def _minutes_to_team_tipoff(sport, team_name, upcoming_games):
    """Match an ESPN injury's team name against our board's team
    names. ESPN uses full display names ('New York Knicks'); our
    board uses variations from Pinnacle/ESPN. Match by substring
    overlap on normalized tokens — if any 4+char token from the
    injury's team matches a token from a board team, consider it
    the same team. Returns minutes to tipoff, or None if no game
    today or already live/final."""
    if not team_name:
        return None
    inj_tokens = set(t for t in team_name.lower().split() if len(t) >= 4)
    best = None
    for key, mins in upcoming_games.items():
        key_sport, _, board_team = key.partition("|")
        if key_sport != sport:
            continue
        board_tokens = set(t for t in board_team.split() if len(t) >= 4)
        if inj_tokens & board_tokens:
            if best is None or mins < best:
                best = mins
    return best


def run_news_watcher():
    """Agent #1 main loop. Poll ESPN injuries per active sport,
    diff, flag NEW + WORSENED items affecting today's teams, and
    (optionally, under daily cap) summarize via Haiku."""
    state = _load_news_state()
    today = datetime.utcnow().date().isoformat()
    daily = state.get("daily") or {}
    if daily.get("date") != today:
        daily = {"date": today, "llm_calls": 0}
    seen = state.get("seen") or {}  # key = sport|team|player → {status, ts}

    changes = []
    for sport in _todays_sports():
        if sport not in ESPN_INJURY_URLS:
            continue
        for inj in _fetch_espn_injuries(sport):
            key = f"{sport}|{inj['team']}|{inj['player']}"
            prev = seen.get(key)
            prev_status = (prev or {}).get("status", "healthy")
            curr_status = inj["status"]
            prev_rank = _STATUS_RANK.get((prev_status or "").lower(), 1)
            curr_rank = _STATUS_RANK.get((curr_status or "").lower(), 1)
            if prev is None:
                change_type = "new"
            elif curr_rank > prev_rank:
                change_type = "worsened"
            elif curr_rank < prev_rank:
                change_type = "improved"
            else:
                continue  # unchanged
            changes.append({
                "sport":        sport,
                "team":         inj["team"],
                "player":       inj["player"],
                "status":       curr_status,
                "prev_status":  prev_status,
                "change":       change_type,
                "desc":         inj["desc"],
            })
            seen[key] = {"status": curr_status, "ts": time.time()}

    # Pre-game window gate (shipped 2026-10-24). LLM analysis only
    # fires for items whose team has a game starting in the next
    # _PREGAME_WINDOW_MIN minutes. Earlier news is already priced
    # into Pinnacle lines; later news (post-tip) can't drive a
    # pre-game bet decision. Also sort within-window items so the
    # most-urgent (closest to tipoff) burn the daily cap first.
    upcoming = _upcoming_games_by_team()
    worth_llm = []
    for c in changes:
        if c["change"] not in ("new", "worsened"):
            continue
        if _STATUS_RANK.get((c["status"] or "").lower(), 0) < 2:
            continue  # less-than-Questionable not worth an LLM call
        mins = _minutes_to_team_tipoff(c["sport"], c["team"], upcoming)
        c["mins_to_tip"] = mins
        if mins is None or mins > _PREGAME_WINDOW_MIN:
            continue  # team not playing today, or game is 2+ hours out
        worth_llm.append(c)

    # Sort: closest-to-tipoff first, then by severity. Urgent news
    # inside _PREGAME_URGENT_MIN gets guaranteed cap room.
    worth_llm.sort(key=lambda c: (
        c["mins_to_tip"],
        -_STATUS_RANK.get((c["status"] or "").lower(), 1),
    ))

    analyzed = []
    remaining = _NEWS_LLM_DAILY_CAP - int(daily.get("llm_calls") or 0)
    for item in worth_llm[:max(0, remaining)]:
        prompt = (
            f"One-sentence (<=25 words) impact assessment for THIS game.\n"
            f"Sport: {item['sport'].upper()}\n"
            f"Team: {item['team']} (plays in {item['mins_to_tip']} min)\n"
            f"Player: {item['player']}\n"
            f"Status change: {item['prev_status']} → {item['status']}\n"
            f"Details: {item['desc'] or '(none)'}\n\n"
            f"Reply with ONE sentence focused on how this affects "
            f"the upcoming game's betting lines. Name the market "
            f"most moved (ML / spread / total / team total / props). "
            f"If this is a role player with low impact, say so plainly."
        )
        reply = _call_claude(prompt, model="claude-haiku-5-5",
                               max_tokens=120)
        if not reply or reply.startswith("__ERROR__"):
            continue
        item["summary"] = reply
        analyzed.append(item)
        daily["llm_calls"] = int(daily.get("llm_calls") or 0) + 1

    state["seen"] = seen
    state["daily"] = daily
    state["last_run"] = datetime.utcnow().isoformat() + "Z"
    state["last_changes"] = changes[:30]  # keep most recent for /debug
    _save_news_state(state)

    return {
        "ok":              True,
        "ran_at":          state["last_run"],
        "sports_checked":  _todays_sports(),
        "changes_total":   len(changes),
        "worsened":        len([c for c in changes if c["change"] == "worsened"]),
        "new":             len([c for c in changes if c["change"] == "new"]),
        "worth_llm_eligible": len(worth_llm),  # passed severity + pregame gates
        "llm_calls_today": daily["llm_calls"],
        "llm_cap":         _NEWS_LLM_DAILY_CAP,
        "pregame_window_min": _PREGAME_WINDOW_MIN,
        "analyzed":        analyzed,
    }


# ============================================================
# Agent #6 — Historical analog finder
# ============================================================
# For a given pick id, finds up to N comparable past picks (same
# sport + market + similar signal + similar line) from plays_log,
# then asks Haiku for a one-line pattern read.

def find_pick_analogs(pick_id, n=3):
    """Return {ok, pick, analogs, pattern_read}. pick is the current
    pick metadata; analogs are up to N comparable past settled picks;
    pattern_read is Haiku's 1-sentence synthesis."""
    try:
        import plays_log
        graded = plays_log.grade_all(limit_dates=180) or {}
    except Exception:
        return {"ok": False, "error": "plays_log import failed"}

    # Find the target pick first (walk recent dates)
    target = None
    for day in sorted(graded.keys(), reverse=True):
        for p in graded[day] or []:
            if p.get("id") == pick_id:
                target = p
                break
        if target:
            break
    if not target:
        return {"ok": False, "error": f"pick {pick_id} not found in logged history"}

    # Collect candidates: same sport, same market category (ML/Spread/Total)
    sport = target.get("sport") or ""
    market = (target.get("market") or "").lower()
    target_sig = int(target.get("signal_score") or 0)
    target_ev  = float(target.get("ev_pct") or 0.0)
    candidates = []
    for day, day_picks in graded.items():
        for p in day_picks or []:
            if p.get("id") == pick_id:
                continue  # don't match itself
            if p.get("sport") != sport:
                continue
            p_market = (p.get("market") or "").lower()
            if market not in p_market and p_market not in market:
                continue
            if p.get("result") not in ("W", "L", "P"):
                continue  # need graded for the analog to be useful
            if p.get("do_not_bet") or p.get("model_only"):
                continue
            try:
                sig = int(p.get("signal_score") or 0)
                ev  = float(p.get("ev_pct") or 0.0)
            except Exception:
                continue
            dist = abs(sig - target_sig) * 10 + abs(ev - target_ev)
            candidates.append((dist, day, p))

    candidates.sort(key=lambda t: t[0])
    analogs = []
    for dist, day, p in candidates[:n]:
        analogs.append({
            "date":   day,
            "pick":   p.get("pick"),
            "market": p.get("market"),
            "signal": p.get("signal_score"),
            "ev":     p.get("ev_pct"),
            "result": p.get("result"),
            "profit": p.get("profit_u"),
        })
    if not analogs:
        return {"ok": True, "pick": {"sport": sport, "market": market,
                                       "pick": target.get("pick")},
                "analogs": [], "pattern_read": "no comparable past picks found"}

    # Haiku composes a one-line pattern read.
    lines = "\n".join(
        f"  {a['date']} | sig {a['signal']} | ev {a['ev']}% | {a['pick'][:40]} | {a['result']} ({a['profit']:+.2f}u)"
        for a in analogs
    )
    prompt = (
        f"Pattern read for this bet based on {len(analogs)} comparable past picks.\n\n"
        f"Current pick: {sport} / {market} / sig {target_sig} / ev {target_ev:+.1f}%\n"
        f"  '{target.get('pick', '')}'\n\n"
        f"Past comparables (same sport, same market type, closest signal + EV):\n{lines}\n\n"
        f"Reply with ONE sentence (<=22 words) summarizing what the "
        f"historical pattern suggests about THIS pick. Be specific "
        f"about the record (e.g. '2-1 at this profile' not 'mostly wins')."
    )
    reply = _call_claude(prompt, model="claude-haiku-5-5", max_tokens=100)
    return {
        "ok":           True,
        "pick":         {"sport": sport, "market": target.get("market"),
                          "pick": target.get("pick"), "signal": target_sig,
                          "ev": target_ev},
        "analogs":      analogs,
        "pattern_read": reply or "",
    }


# ============================================================
# Agent #8 — Smart Discord moderator (per-pick individual posts)
# ============================================================
# Shift from slate-based blast (top 5 per day) to individual-pick
# posts timed for maximum evidence. A pick qualifies for posting
# only when its evidence score crosses a threshold AND the game is
# inside the sport-specific pre-game window where news/lineups
# have surfaced. Expected volume: 1-6 posts/day instead of 15/day.

# Per-sport pre-game window where posting makes sense. Chosen based
# on when lineups + inactives + weather clear enough that evidence
# is stable. Earlier = picks still subject to lineup risk; later =
# too close to tip for subscribers to place the bet.
_SPORT_PREGAME_WINDOW_MIN = {
    "nba":    (30, 90),     # 30-90 min pre-tip (post inactives list)
    "nfl":    (60, 120),    # 1-2 hr pre-kick (inactives report)
    "nhl":    (90, 180),    # 1.5-3 hr pre (post morning skate)
    "mlb":    (90, 180),    # 1.5-3 hr pre (confirmed starter)
    "ncaaf":  (90, 180),
    "epl":    (30, 90),     # 30-90 min pre (confirmed XI)
    "laliga": (30, 90),
    "ligamx": (30, 90),
    "ucl":    (30, 90),
    "europa": (30, 90),
    "international": (30, 90),
    "ufc":   (240, 1440),   # 4-24 hr pre-fight (post weigh-in)
}
_DEFAULT_PREGAME_WINDOW = (45, 150)

# Evidence threshold — picks score 0-10. The threshold is DYNAMIC
# per the pick's Signal composite (0-5): stronger Signal gets a lower
# evidence bar, since the composite already encodes most of the
# convergence we'd otherwise want from AI+/Line+/Kalshi+.
#
#   Signal 5 ⇒ evidence >= 3
#   Signal 4 ⇒ evidence >= 3
#   Signal 3 ⇒ evidence >= 4   (tuned down from 5 on 2026-10-24 so
#                                 Signal 3 + AI + News-clean + any
#                                 one corroborating signal posts —
#                                 quiet-slate days were posting zero)
#   Signal 0-2 ⇒ evidence >= 5
#
# Keeps the quality bar high while letting a 5/5 pick with just one
# supporting signal (e.g. AI+ or Line+) through, and a 3/5 pick with
# three convergent signals (AI + Whale + News-clean = 4) through too.
_EVIDENCE_POST_THRESHOLD = 5  # back-compat; still used for stats labels


def _evidence_bar_for(pick):
    """Return the evidence-score threshold the pick must clear."""
    try:
        sig = int(pick.get("signal_score") or 0)
    except Exception:
        sig = 0
    if sig >= 5:
        return 3
    if sig >= 4:
        return 3
    if sig >= 3:
        return 4
    return 5

# Where we track which pick IDs have already been posted to Discord
# so we don't double-post. Keyed by pick id → {posted_at, score}.
_POSTED_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "logs", "discord_posted_individual.json")


def _load_posted_state():
    try:
        with open(_POSTED_STATE_PATH) as f:
            return json.load(f) or {}
    except Exception:
        try:
            import log_persist
            text, _sha = log_persist.read_file("logs/discord_posted_individual.json")
            if text:
                return json.loads(text) or {}
        except Exception:
            pass
        return {}


def _save_posted_state(state):
    try:
        os.makedirs(os.path.dirname(_POSTED_STATE_PATH), exist_ok=True)
        with open(_POSTED_STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        pass
    try:
        import log_persist
        log_persist.write_file("logs/discord_posted_individual.json",
                                 json.dumps(state))
    except Exception:
        pass


def pick_signal_for_debug(pick):
    """Return the pick's Signal composite score (0-5) as an int.
    Used only by the moderator diagnostic payload so the user can
    tell Signal composite apart from Evidence Score."""
    try:
        return int(pick.get("signal_score") or 0)
    except Exception:
        return 0


def compute_evidence_score(pick, news_state=None):
    """Return (score, reasons_list) for a pick. Score is 0-10.
    reasons_list is the compact labels used in the Discord post
    (e.g. ['AI+', 'Line+', 'Kalshi+', 'Whale']).

    All signals come from fields the UI already computes — this
    is a pure data aggregator, no LLM involved."""
    score = 0
    reasons = []
    # AI endorsement with full bullets (vs legacy one-liner)
    if (pick.get("ai_verdict") in ("Strong Take", "Lean")
            and not pick.get("do_not_bet")
            and (pick.get("ai_matchup") or pick.get("ai_factors"))):
        score += 2
        reasons.append("AI+")
    # Favorable line movement toward us
    try:
        lm = float(pick.get("line_move_pp") or 0.0)
        if lm >= 0.005:
            score += 2
            reasons.append("Line+")
    except (TypeError, ValueError):
        pass
    # Kalshi cross-check: our side agrees with Kalshi (NOT contra)
    if pick.get("kalshi_live_prob") is not None and not pick.get("kalshi_contra"):
        try:
            fp = float(pick.get("fair_prob") or 0.0)
            kp = float(pick.get("kalshi_live_prob") or 0.0)
            if fp > 0 and kp > 0 and abs(fp - kp) * 100 < 5:
                score += 2
                reasons.append("Kalshi+")
        except Exception:
            pass
    # Whale tier
    if pick.get("whale"):
        score += 1
        reasons.append("Whale")
    # Signal >= 4 (upgrades beyond baseline)
    try:
        if int(pick.get("signal_score") or 0) >= 4:
            score += 1
            reasons.append("Sig4+")
    except Exception:
        pass
    # News-watcher: no negative items for either team in pre-game window
    if news_state is None:
        news_state = _load_news_state()
    try:
        recent_changes = news_state.get("last_changes") or []
        teams = {
            (pick.get("home_team") or "").lower(),
            (pick.get("away_team") or "").lower(),
        }
        blocking = any(
            c for c in recent_changes
            if (c.get("team") or "").lower() in teams
            and c.get("change") == "worsened"
        )
        if not blocking:
            score += 1
            reasons.append("News-clean")
    except Exception:
        pass
    return (min(10, score), reasons)


def _pick_sport_pregame_window(sport_slug):
    return _SPORT_PREGAME_WINDOW_MIN.get(sport_slug, _DEFAULT_PREGAME_WINDOW)


def _minutes_to_game_start(start_time_iso):
    try:
        import mlb_ui
        from datetime import datetime as _dt
        start = _dt.fromisoformat(start_time_iso.replace("Z", "+00:00")).astimezone(mlb_ui.CENTRAL)
        now = _dt.now(mlb_ui.CENTRAL)
        return int((start - now).total_seconds() / 60)
    except Exception:
        return None


MODERATOR_POST_PROMPT = """You are writing a SHORT Discord post for one betting pick that
cleared our highest-evidence filter. Lead with the pick + price. Then
3 concrete bullets:

  🏉 Matchup
  📋 Key factor
  ⚠ Risk

Each bullet one sentence, <=18 words, naming a specific player/stat/trend.
No preamble, no sign-off, no emoji in the body text beyond the three
leading icons. Match this exact format:

**{sport} · {pick}** @ {decimal}
Evidence {score}/10 — {reason_tags}  |  Tipoff in {mins_to_tip} min

🏉 {matchup_bullet}
📋 {factor_bullet}
⚠ {risk_bullet}
"""


def resend_picks_posted_after(hour_ct=15, webhook_url=None, dry=False):
    """Force-resend every pick that was posted to Discord today after
    the given CT hour (default 3pm CT), in the current embed format.
    Pulls the pick list from today's logged plays_log file and the
    moderator's posted_state, filters by posted_at >= hour_ct CT,
    then calls post_individual_pick(allow_replay=True) on each.

    Useful after a format change (e.g. switching from text to
    embeds) to re-send today's picks in the new look. Does NOT
    delete the originals — webhook self-delete only works for
    message IDs we captured, and older posts predate that.
    Operator can manually delete the old text posts OR use
    /discord/dedup-sweep if a bot token is configured."""
    try:
        import mlb_ui
        import plays_log
    except Exception:
        return {"ok": False, "error": "mlb_ui / plays_log import failed"}
    today = datetime.now(mlb_ui.CENTRAL).date()
    today_iso = today.isoformat()
    # Pull from the LIVE board — picks there carry the AI bullets
    # (served from the 24h verdict cache). The saved plays_log file
    # historically dropped those fields, so pulling from there gave
    # us empty body_text. Fall back to graded if a pick isn't on
    # today's live board (game already ended + Pinnacle dropped it).
    live_by_id = {}
    try:
        board_rows, _ = mlb_ui._build_board_rows(today_iso, today, "all",
                                                   strong_only=False)
        for row in board_rows:
            for p in (row.get("picks") or []):
                _pid = p.get("id")
                if _pid:
                    live_by_id[_pid] = p
    except Exception:
        pass
    graded_file = plays_log.read_graded(today_iso) or {}
    graded_picks = graded_file.get("picks") or []
    posted_state = _load_posted_state()
    cutoff_ct = datetime.combine(today,
                                   datetime.min.time().replace(hour=hour_ct),
                                   tzinfo=mlb_ui.CENTRAL)
    cutoff_utc = cutoff_ct.astimezone().replace(tzinfo=None)
    # datetime.utcnow() is naive UTC — compare to cutoff_utc (naive)
    posted_pids = []
    for pid, entry in posted_state.items():
        if entry.get("day") != today_iso:
            continue
        posted_at_str = entry.get("posted_at") or ""
        try:
            posted_at = datetime.fromisoformat(
                posted_at_str.replace("Z", "")).replace(tzinfo=None)
        except Exception:
            continue
        if posted_at >= cutoff_utc:
            posted_pids.append(pid)
    # Build pick dict map for lookup — prefer live board (has AI
    # bullets); fall back to the saved graded file for anything the
    # live board no longer carries.
    picks_by_id = {p.get("id"): p for p in graded_picks if p.get("id")}
    for _pid, _lp in live_by_id.items():
        # Merge: start from the saved pick, overlay the live fields
        # on top so AI bullets + current pricing come through.
        base = picks_by_id.get(_pid) or {}
        picks_by_id[_pid] = {**base, **_lp}
    resent = []
    missing = []
    for pid in posted_pids:
        pick = picks_by_id.get(pid)
        if not pick:
            missing.append(pid)
            continue
        if dry:
            # Just report what body_text + available fields WOULD be sent.
            resent.append({
                "pid":           pid,
                "has_ai_matchup": bool(pick.get("ai_matchup")),
                "has_ai_factors": bool(pick.get("ai_factors")),
                "has_ai_risks":   bool(pick.get("ai_risks")),
                "ai_matchup_len": len(pick.get("ai_matchup") or []),
                "ai_factors_len": len(pick.get("ai_factors") or []),
                "ai_risks_len":   len(pick.get("ai_risks") or []),
                "ai_verdict":     pick.get("ai_verdict"),
                "keys_sample":    sorted(pick.keys())[:30],
            })
            continue
        try:
            r = post_individual_pick(pick, webhook_url=webhook_url,
                                      allow_replay=True)
            resent.append({"pid": pid, "ok": r.get("ok"),
                            "message_id": r.get("message_id")})
        except Exception as e:
            resent.append({"pid": pid, "ok": False,
                            "error": f"{type(e).__name__}: {e}"})
    return {
        "ok":           True,
        "today":        today_iso,
        "cutoff_ct":    f"{hour_ct:02d}:00 CT",
        "candidate_count": len(posted_pids),
        "resent_count": sum(1 for r in resent if r.get("ok")),
        "resent":       resent,
        "missing_from_log": missing,
    }


def delete_my_posts_today(webhook_url=None):
    """Delete every pick the moderator posted today whose message_id
    we captured. Also prunes the dedup state for those picks so they
    can re-post in the new format. Returns a result dict with the
    deleted count. Uses the webhook's own ?wait=true/DELETE flow so
    no bot token is required."""
    try:
        import mlb_ui
        import discord_bot
    except Exception:
        return {"ok": False, "error": "mlb_ui / discord_bot import failed"}
    if not webhook_url:
        try:
            settings = mlb_ui._load_settings()
            webhook_url = _resolve_discord_webhook(settings)
        except Exception:
            webhook_url = None
    if not webhook_url:
        return {"ok": False, "error": "no webhook configured"}
    today_iso = datetime.utcnow().date().isoformat()
    posted_state = _load_posted_state()
    deleted = []
    skipped = []
    for pid, entry in list(posted_state.items()):
        if entry.get("day") != today_iso:
            continue
        msg_id = entry.get("message_id")
        if not msg_id:
            skipped.append({"pid": pid, "reason": "no_message_id"})
            # Prune dedup anyway so a re-post can happen
            del posted_state[pid]
            continue
        if discord_bot.delete_webhook_message(webhook_url, msg_id):
            deleted.append({"pid": pid, "message_id": msg_id})
            del posted_state[pid]
        else:
            skipped.append({"pid": pid, "reason": "delete_failed",
                             "message_id": msg_id})
    _save_posted_state(posted_state)
    return {"ok": True, "deleted_count": len(deleted),
            "skipped_count": len(skipped),
            "deleted": deleted[:20], "skipped": skipped[:20]}


def _fmt_bullets(items, emoji):
    """Format a list of bullet strings as 'emoji text' lines.
    Returns empty string if items is empty / None so the caller
    can skip that section entirely."""
    if not items:
        return ""
    # Accept either list[str] or a single str
    if isinstance(items, str):
        items = [items]
    lines = []
    for it in items:
        it = (it or "").strip().lstrip("-•*").strip()
        if it:
            lines.append(f"{emoji} {it}")
    return "\n".join(lines)


def post_individual_pick(pick, webhook_url=None, allow_replay=False):
    """Compose + send one pick to Discord. Returns {ok, posted_text}.

    Builds a full Discord post with:
      - Sport · pick · price · evidence · tipoff
      - AI verdict label (e.g. "AI: Lean")
      - FULL matchup bullets
      - FULL key-factor bullets
      - FULL risk bullets
      - Best US-book price chip (if a book beats Pinnacle by >=1%)

    Called by the moderator when a pick clears its evidence bar,
    or on-demand via /pick/force-post?id=<pid>.

    Idempotent: if the pick id is already in today's posted state,
    returns {ok:false, dedup:true, ...} without hitting Discord
    unless allow_replay=True. This is the belt-and-suspenders guard
    against double-posts across concurrent moderator ticks and
    deploy-wipe-then-restore races."""
    try:
        import mlb_ui
        import discord_bot
    except Exception:
        return {"ok": False, "error": "mlb_ui / discord_bot import failed"}

    # Idempotence check — don't post the same pick twice in one day.
    pid = pick.get("id")
    if pid and not allow_replay:
        posted_state = _load_posted_state()
        if pid in posted_state:
            return {"ok": False, "dedup": True,
                    "error": "pick already posted today",
                    "posted_at": posted_state[pid].get("posted_at")}

    if not webhook_url:
        try:
            settings = mlb_ui._load_settings()
            webhook_url = _resolve_discord_webhook(settings)
        except Exception:
            webhook_url = None
    if not webhook_url:
        return {"ok": False, "error": "no webhook configured"}

    score, reasons = compute_evidence_score(pick)
    mins = _minutes_to_game_start(pick.get("start_time") or "")

    # Attach computed fields the PNG renderer reads.
    pick["evidence_score"] = score
    pick["mins_to_tip"]    = mins
    pick["reasons"]        = reasons

    # Compose the text body (analysis sections only — header info is
    # now in the PNG card). Bullets rendered as emoji-led lines.
    def _bullets(items, emoji, cap=600):
        if not items:
            return ""
        if isinstance(items, str):
            items = [items]
        lines = []
        for it in items:
            it = (it or "").strip().lstrip("-•*").strip()
            if it:
                lines.append(f"{emoji} {it}")
        text = "\n".join(lines).strip()
        return (text[:cap - 3] + "...") if len(text) > cap else text

    m_block = _bullets(pick.get("ai_matchup"), "🎯")
    f_block = _bullets(pick.get("ai_factors"), "📋")
    r_block = _bullets(pick.get("ai_risks"),   "⚠")
    body_parts = []
    if m_block:
        body_parts.append("**Matchup**\n" + m_block)
    if f_block:
        body_parts.append("**Key factors**\n" + f_block)
    if r_block:
        body_parts.append("**Risks**\n" + r_block)
    body_text = "\n\n".join(body_parts)

    # Send PNG card + text body as ONE Discord message
    msg_id = discord_bot.send_pick_card(webhook_url, pick, body_text,
                                          return_id=True)
    ok = bool(msg_id)
    # Record immediately so a concurrent tick (UptimeRobot + manual
    # curl + Render internal) can't double-post. The moderator loop
    # also records, but doing it here closes the race entirely.
    # Also store message_id so delete_my_posts can clean up later.
    if ok and pid:
        try:
            posted_state = _load_posted_state()
            posted_state[pid] = {
                "posted_at":   datetime.utcnow().isoformat() + "Z",
                "score":       score,
                "reasons":     reasons,
                "mins":        mins,
                "day":         datetime.utcnow().date().isoformat(),
                "message_id":  msg_id if isinstance(msg_id, str) else None,
                "webhook_tail": (webhook_url or "")[-24:],
            }
            _save_posted_state(posted_state)
        except Exception:
            pass
    return {"ok": ok, "message_id": msg_id, "score": score,
            "reasons": reasons, "mins_to_tip": mins}


def run_moderator_tick():
    """Called every ~10 min. Scans the current board, computes
    evidence per pick, posts picks that:
      - clear evidence threshold (>= 5)
      - are inside their sport-specific pre-game window
      - haven't already been posted today
    Returns summary of what was posted + what was eligible."""
    try:
        import mlb_ui
        today = datetime.now(mlb_ui.CENTRAL).date()
        board, _ = mlb_ui._build_board_rows(today.isoformat(), today, "all",
                                              strong_only=False)
    except Exception as e:
        return {"ok": False, "error": f"board build failed: {e}"}

    posted_state = _load_posted_state()
    news_state   = _load_news_state()
    today_str    = today.isoformat()
    posted_today = []
    eligible     = []
    # Diagnostic counters — surfaced in the response so the operator
    # can tell at a glance whether an empty eligible list means "no
    # picks on the board" vs "picks exist but none cleared the bar"
    # vs "picks cleared the bar but none are in-window yet".
    board_total   = 0
    evaluated     = 0
    dnb_or_model  = 0
    already_posted = 0
    score_hist    = {i: 0 for i in range(11)}        # Evidence (0-10)
    signal_hist   = {i: 0 for i in range(6)}         # Signal (0-5)
    reason_tally  = {"AI+": 0, "Line+": 0, "Kalshi+": 0,
                      "Whale": 0, "Sig4+": 0, "News-clean": 0}
    below_thresh  = []
    window_blocked = []
    live_or_done  = 0

    for row in board:
        for p in row.get("picks") or []:
            board_total += 1
            if p.get("do_not_bet") or p.get("model_only"):
                dnb_or_model += 1
                continue
            pid = p.get("id")
            if not pid or pid in posted_state:
                if pid:
                    already_posted += 1
                continue
            score, reasons = compute_evidence_score(p, news_state=news_state)
            evaluated += 1
            score_hist[max(0, min(10, int(score)))] += 1
            signal_hist[max(0, min(5, pick_signal_for_debug(p)))] += 1
            for r in reasons:
                if r in reason_tally:
                    reason_tally[r] += 1
            bar = _evidence_bar_for(p)
            if score < bar:
                # Keep the top of the "almost-cleared" list visible so
                # the operator can see what's just below the gate.
                if len(below_thresh) < 10:
                    below_thresh.append({
                        "pid":            pid,
                        "score":          score,
                        "bar":            bar,       # per-pick threshold
                        "reasons":        reasons,  # ['AI+','Line+',...]
                        "signal_composite": pick_signal_for_debug(p),
                        "has_ai_bullets": bool(p.get("ai_matchup") or p.get("ai_factors")),
                        "ai_verdict":     p.get("ai_verdict"),
                        "line_move_pp":   p.get("line_move_pp"),
                        "kalshi_live_prob": p.get("kalshi_live_prob"),
                        "kalshi_contra":  p.get("kalshi_contra"),
                        "whale":          bool(p.get("whale")),
                        "pick":           p.get("pick"),
                        "sport":          p.get("sport_slug"),
                    })
                continue
            mins = _minutes_to_game_start(p.get("start_time") or "")
            if mins is None or mins < 0:
                live_or_done += 1
                continue  # game live or finished
            lo, hi = _pick_sport_pregame_window(p.get("sport_slug") or "")
            eligible.append({"pid": pid, "score": score, "mins": mins,
                              "window": [lo, hi], "pick": p.get("pick"),
                              "sport": p.get("sport_slug")})
            if not (lo <= mins <= hi):
                window_blocked.append({"pid": pid, "score": score,
                                        "mins_to_start": mins,
                                        "window": [lo, hi],
                                        "pick": p.get("pick"),
                                        "sport": p.get("sport_slug")})
                continue  # not in pre-game window yet
            # Post + record
            r = post_individual_pick(p)
            if r.get("ok"):
                posted_state[pid] = {
                    "posted_at": datetime.utcnow().isoformat() + "Z",
                    "score":     score,
                    "reasons":   reasons,
                    "mins":      mins,
                    "day":       today_str,
                }
                posted_today.append({**r, "pid": pid, "pick": p.get("pick")})

    # Prune old entries so the state file stays small (keep last 14 days)
    try:
        cutoff = (today - timedelta(days=14)).isoformat()
        posted_state = {k: v for k, v in posted_state.items()
                         if (v.get("day") or "") >= cutoff}
    except Exception:
        pass
    _save_posted_state(posted_state)

    # Housekeeping: once we've gone through posts for this tick, sweep
    # the channel for any accidental duplicates (two of the same pick
    # that slipped through a dedup race in the past, or anything a
    # retry sent twice). Needs a bot token; no-op without. Capped at
    # 100 recent messages per tick so it's cheap.
    dedup_sweep = None
    try:
        import mlb_ui
        settings = mlb_ui._load_settings()
        bot = (settings.get("discord_bot_token") or "").strip()
        hook = _resolve_discord_webhook(settings)
        if bot and hook:
            import discord_bot
            channel_id = discord_bot._channel_id_from_webhook(hook)
            if channel_id:
                dedup_sweep = discord_bot.delete_duplicate_messages(
                    bot, channel_id, scan_limit=100)
    except Exception as e:
        dedup_sweep = {"ok": False, "error": f"sweep failed: {e}"}

    return {
        "ok":              True,
        "ran_at":          datetime.utcnow().isoformat() + "Z",
        "dedup_sweep":     dedup_sweep,
        "threshold":       _EVIDENCE_POST_THRESHOLD,
        "eligible_count":  len(eligible),
        "posted_count":    len(posted_today),
        "eligible":        eligible[:20],
        "posted":          posted_today,
        # Diagnostic snapshot of what the moderator saw this tick
        "board_total_picks":    board_total,
        "dnb_or_model_only":    dnb_or_model,
        "already_posted_today": already_posted,
        "evaluated":            evaluated,
        "live_or_done":         live_or_done,
        "window_blocked_count": len(window_blocked),
        "window_blocked":       window_blocked[:10],
        "below_threshold":      below_thresh,
        "score_histogram":      score_hist,       # Evidence Score (0-10) distribution
        "signal_histogram":     signal_hist,      # Signal composite (0-5) distribution
        "reason_tally":         reason_tally,     # How many picks earned each component
    }


def run_line_move_followup():
    """Agent #8 follow-up: for picks posted 90-180 min ago, check if
    DK caught up to our price (= we were ahead of the market). Posts
    a one-line update per pick that moved toward us. Free — no LLM
    needed, pure math on existing line_move_pp field (now that
    enough time has passed to see post-posting movement)."""
    try:
        import mlb_ui
        import discord_bot
        settings = mlb_ui._load_settings()
        webhook  = _resolve_discord_webhook(settings)
        if not webhook:
            return {"ok": False, "error": "no webhook"}
        today = datetime.now(mlb_ui.CENTRAL).date()
        board, _ = mlb_ui._build_board_rows(today.isoformat(), today, "all",
                                              strong_only=False)
    except Exception as e:
        return {"ok": False, "error": f"board/webhook failed: {e}"}

    posted_state = _load_posted_state()
    now = datetime.utcnow()
    confirmed = []
    for row in board:
        for p in row.get("picks") or []:
            pid = p.get("id")
            entry = posted_state.get(pid)
            if not entry:
                continue
            # Check posted-at age
            try:
                posted_at = datetime.fromisoformat(entry["posted_at"].replace("Z", ""))
                age_min = (now - posted_at).total_seconds() / 60
            except Exception:
                continue
            if age_min < 90 or age_min > 240:
                continue  # only follow up on 1.5-4 hr old posts
            if entry.get("followup_sent"):
                continue
            # Did the line move further toward us since posting?
            try:
                lm_now = float(p.get("line_move_pp") or 0.0)
                if lm_now >= 0.01:
                    confirmed.append({"pid": pid, "pick": p.get("pick"),
                                       "sport": p.get("sport_name"),
                                       "move_pp": lm_now * 100})
                    entry["followup_sent"] = True
            except Exception:
                continue

    if confirmed:
        lines = ["📈 **Line check — picks the market caught up to:**"]
        for c in confirmed:
            lines.append(f"• {c['sport']} · {c['pick']} — "
                         f"moved +{c['move_pp']:.1f}pp toward our side")
        discord_bot.post_text(webhook, "\n".join(lines))
        _save_posted_state(posted_state)

    return {"ok": True, "confirmed_count": len(confirmed),
            "confirmed": confirmed}


# ============================================================
# Agent #2 — Line-shopper (free, no LLM)
# ============================================================
# Compares our picks (priced off Pinnacle) against US-book prices
# from The Odds API (DK / FanDuel / BetMGM / Caesars). When another
# book has a materially better price (>= +1% decimal), the pick card
# gets a chip pointing the user to the better book. Also exposes a
# /cron/line-shop-scan route that batch-logs all current edges to
# logs/line_shop_findings.json for review.
#
# Zero LLM calls — pure math on prices The Odds API already provides.
# User pays only for the Odds API call quota (configured per their
# ODDS_API_KEY) which we already use for other UI chips.

_BOOK_PRIORITY = ("draftkings", "fanduel", "betmgm", "caesars")
_LINESHOP_FINDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs", "line_shop_findings.json",
)

# Minimum delta (as decimal fraction) over Pinnacle to flag an
# alt-book as "better price here". +1% delta means 1.95 vs 1.97
# — real but not noise.
_LINESHOP_MIN_EDGE = 0.01


def _market_to_odds_api_key(market):
    """Map our market labels to Odds API keys."""
    m = (market or "").lower()
    if "moneyline" in m or m == "ml": return "h2h"
    if "spread" in m or "run line" in m or "puck line" in m: return "spreads"
    if "total" in m: return "totals"
    return None


def _pick_side_label(pick):
    """Return the Odds API 'name' field value our pick corresponds to.
    For moneyline it's the team name. For spread/total it's the side
    marker ('over'/'under' for totals, team name for spreads)."""
    sel = (pick.get("pick") or "").strip()
    # Totals: pick like 'Over 2.5' / 'Under 215.5'
    low = sel.lower()
    if low.startswith("over"):  return "over"
    if low.startswith("under"): return "under"
    # Spreads / ML: team name is the first word(s) before the number
    # e.g. 'Kansas City Chiefs -3.5' → 'Kansas City Chiefs'
    import re as _re
    cleaned = _re.sub(r"\s*[-+]?\d+(\.\d+)?.*$", "", sel).strip()
    return cleaned or sel


def _match_odds_api_event(oapi_events, home, away):
    """Find the Odds API event that matches our home/away teams.
    Our pick carries home_team / away_team; Odds API event keys are
    tuples of (away_norm, home_norm) using generic_odds._team_key.
    We import that helper inline to stay backwards-compatible."""
    try:
        from generic_odds import _team_key
    except Exception:
        return None
    key = (_team_key(away or ""), _team_key(home or ""))
    return oapi_events.get(key)


def find_best_book_for_pick(pick, oapi_events=None):
    """Return the best US-book price for ONE pick:
      {book, decimal, delta_pct}  — delta_pct > 0 means better than
      the pick's current (Pinnacle) price.
    Returns None when no US book covers this market, or when every
    book's price is worse than or equal to Pinnacle's."""
    market_key = _market_to_odds_api_key(pick.get("market") or "")
    if not market_key:
        return None
    sport_key = pick.get("sport_odds_api_key") or ""
    if not sport_key:
        # Look up from sports registry
        try:
            import sports
            cfg = sports.by_slug(pick.get("sport_slug") or "")
            sport_key = (cfg or {}).get("odds_api_key") or ""
        except Exception:
            pass
    if not sport_key:
        return None

    try:
        if oapi_events is None:
            from generic_odds import fetch_odds_api
            # Request the richer market set so spreads + totals land too
            oapi_events = fetch_odds_api(sport_key, markets="h2h,spreads,totals")
    except Exception:
        return None
    if not oapi_events:
        return None

    event = _match_odds_api_event(oapi_events, pick.get("home_team"),
                                     pick.get("away_team"))
    if not event:
        return None

    side_label = _pick_side_label(pick).lower()
    books = event.get("books") or {}

    # For spreads/totals we also match on the line (e.g. -3.5 vs -3)
    want_line = None
    try:
        import re as _re
        m = _re.search(r"[-+]?\d+(\.\d+)?", pick.get("pick") or "")
        if m:
            want_line = float(m.group(0))
    except Exception:
        pass

    best = None  # (book, decimal)
    for book_name in _BOOK_PRIORITY:
        outcomes = (books.get(book_name) or {}).get(market_key) or []
        for o in outcomes:
            name = (o.get("name") or "").lower()
            price = o.get("price")
            if price is None:
                continue
            # Match side
            if market_key == "h2h":
                if side_label not in name and name not in side_label:
                    continue
            elif market_key == "totals":
                if side_label not in name:
                    continue
                if want_line is not None:
                    try:
                        if abs(float(o.get("point") or 0) - want_line) > 0.01:
                            continue
                    except Exception:
                        continue
            elif market_key == "spreads":
                if side_label not in name and name not in side_label:
                    continue
                if want_line is not None:
                    try:
                        if abs(float(o.get("point") or 0) - want_line) > 0.01:
                            continue
                    except Exception:
                        continue
            try:
                dec = float(price)
            except Exception:
                continue
            if best is None or dec > best[1]:
                best = (book_name, dec)

    if not best:
        return None
    cur = pick.get("decimal") or 0
    try:
        cur = float(cur)
    except Exception:
        cur = 0
    if cur <= 0:
        return None
    delta_pct = (best[1] / cur - 1.0) * 100
    if delta_pct < _LINESHOP_MIN_EDGE * 100:
        return None
    return {"book": best[0], "decimal": best[1], "delta_pct": round(delta_pct, 2)}


def scan_board_for_better_prices():
    """Loop today's board, find all picks where a US book beats our
    Pinnacle price by >= _LINESHOP_MIN_EDGE. Writes findings to the
    log for later review. Returns summary."""
    try:
        import mlb_ui
        today = datetime.now(mlb_ui.CENTRAL).date()
        board, _ = mlb_ui._build_board_rows(today.isoformat(), today, "all",
                                              strong_only=False)
    except Exception as e:
        return {"ok": False, "error": f"board build failed: {e}"}

    findings = []
    # Group picks by sport_key so we only fetch Odds API once per sport
    by_sport = {}
    for row in board:
        for p in row.get("picks") or []:
            if p.get("model_only") or p.get("do_not_bet"):
                continue
            if (p.get("signal_score") or 0) < 3:
                continue  # only line-shop picks we might actually place
            # Add home/away/sport context onto the pick
            p["home_team"] = p.get("home_team") or row.get("home")
            p["away_team"] = p.get("away_team") or row.get("away")
            p["sport_slug"] = p.get("sport_slug") or row.get("sport_slug")
            by_sport.setdefault(p["sport_slug"], []).append(p)

    try:
        import sports
        from generic_odds import fetch_odds_api
    except Exception:
        return {"ok": False, "error": "sports/generic_odds import failed"}

    for slug, picks in by_sport.items():
        cfg = sports.by_slug(slug or "")
        if not cfg:
            continue
        sport_key = cfg.get("odds_api_key")
        if not sport_key:
            continue
        try:
            events = fetch_odds_api(sport_key, markets="h2h,spreads,totals")
        except Exception:
            events = {}
        for p in picks:
            best = find_best_book_for_pick(p, oapi_events=events)
            if best:
                findings.append({
                    "pid":        p.get("id"),
                    "sport":      slug,
                    "match":      f"{p.get('away_team')} @ {p.get('home_team')}",
                    "market":     p.get("market"),
                    "pick":       p.get("pick"),
                    "our_price":  p.get("decimal"),
                    "best_book":  best["book"],
                    "best_price": best["decimal"],
                    "delta_pct":  best["delta_pct"],
                })

    # Sort by delta descending — biggest edges first
    findings.sort(key=lambda f: -f["delta_pct"])

    # Save to log
    try:
        payload = {
            "ran_at":   datetime.utcnow().isoformat() + "Z",
            "findings": findings,
        }
        os.makedirs(os.path.dirname(_LINESHOP_FINDINGS_PATH), exist_ok=True)
        with open(_LINESHOP_FINDINGS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        try:
            import log_persist
            log_persist.write_file("logs/line_shop_findings.json",
                                     json.dumps(payload, indent=2))
        except Exception:
            pass
    except Exception:
        pass

    return {"ok":            True,
            "count":         len(findings),
            "sports_scanned":list(by_sport.keys()),
            "top_findings":  findings[:10]}


WEEKLY_AUDIT_PROMPT = """You are auditing our Discord moderator's picks over the last 7
days. Evidence filter posts only picks with score >= 5 (out of 10),
timed to each sport's pre-game window.

RESULT DATA:
{result_table}

Reply in <=5 sentences:
1. Did the evidence filter pay off? (compare A-tier posted record vs
   the broader board record)
2. Which evidence component correlated most with wins? (Line, AI,
   Kalshi, Whale, News-clean, Sig4+)
3. Any adjustments to the threshold (currently 5) you'd suggest?
4. One honest sentence on sample size / confidence.
"""


def run_weekly_moderator_audit():
    """Sunday-ish agent: audits last 7 days of individually-posted
    picks. Haiku composes a short report. Posts to Discord as a
    footnote-style message."""
    try:
        import mlb_ui
        import plays_log
        import discord_bot
        settings = mlb_ui._load_settings()
        webhook  = _resolve_discord_webhook(settings)
    except Exception as e:
        return {"ok": False, "error": f"import failed: {e}"}

    posted_state = _load_posted_state()
    cutoff = (datetime.utcnow().date() - timedelta(days=7)).isoformat()
    posted_ids = [pid for pid, v in posted_state.items()
                  if (v.get("day") or "") >= cutoff]
    if not posted_ids:
        return {"ok": True, "audit": "no posts in the last 7 days"}

    try:
        graded_by_date = plays_log.grade_all(limit_dates=14) or {}
    except Exception:
        graded_by_date = {}
    all_picks = []
    for day_picks in graded_by_date.values():
        all_picks.extend(day_picks or [])
    by_id = {p.get("id"): p for p in all_picks if p.get("id")}

    # Build a compact table of posted + graded picks per component
    rows = []
    totals = {"w": 0, "l": 0, "u": 0.0}
    comp_hits = {}
    for pid in posted_ids:
        p = by_id.get(pid)
        if not p or p.get("result") not in ("W", "L", "P"):
            continue
        rows.append(p)
        if p["result"] == "W": totals["w"] += 1
        elif p["result"] == "L": totals["l"] += 1
        try: totals["u"] += float(p.get("profit_u") or 0.0)
        except Exception: pass
        for comp in (posted_state.get(pid, {}).get("reasons") or []):
            c = comp_hits.setdefault(comp, {"n": 0, "w": 0})
            c["n"] += 1
            if p["result"] == "W":
                c["w"] += 1

    table = (f"Overall posted: {totals['w']}-{totals['l']}  "
             f"{totals['u']:+.2f}u  ({len(rows)} graded)\n"
             f"\nBy component (hits / sample):\n")
    for comp, c in sorted(comp_hits.items(), key=lambda kv: -kv[1]["n"]):
        rate = (c["w"] / c["n"] * 100) if c["n"] else 0
        table += f"  {comp:12s} {c['w']}/{c['n']}  {rate:.0f}%\n"

    prompt = WEEKLY_AUDIT_PROMPT.format(result_table=table)
    reply = _call_claude(prompt, model="claude-haiku-5-5",
                           max_tokens=400,
                           system="You audit sports-betting models honestly. Dry tone.")
    if webhook and reply:
        discord_bot.post_text(webhook,
            f":bar_chart: **Weekly moderator audit**\n\n```\n{table}```\n\n{reply}")
    return {"ok": True, "data": table, "analysis": reply or "(no reply)"}


# ============================================================
# Agent #9 — Live board grader (rule-based, no LLM)
# ============================================================
# Walks today's live/final rows, matches to ESPN running scores,
# and advances pick results that can be settled deterministically
# BEFORE the game ends:
#   - Over X.5: current total > X.5     → W
#   - Under X.5: game ended + total < X.5 → W (full-time only)
#   - Team Total Over X.5: team > X.5   → W (early-settle)
#   - BTTS Yes: both teams scored       → W (early-settle)
#   - Period markets (1H/F5/P1): once that period closes
#
# Writes to plays_log so the records view picks it up alongside the
# post-game grader. Called from /cron/dispatch so it runs every tick.
# ============================================================

import re as _re_lg


def _parse_total_line(pick_text):
    """Pull ('over'|'under', line_float) from a pick like 'Over 2.5'."""
    m = _re_lg.search(r"(over|under)\s+([0-9]+(?:\.[0-9]+)?)",
                       (pick_text or "").lower())
    if not m:
        return None, None
    return m.group(1), float(m.group(2))


def _settle_early(pick, home_score, away_score, period_scores=None):
    """Return 'W' if this pick is deterministically a win given the
    current scores, else None. Only returns W; L/P wait for full-time
    because an Over can still cash late. Period_scores is a list of
    {num, home, away} dicts for sports with linescores (NFL/NBA/NHL)."""
    market = (pick.get("market") or "").lower()
    pick_text = (pick.get("pick") or "")
    pick_lc = pick_text.lower()
    try:
        h = int(home_score or 0)
        a = int(away_score or 0)
    except (TypeError, ValueError):
        return None
    tot = h + a

    # Full-game Total
    if "total" in market and "team" not in market and "1h" not in market and "p1" not in market:
        side, line = _parse_total_line(pick_text)
        if side == "over" and line is not None and tot > line:
            return "W"

    # Team Total
    if "team total" in market or "tt" in market:
        side, line = _parse_total_line(pick_text)
        if side == "over" and line is not None:
            # Identify which team the pick is for by name presence
            ht = (pick.get("home_team") or "").lower()
            at = (pick.get("away_team") or "").lower()
            team_score = None
            if ht and ht in pick_lc:
                team_score = h
            elif at and at in pick_lc:
                team_score = a
            if team_score is not None and team_score > line:
                return "W"

    # BTTS Yes — settle as soon as both teams have scored
    if "btts" in market and "yes" in pick_lc:
        if h >= 1 and a >= 1:
            return "W"

    # 1H / P1 / F5 Over — once that period closes with a cleared total
    if period_scores:
        if "1h" in market or "first half" in market or "p1" in market or "period 1" in market or "f5" in market:
            # 1H = first 2 quarters (NFL/NBA), first period (NHL), F5 (MLB)
            # We only have confident 1H/P1 data when the period is complete.
            # For 1H in football/basketball: use first 2 linescores if present.
            # For P1 (NHL): first linescore.
            # For F5 (MLB): sum of innings 1-5 if 5+ innings played.
            side, line = _parse_total_line(pick_text)
            if side == "over" and line is not None:
                try:
                    if "1h" in market or "first half" in market:
                        # NFL/NBA 1H = 2 periods
                        if len(period_scores) >= 2:
                            p1h = sum(int(ps.get("home") or 0) + int(ps.get("away") or 0)
                                       for ps in period_scores[:2])
                            if p1h > line:
                                return "W"
                    elif "p1" in market or "period 1" in market:
                        if len(period_scores) >= 1:
                            p1 = (int(period_scores[0].get("home") or 0)
                                   + int(period_scores[0].get("away") or 0))
                            # Only settle if the period is actually CLOSED
                            # (period 2 has started or period 1's score is
                            # the final one).
                            closed = len(period_scores) >= 2
                            if closed and p1 > line:
                                return "W"
                    elif "f5" in market:
                        if len(period_scores) >= 5:
                            f5 = sum(int(ps.get("home") or 0) + int(ps.get("away") or 0)
                                      for ps in period_scores[:5])
                            if f5 > line:
                                return "W"
                except (TypeError, ValueError):
                    pass
    return None


_LIVE_SCORES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "logs", "live_scores.json")


def _write_live_scores(all_scores):
    """Dump every matched (sport, home, away) → (home_score, away_score)
    to logs/live_scores.json so the board can overlay them without
    doing its own ESPN fetch + fuzzy match. Called by run_live_grader."""
    try:
        os.makedirs(os.path.dirname(_LIVE_SCORES_PATH), exist_ok=True)
        payload = {"ran_at": datetime.utcnow().isoformat() + "Z",
                   "scores": all_scores}
        with open(_LIVE_SCORES_PATH, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def run_live_grader():
    """Scan today's picks against ESPN live scores and advance any
    pick that can be settled early (Over/BTTS/TT Over hit mid-game).
    Writes results back to plays_log so /logged + the daily P&L
    panel reflect them immediately. ALSO dumps every match to
    logs/live_scores.json so the board can overlay scores without
    repeating the ESPN fetch + fuzzy match. Returns per-sport counts.

    Rule-based — no LLM calls. Runs on /cron/dispatch so it ticks
    every 5 min alongside the moderator."""
    import mlb_ui
    import plays_log
    import sports as _sports_lg

    today = datetime.now(mlb_ui.CENTRAL).date()
    today_iso = today.isoformat()
    graded_file = plays_log.read_graded(today_iso) or {}
    graded_picks = graded_file.get("picks") or []
    # NOTE: we still proceed to the live-score dump below even when
    # graded_picks is empty, so the board gets its score overlay on
    # days where no picks have been logged yet. Just skip the early-
    # settle loop in that case.
    # Integrity sweep (shipped 2026-10-24): any pick that was graded
    # BEFORE its game started is a bogus result (grader hit a stale
    # ESPN event or an ID collision). Rewind those back to pending so
    # the regular grader can take another pass once the game actually
    # ends. We identify them by comparing start_time to now_dt_utc:
    # if start_time is in the future (or missing) but result is set,
    # ungrade.
    pre_kickoff_fixed = []
    now_utc = datetime.utcnow()
    for p in graded_picks:
        if p.get("result") not in ("W", "L", "P"):
            continue
        st = p.get("start_time") or ""
        if not st:
            continue
        try:
            start_dt = datetime.fromisoformat(st.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        # Allow a small grace window after kickoff before considering
        # a grade legitimate — a pick settled 10+ min after kickoff
        # is fine (early-settle on Over hit, etc.); anything graded
        # BEFORE kickoff is wrong.
        if start_dt > now_utc:
            prev_result = p.get("result")
            p["result"] = None
            p["profit_u"] = None
            p["graded_at"] = None
            p["graded_by"] = "integrity_sweep_rewind"
            pre_kickoff_fixed.append({
                "id":     p.get("id"),
                "pick":   p.get("pick"),
                "sport":  p.get("sport"),
                "was":    prev_result,
                "kickoff": start_dt.isoformat() + "Z",
            })
    if pre_kickoff_fixed:
        try:
            plays_log.write_graded(today_iso,
                                     {"date": today_iso, "picks": graded_picks})
            plays_log.invalidate_grade_all_cache()
        except Exception:
            pass

    # Group by sport for ESPN fetch batching
    by_sport = {}
    for p in graded_picks:
        if p.get("result") in ("W", "L", "P"):
            continue  # already settled
        slug = p.get("sport") or ""
        by_sport.setdefault(slug, []).append(p)
    settled = []
    # Early-settle loop (only runs when we have pending picks)
    for slug, picks in by_sport.items():
        sport = _sports_lg.by_slug(slug)
        if not sport:
            continue
        espn_path = sport.get("espn_scoreboard")
        if not espn_path:
            continue
        try:
            events = plays_log.fetch_scores_espn(espn_path,
                                                   days_back=1,
                                                   ref_date=today)
        except Exception:
            continue
        # Index events by team name (fuzzy)
        def _toks(name):
            n = (name or "").lower().strip()
            if not n: return set()
            t = {n}
            for w in n.split():
                if len(w) >= 4: t.add(w)
            return t
        _by_tok = {}
        for ev in events:
            ht = ev.get("home_team") or ""
            at = ev.get("away_team") or ""
            if not ht or not at: continue
            for t in _toks(ht):
                _by_tok.setdefault(t, []).append(("home", at, ht, ev))
            for t in _toks(at):
                _by_tok.setdefault(t, []).append(("away", ht, at, ev))
        for pick in picks:
            ht = pick.get("home_team") or ""
            at = pick.get("away_team") or ""
            if not ht or not at:
                continue
            ht_toks = _toks(ht)
            at_toks = _toks(at)
            found_ev = None
            for tok in ht_toks:
                for role, opp, _me, ev in _by_tok.get(tok, []):
                    if any(t in _toks(opp) for t in at_toks):
                        # Pull home/away scores from event
                        scores = ev.get("scores") or []
                        if len(scores) < 2:
                            continue
                        try:
                            if role == "home":
                                hs = int(float(scores[0].get("score") or 0))
                                ascore = int(float(scores[1].get("score") or 0))
                            else:
                                ascore = int(float(scores[0].get("score") or 0))
                                hs = int(float(scores[1].get("score") or 0))
                        except (TypeError, ValueError):
                            continue
                        found_ev = (ev, hs, ascore)
                        break
                if found_ev:
                    break
            if not found_ev:
                continue
            ev, hs, ascore = found_ev
            # Try early-settle
            period_scores = ev.get("innings") or []
            res = _settle_early(pick, hs, ascore,
                                 period_scores=period_scores)
            if res == "W":
                pick["result"] = "W"
                try:
                    dec = float(pick.get("decimal") or 1.0)
                except (TypeError, ValueError):
                    dec = 1.0
                pick["profit_u"] = dec - 1.0 if dec > 1.0 else 0.0
                pick["graded_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                pick["graded_by"] = "live_grader_early"
                settled.append({
                    "id":    pick.get("id"),
                    "pick":  pick.get("pick"),
                    "sport": slug,
                    "score": f"{ascore}-{hs}",
                })
    # Persist if we settled anything
    if settled:
        try:
            plays_log.write_graded(today_iso,
                                     {"date": today_iso, "picks": graded_picks})
        except Exception:
            pass
        try:
            plays_log.invalidate_grade_all_cache()
        except Exception:
            pass

    # Also run the regular post-game backfill so pending picks from
    # the last couple of days get re-graded on this tick. Keeps the
    # /logged tab's records fresh without the user having to visit
    # the page. Cheap — grade_pending_backfill short-circuits games
    # that haven't ended yet.
    try:
        plays_log.grade_pending_backfill(days_back=3, grace_hours=1)
    except Exception:
        pass

    # Live-score dump (broad). Even sports with 0 pending picks (e.g.
    # a game whose pick already settled) may need live scores for the
    # board's row display. Walk every sport that has a scoreboard
    # path, pull ESPN, dump every event's score keyed by sport/home-
    # token/away-token to logs/live_scores.json. The board reads this
    # file to overlay scores on its rows without redoing the ESPN
    # fetch + fuzzy match itself (decouples the board build from
    # per-sport matcher bugs).
    try:
        def _toks_write(name):
            n = (name or "").lower().strip()
            if not n:
                return []
            out = {n}
            for w in n.split():
                if len(w) >= 4:
                    out.add(w)
            return sorted(out)
        live_dump = {}  # sport_slug -> list of {home, away, home_score, away_score, home_tokens, away_tokens}
        for sport in _sports_lg.SPORTS:
            path = sport.get("espn_scoreboard")
            if not path:
                continue
            try:
                evs = plays_log.fetch_scores_espn(path, days_back=1,
                                                    ref_date=today)
            except Exception:
                continue
            per_sport = []
            for ev in evs:
                ht = ev.get("home_team") or ""
                at = ev.get("away_team") or ""
                scs = ev.get("scores") or []
                if not ht or not at or len(scs) < 2:
                    continue
                try:
                    hs = int(float(scs[0].get("score") or 0))
                    as_ = int(float(scs[1].get("score") or 0))
                except (TypeError, ValueError):
                    continue
                per_sport.append({
                    "home":          ht,
                    "away":          at,
                    "home_score":    hs,
                    "away_score":    as_,
                    "home_tokens":   _toks_write(ht),
                    "away_tokens":   _toks_write(at),
                })
            if per_sport:
                live_dump[sport["slug"]] = per_sport
        _write_live_scores(live_dump)
    except Exception:
        pass

    return {
        "ok":                True,
        "today":             today_iso,
        "sports":            list(by_sport.keys()),
        "pending_count":     sum(len(v) for v in by_sport.values()),
        "settled_count":     len(settled),
        "settled":           settled,
        "pre_kickoff_fixed": pre_kickoff_fixed,
        "pre_kickoff_count": len(pre_kickoff_fixed),
    }
