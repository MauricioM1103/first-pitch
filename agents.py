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
            webhook = (settings.get("daily_picks_webhook_url") or "").strip()
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

    # Only new-or-worsened items trigger LLM analysis, and we cap
    # LLM calls per day. Items are sorted by severity so the cap
    # eats the highest-impact ones first.
    worth_llm = [c for c in changes if c["change"] in ("new", "worsened")]
    worth_llm.sort(key=lambda c: -_STATUS_RANK.get((c["status"] or "").lower(), 1))

    analyzed = []
    remaining = _NEWS_LLM_DAILY_CAP - int(daily.get("llm_calls") or 0)
    for item in worth_llm[:max(0, remaining)]:
        # Only summarize for status at Questionable or worse
        if _STATUS_RANK.get((item["status"] or "").lower(), 0) < 2:
            continue
        prompt = (
            f"One-sentence (<=25 words) impact assessment for today's bets:\n"
            f"Sport: {item['sport'].upper()}\n"
            f"Team: {item['team']}\n"
            f"Player: {item['player']}\n"
            f"Status change: {item['prev_status']} → {item['status']}\n"
            f"Details: {item['desc'] or '(none)'}\n\n"
            f"Reply with ONE sentence focused on how this affects "
            f"betting lines on their next game (if at all). Be specific; "
            f"name the market most affected (ML / spread / total / "
            f"team total / props). If low impact, say so plainly."
        )
        reply = _call_claude(prompt, model="claude-haiku-5-5",
                               max_tokens=120)
        # Only count a call toward the daily cap when the API actually
        # returned text (not None from a missing key / SDK import fail,
        # and not a __ERROR__ placeholder from a caught exception).
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
        "ok":           True,
        "ran_at":       state["last_run"],
        "sports_checked": _todays_sports(),
        "changes_total":  len(changes),
        "worsened":       len([c for c in changes if c["change"] == "worsened"]),
        "new":            len([c for c in changes if c["change"] == "new"]),
        "llm_calls_today":daily["llm_calls"],
        "llm_cap":        _NEWS_LLM_DAILY_CAP,
        "analyzed":       analyzed,
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
