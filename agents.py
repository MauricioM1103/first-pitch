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
