#!/usr/bin/env python3
"""AI analyst — Claude-driven structured matchup analysis.

Calls Anthropic's API with the data we already fetch (model probability,
pitcher lines, Pinnacle lines, Monte Carlo projection) and asks Claude
for a tight, structured read: stat take, matchup factors, risk flags,
and a pick with confidence.

Requires ANTHROPIC_API_KEY env var. If missing, the module returns a
stub explaining what to configure.

Design is extensible: `analyze_game` dispatches to a provider function
chosen by provider= argument. Right now only "claude" is wired, but
"openai", "google", etc. can be added without changing callers.
"""
import json
import os
import time

_cache = {}  # (provider, game_pk) -> (analysis_dict, timestamp)
_CACHE_TTL_S = 6 * 3600  # 6 hours per game
_CACHE_MAX = 32          # cap entries so free-tier memory stays sane

CLAUDE_MODEL = "claude-sonnet-5-5"  # Current Sonnet; swap to claude-opus-5-5 for depth
# Enough headroom for the structured JSON schema (4 arrays × 3-4 bullets +
# bullet text). 900 was tight — responses were getting truncated mid-string
# and json.loads blew up before the ai-pick field even landed.
CLAUDE_MAX_TOKENS = 2200


def is_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _build_prompt(game):
    """Serialize the game context into a prompt. `game` is the dict from get_games."""
    a = game["away"]
    h = game["home"]
    ap = game["away_pitcher"]
    hp = game["home_pitcher"]
    o = game.get("odds") or {}
    pin = o.get("pinnacle") or {}
    ml = pin.get("moneyline") or {}
    tot = pin.get("total") or {}

    def _am(v):
        if v is None:
            return "—"
        return f"+{v}" if v > 0 else str(v)

    # Model probability
    p_home = game.get("p_home")
    model_line = (
        f"Elo+SP model: home {int(round(p_home*100))}%" if p_home is not None else "—"
    )

    # Pinnacle devigged fair
    pin_fair = o.get("pin_fair") or {}
    pin_fair_line = (
        f"Pinnacle fair: home {int(round(pin_fair['home']*100))}%  away {int(round(pin_fair['away']*100))}%"
        if pin_fair and pin_fair.get("home") else "—"
    )

    weather = ""
    if game.get("wx_temp"):
        weather = f"{game['wx_temp']}°"
        if game.get("wx_condition"):
            weather += f", {game['wx_condition'].lower()}"
        if game.get("wx_wind"):
            weather += f", wind {game['wx_wind']}"

    return f"""Matchup: {a['team']} at {h['team']}, first pitch {game['first_pitch']} at {game.get('venue','')}.

TEAM CONTEXT:
- {a['team']}: {a.get('wins','?')}-{a.get('losses','?')}, streak {a.get('streak_code','—')}, OPS {a.get('ops','?')}, team ERA {a.get('team_era','?')}, run diff {a.get('run_diff','?')}
- {h['team']}: {h.get('wins','?')}-{h.get('losses','?')}, streak {h.get('streak_code','—')}, OPS {h.get('ops','?')}, team ERA {h.get('team_era','?')}, run diff {h.get('run_diff','?')}

STARTING PITCHERS:
- Away: {ap['name']} ({ap.get('hand','?')}HP) — {ap.get('wl','?')} record, {ap.get('era','?')} ERA, {ap.get('whip','?')} WHIP, {ap.get('k9','?')} K/9, {ap.get('ip','?')} IP
- Home: {hp['name']} ({hp.get('hand','?')}HP) — {hp.get('wl','?')} record, {hp.get('era','?')} ERA, {hp.get('whip','?')} WHIP, {hp.get('k9','?')} K/9, {hp.get('ip','?')} IP

MARKET AND MODEL:
- {model_line}
- {pin_fair_line}
- Pinnacle moneyline: home {_am(ml.get('home_am'))}, away {_am(ml.get('away_am'))}
- Pinnacle total: {tot.get('line','—')} (over {_am(tot.get('over_am'))}, under {_am(tot.get('under_am'))})
- Weather: {weather or 'indoor/unknown'}

Respond with a tight, structured analysis in this exact JSON format:

{{
  "stat_read": ["<= 20 words", "<= 20 words", "<= 20 words"],
  "matchup_factors": ["<= 25 words", "<= 25 words", "<= 25 words"],
  "risk_flags": ["<= 20 words", "<= 20 words"],
  "pick": {{
    "market": "ML" | "Total" | "Run Line",
    "side": "home" | "away" | "over" | "under",
    "rationale": "<= 30 words",
    "confidence": 1 | 2 | 3
  }}
}}

Be concrete and honest. If the market agrees with the model, say so and lean lean. If model and market disagree, pick the side with better supporting signals or say "no play" by returning confidence 1 with rationale explaining the disagreement."""


def _call_claude(prompt):
    """Call Anthropic API; return parsed JSON dict or {'error': ...}."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return {"error": "anthropic SDK not installed. pip install anthropic"}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set — add it in Render Settings → Environment."}
    try:
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=(
                "You are a sharp baseball analyst. Prioritize concrete, testable "
                "observations over narrative. Keep every bullet focused on what "
                "actually moves the win probability. Prefer specifying when the "
                "model and market agree vs disagree. Return ONLY valid JSON — no "
                "preamble, no code fences, no commentary."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude 5 may return a ThinkingBlock before the text block — pick the
        # first block that actually has a .text attribute (type == "text").
        text = ""
        for block in (msg.content or []):
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                break
        if not text:
            # Last-ditch: any block exposing .text
            for block in (msg.content or []):
                t = getattr(block, "text", None)
                if t:
                    text = t
                    break
    except Exception as e:
        return {"error": f"Claude API error: {e}"}

    # Strip code fences if the model added them anyway
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Normal path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Repair path: Claude occasionally hits max_tokens mid-string. Close the
    # open string, drop the trailing partial element, then close any open
    # arrays and objects by counting unmatched brackets.
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Last resort: surface Claude's partial reply as plain narrative so the
    # user still sees the analysis instead of a scary error. The pick-card
    # renderer handles a `narrative` field as a fallback.
    salvage = _salvage_narrative_from_partial(text)
    if salvage:
        return {"narrative": salvage, "partial": True, "model": CLAUDE_MODEL}
    return {"error": "Model returned non-JSON and no partial text could be salvaged.",
            "raw": text[:500]}


def _repair_truncated_json(s):
    """Close trailing open string/array/object in a truncated JSON payload.
    Returns the repaired string or None if it doesn't look salvageable."""
    if not s or not s.lstrip().startswith("{"):
        return None
    # Walk the string, track string state and bracket stack
    stack = []        # stack of '{' or '['
    in_str = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_str:
                escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    out = s
    if in_str:
        out += '"'  # close the unterminated string
    # Trim trailing comma or bare partial token before closing containers
    out = out.rstrip().rstrip(",").rstrip()
    # If we trimmed back into a key like `"foo":` with no value, give it one
    if out.endswith(":"):
        out += ' ""'
    # Close containers in reverse
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def _salvage_narrative_from_partial(s):
    """Pull human-readable bullets out of a partially-formed JSON reply so the
    user sees SOMETHING useful instead of a parser error."""
    import re
    bullets = re.findall(r'"([^"\\]{20,})"', s or "")
    if not bullets:
        return None
    # De-dup while preserving order; cap at ~8 bullets
    seen = set(); out = []
    for b in bullets:
        if b in seen or b.lower() in ("stat_read", "matchup_factors",
                                       "risk_flags", "ai_pick", "rationale",
                                       "narrative", "pick", "market", "side",
                                       "confidence"):
            continue
        seen.add(b)
        out.append(b)
        if len(out) >= 8:
            break
    return "\n\n".join("• " + b for b in out) if out else None


def analyze_game(game, provider="claude", force_refresh=False):
    """Return analysis dict for a given game dict (from get_games)."""
    key = (provider, game.get("game_pk"))
    now = time.time()
    if not force_refresh:
        hit = _cache.get(key)
        if hit and now - hit[1] < _CACHE_TTL_S:
            return hit[0]

    if provider != "claude":
        return {"error": f"provider '{provider}' not supported yet; use 'claude'"}

    prompt = _build_prompt(game)
    result = _call_claude(prompt)
    result["generated_at"] = time.strftime("%I:%M %p ET", time.localtime())
    result["model"] = CLAUDE_MODEL
    _cache_put(key, result, now)
    return result


def _cache_put(key, value, now):
    """Insert with LRU-ish eviction so the cache stays below _CACHE_MAX."""
    _cache[key] = (value, now)
    if len(_cache) > _CACHE_MAX:
        # Drop oldest entry
        oldest = min(_cache, key=lambda k: _cache[k][1])
        _cache.pop(oldest, None)


# ============================================================================
# Sport-agnostic analyst for picks (NFL, soccer, UFC, UCL, etc.)
# ============================================================================

def _build_pick_prompt(pick):
    """Build an analysis prompt from a picks.collect_picks() entry.

    This works across every sport since the pick dict is uniform. Sport-
    specific color goes in the pick's model_source field.
    """
    market_implied = (1.0 / pick["decimal"]) * 100 if pick.get("decimal") else None
    fair_pct = (pick.get("fair_prob") or 0) * 100

    return f"""Sport: {pick.get('sport', '?')}
Matchup: {pick.get('game', '?')}
{'First pitch / kickoff: ' + pick['start_time'] if pick.get('start_time') else ''}

THE PICK:
  Market  : {pick.get('market', '?')}
  Side    : {pick.get('pick', '?')}
  Price   : {('+' if (pick.get('american') or 0) > 0 else '')}{pick.get('american', '—')} ({pick.get('decimal', '?')} decimal) @ {pick.get('book', '?')}

MODEL vs MARKET:
  Fair prob (model)      : {fair_pct:.1f}%
  Implied prob (market)  : {market_implied:.1f}% ({market_implied - fair_pct:+.1f} pt gap vs fair)
  EV                     : +{pick.get('ev_pct', 0):.2f}%
  Quarter-Kelly stake    : {pick.get('kelly_pct', 0):.2f}% of bankroll

MODEL SOURCE: {pick.get('model_source', '?')}

Respond with a tight, structured analysis in this exact JSON format:

{{
  "stat_read": ["<= 20 words", "<= 20 words", "<= 20 words"],
  "matchup_factors": ["<= 25 words", "<= 25 words", "<= 25 words"],
  "risk_flags": ["<= 20 words", "<= 20 words"],
  "pick": {{
    "market": "<market>",
    "side": "<side>",
    "rationale": "<= 30 words",
    "confidence": 1 | 2 | 3
  }}
}}

The pick field should either CONFIRM the shown pick with your rationale, or set confidence to 1 with a rationale explaining why you'd fade it. Return ONLY valid JSON — no preamble, no code fences, no commentary."""


def analyze_pick(pick, provider="claude", force_refresh=False):
    """Return structured analysis for a pick dict from picks.collect_picks()."""
    key = (provider, "pick", pick.get("id"))
    now = time.time()
    if not force_refresh:
        hit = _cache.get(key)
        if hit and now - hit[1] < _CACHE_TTL_S:
            return hit[0]
    if provider != "claude":
        return {"error": f"provider '{provider}' not supported yet"}
    prompt = _build_pick_prompt(pick)
    result = _call_claude(prompt)
    result["generated_at"] = time.strftime("%I:%M %p ET", time.localtime())
    result["model"] = CLAUDE_MODEL
    _cache_put(key, result, now)
    return result
