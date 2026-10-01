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

CLAUDE_MODEL = "claude-sonnet-4-5"  # Fast + inexpensive; swap to opus for depth
CLAUDE_MAX_TOKENS = 900


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
        text = msg.content[0].text if msg.content else ""
    except Exception as e:
        return {"error": f"Claude API error: {e}"}

    # Strip code fences if the model added them anyway
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"Model returned non-JSON: {e}", "raw": text[:500]}


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
    _cache[key] = (result, now)
    return result
