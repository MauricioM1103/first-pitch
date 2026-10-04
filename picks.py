#!/usr/bin/env python3
"""Daily picks aggregator across every sport.

Collects positive-EV plays from every sport's model + Pinnacle devig,
ranks them, and writes a short template-based bulletin per pick. The
AI Analyst (analyst.py) can be invoked per pick for a longer narrative
on demand.

Signal sources per sport:
  * MLB     — Elo + SP + Poisson model; EV = model_prob × book_decimal − 1
              against the best available US book price
  * NFL     — Elo + QB model (12-season fit); same EV formula
  * Soccer  — 3-way Elo per league; 3 separate EV calculations
  * UFC / UCL / Europa / Int'l — no bespoke model; falls back to
              Pinnacle devig as "fair" (pure line-shop EV)
"""
import mlb_odds
import generic_odds
import sports


def _mlb_bets(date_str):
    """Collect MLB +EV picks via existing get_games pipeline."""
    try:
        from mlb_ui import get_games
    except ImportError:
        return []
    try:
        games = get_games(date_str)
    except Exception:
        return []
    out = []
    for g in games:
        o = g.get("odds") or {}
        for b in o.get("bets") or []:
            # Probability-first: let picks through on probability, filter later
            # Prefer best US book for full-game markets; else Pinnacle
            best_dec = (o.get("best_decimal") or {}).get(b["side"]) if b["market"] in ("ML", "Run Line", "Total") else None
            best_book = (o.get("best_book") or {}).get(b["side"]) if b["market"] in ("ML", "Run Line", "Total") else None
            use_dec = best_dec if (best_dec and best_dec > (b["decimal"] or 0)) else b["decimal"]
            use_book = best_book if (best_dec and best_dec > (b["decimal"] or 0)) else "pinnacle"
            # Recompute EV against the chosen book
            ev_pct = (b["fair_prob"] * use_dec - (1 - b.get("push_prob", 0))) * 100 if use_dec and b.get("fair_prob") else b["ev_pct"]
            # MLB bets path doesn't carry a devigged Pinnacle prob yet; use the
            # best-available decimal's inverse as a sharp-ish approximation.
            # The Pinnacle moneyline is already low-vig so this is close.
            pin_dec = b.get("decimal") or 0
            pin_approx = (1.0 / pin_dec) if pin_dec > 1.0 else b.get("fair_prob")
            out.append({
                "id": f"mlb_{g.get('game_pk')}_{b['market']}_{b['side']}",
                "sport": "MLB",
                "sport_slug": "mlb",
                "game": f"{g['away']['team']} at {g['home']['team']}",
                "start_time": g.get("first_pitch_utc") or g.get("first_pitch"),
                "market": b["market"],
                "pick": b["pick"],
                "fair_prob": b["fair_prob"],
                "pinnacle_prob": b.get("pinnacle_prob") or pin_approx,
                "decimal": use_dec,
                "american": mlb_odds.decimal_to_american(use_dec) if use_dec else b.get("american"),
                "book": use_book,
                "ev_pct": ev_pct,
                "kelly_pct": b["kelly_pct"],
                "model_source": "Elo + starting-pitcher model (fit on 12 MLB seasons)",
                "home_team": g["home"]["team"],
                "away_team": g["away"]["team"],
                "venue": g.get("venue", ""),
                "p_home_model": g.get("p_home"),
            })
    return out


def _nfl_model_prob_fn():
    # Use the lightweight final-state loader so we don't pull the full
    # multi-season predictions list into memory on every picks render.
    try:
        import nfl_model
        state = nfl_model.get_final_state()
        final_elo = (state or {}).get("final_elo", {})
        def _fn(g):
            h = nfl_model.abbr_from_name(g.get("home_name", ""))
            a = nfl_model.abbr_from_name(g.get("away_name", ""))
            if not h or not a:
                return None
            h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
            a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
            p = nfl_model.predict_win_prob(h_elo, a_elo)
            return {"home": p, "away": 1 - p, "draw": None}
        return _fn
    except Exception:
        return None


def _soccer_model_prob_fn(slug):
    try:
        import soccer_model
        final_elo = soccer_model.get_final_elo(slug) or {}
        def _fn(g):
            h_name = g.get("home_name", "")
            a_name = g.get("away_name", "")
            h_elo = final_elo.get(h_name, soccer_model.INITIAL_ELO)
            a_elo = final_elo.get(a_name, soccer_model.INITIAL_ELO)
            p_h, p_d, p_a = soccer_model.predict_3way(h_elo, a_elo)
            return {"home": p_h, "away": p_a, "draw": p_d}
        return _fn
    except Exception:
        return None


def _model_source(slug):
    labels = {
        "mlb":    "Elo + SP model (12-season fit)",
        "nfl":    "Elo + QB model (12-season fit)",
        "ncaaf":  "Normal-dist scoring from CFBD-fit team Elo",
        "nhl":    "Poisson goals + Elo (5-season fit)",
        "epl":    "3-way Elo + Dixon-Coles (12-season fit)",
        "laliga": "3-way Elo + Dixon-Coles (12-season fit)",
        "ligamx": "3-way Elo + Dixon-Coles (12-season fit)",
        "ucl":    "Dixon-Coles sim with EPL-prior goal rates",
        "europa": "Dixon-Coles sim with EPL-prior goal rates",
        "international": "Dixon-Coles sim with league-average goal rates",
        "ufc":    "Pinnacle devig only (fighter-level model TBD)",
    }
    return labels.get(slug, "Pinnacle devig")


_SOCCER_WITH_MODEL = {"epl", "laliga", "ligamx"}


def _sport_bets(sport):
    """Collect +EV picks for a non-MLB sport."""
    slug = sport["slug"]
    if slug == "nfl":
        mp_fn = _nfl_model_prob_fn()
    elif slug in _SOCCER_WITH_MODEL:
        mp_fn = _soccer_model_prob_fn(slug)
    else:
        mp_fn = None
    try:
        games = generic_odds.build_sport_games(sport, model_prob_fn=mp_fn)
    except Exception:
        return []
    out = []
    for g in games:
        for b in g.get("bets") or []:
            # Probability-first: let picks through on probability, not EV sign
            out.append({
                "id": f"{slug}_{g.get('matchup_id')}_{b['market']}_{b['side']}",
                "sport": sport["name"],
                "sport_slug": slug,
                "game": f"{g['away_name']} at {g['home_name']}",
                "start_time": g.get("start_time"),
                "market": b["market"],
                "pick": b["pick"],
                "fair_prob": b["fair_prob"],
                "pinnacle_prob": b.get("pinnacle_prob"),
                "decimal": b["book_decimal"] or b["pin_decimal"],
                "american": b["book_american"] or b["pin_american"],
                "book": b["book"],
                "ev_pct": b["ev_pct"],
                "kelly_pct": b["kelly_pct"],
                "model_source": _model_source(slug),
                "home_team": g["home_name"],
                "away_team": g["away_name"],
            })
    return out


# ============================================================================
# Confidence-based filter and ranking (reverse-engineered from Blacksmith Bets
# / The Syndicate's pick style: mostly 1.60-2.00 decimal, favorites rather
# than longshots, picks the model has real conviction on, not pure EV).
# ============================================================================

MIN_FAIR_PROB      = 0.46   # fair/sharp AND model probability threshold
MIN_DECIMAL        = 1.60   # don't show short-favorite picks (-167+)
MIN_EV_PCT         = 2.0    # require real edge after market blend — 0% EV is not a pick
SOCCER_DRAW_MAX_DEC = 3.70  # soccer draws only when market has them in reach
STRONG_PROB_TIER   = 0.60   # "strong" badge for high-conviction picks
MAX_PICKS          = 30     # cap list length (concise board, not firehose)


def _is_soccer(slug):
    return slug in {"epl", "laliga", "ligamx", "ucl", "europa", "international"}


def _is_draw_pick(pick):
    """Detect a soccer draw bet (pick text contains 'Draw')."""
    return _is_soccer(pick.get("sport_slug", "")) and "draw" in pick.get("pick", "").lower()


import re as _re

_LINE_RE = _re.compile(r"[+-]?\d+(?:\.\d+)?")


def _is_half_or_whole(line):
    """True if line is a half-point (X.5) or whole number (X.0)."""
    if line is None:
        return False
    frac = abs(line - int(line))
    return frac < 1e-9 or abs(frac - 0.5) < 1e-9


def _is_valid_pick_line(p):
    """Reject Pinnacle Asian handicap lines that don't exist on US books
    (0.0 / 0.25 / 0.75 spreads, 1.75 / 2.25 totals, etc.)."""
    market = p.get("market", "") or ""
    text = (p.get("pick", "") or "")
    slug = p.get("sport_slug", "")

    if "Spread" in market or market in ("Run Line", "Puck Line"):
        # Pull a signed number from the pick label ("Team X +1.5", "Team Y -3.5")
        m = _LINE_RE.search(text)
        if not m:
            return False
        try:
            line = float(m.group(0))
        except ValueError:
            return False
        # 0.0 handicap (draw-no-bet style) — no US-book equivalent
        if abs(line) < 0.01:
            return False
        # Baseball run line + hockey puck line are fixed at ±1.5
        if slug in ("mlb", "nhl"):
            return abs(abs(line) - 1.5) < 0.01
        # Everything else: half-points or whole points (skip 0.25-step Asian lines)
        return _is_half_or_whole(line)

    if "Total" in market:
        m = _re.search(r"(?i)(over|under)\s*([\d.]+)", text)
        if not m:
            return True
        try:
            line = float(m.group(2))
        except ValueError:
            return False
        return _is_half_or_whole(line)

    return True  # ML, BTTS, Draw — no line to validate


def _passes_filter(p):
    """Hard filters applied before the consensus EV check."""
    fair = p.get("fair_prob") or 0.0
    dec = p.get("decimal") or 0.0
    if fair < MIN_FAIR_PROB:
        return False
    if dec < MIN_DECIMAL:
        return False
    if _is_draw_pick(p) and dec >= SOCCER_DRAW_MAX_DEC:
        return False
    if not _is_valid_pick_line(p):
        return False
    return True


def _passes_ev_filter(p):
    """Final EV check — only run after consensus has blended with the market.
    A pick with 55% fair at -118 (implied 54%) is +1pp of edge, not a bet.
    Alt-line picks (alt_line_flag) are informational and bypass the EV
    gate; they get surfaced for their fair probability only."""
    if p.get("alt_line_flag"):
        return True
    ev = p.get("ev_pct")
    if ev is None:
        return True
    return ev >= MIN_EV_PCT


def _is_started(pick, now_utc=None):
    """True if the game has already kicked off (don't show those on the board)."""
    st = pick.get("start_time")
    if not st:
        return False
    try:
        from datetime import datetime as _dt, timezone
        iso = st.replace("Z", "+00:00") if st.endswith("Z") else st
        dt = _dt.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = now_utc or _dt.now(timezone.utc)
        return dt < now
    except Exception:
        return False


def _pick_is_on_date(pick, date_str):
    """True if the pick's game starts on the given local date (Eastern).

    Pinnacle and MLB start_times are UTC ISO strings. We convert to Eastern
    local date for comparison — matches how the user picks dates in the UI.
    """
    st = pick.get("start_time")
    if not st:
        return True  # keep picks with no start_time to avoid dropping them silently
    try:
        from datetime import datetime, timezone, timedelta
        # Parse "...Z" or "...+00:00"
        iso = st.replace("Z", "+00:00") if st.endswith("Z") else st
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to Central (UTC-5 CDT / UTC-6 CST, approximate; close enough
        # for date-bucketing — all games on a given CT calendar day).
        try:
            from zoneinfo import ZoneInfo
            ct = dt.astimezone(ZoneInfo("America/Chicago"))
        except Exception:
            ct = dt.astimezone(timezone(timedelta(hours=-5)))
        return ct.date().isoformat() == date_str
    except Exception:
        return True


# ============================================================================
# Monte Carlo consensus layer
# ============================================================================
#
# Problem: picks for sports without a bespoke model (UCL, Europa, Int'l, UFC)
# were using Pinnacle's devigged fair probability straight through, while the
# Monte Carlo page was running a Dixon-Coles (soccer) or Poisson (NHL/MLB) sim
# with team-level data. The two could disagree sharply — Pinnacle might show
# England ML at 55% while the DC sim gives 28%, and the picks board would
# surface the pick anyway. That's the "England ML 55 vs 27.9" bug.
#
# Fix: for every candidate pick, run a quick MC sim (same engine as the MC tab)
# and compute the sim's probability for that pick's exact market. We then:
#   * drop the pick entirely if |sim - fair| exceeds CONSENSUS_MAX_DIVERGENCE
#   * rank by the average of fair + sim ("consensus_prob") instead of fair alone
#
# Sims are cached per (sport, home, away) for the request so a game with three
# bets (ML, Spread, Total) runs the sim once.

CONSENSUS_MAX_DIVERGENCE = 0.08   # drop pick if |sim − fair| > 8pp (tightened from 15)
CONSENSUS_FLAG_DIVERGENCE = 0.05  # flag the pick as "divergent" between 5-8pp
CONSENSUS_SIM_TRIALS     = 1500   # smaller sims for consensus — memory + speed
# Final blend used for the fair prob we actually bet into: 40% our model
# (which itself is the avg of pricing + MC), 60% Pinnacle's devigged line.
# Pinnacle is the sharpest line we have access to; our models beat them
# only in flashes, so a market-weighted blend is honest.
MARKET_WEIGHT            = 0.60
MODEL_WEIGHT             = 0.40

_SIM_CACHE = {}


def _sim_for(sport_slug, home, away, market_total=None):
    """Run (or fetch cached) MC sim for this game. Returns dict or None.

    market_total: Pinnacle's main total line, if available. Only used by
    the NCAAF sim to anchor its scoring projection against the market
    (prevents the Over bias flagged in the review).
    """
    key = (sport_slug, home or "", away or "", market_total)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    sim = None
    try:
        if sport_slug in ("epl", "laliga", "ligamx", "ucl", "europa", "international"):
            import soccer_model
            model_slug = sport_slug if sport_slug in ("epl", "laliga", "ligamx") else "epl"
            sim = soccer_model.simulate_match(home, away, model_slug, n=CONSENSUS_SIM_TRIALS)
        elif sport_slug == "nhl":
            import nhl_model
            sim = nhl_model.simulate_match(home, away, n=CONSENSUS_SIM_TRIALS)
        elif sport_slug == "ncaaf":
            import cfb_model
            sim = cfb_model.simulate_match(home, away, n=CONSENSUS_SIM_TRIALS,
                                           market_total=market_total)
        elif sport_slug == "nfl":
            sim = _nfl_quick_sim(home, away, CONSENSUS_SIM_TRIALS)
        elif sport_slug == "mlb":
            # MLB's full sim needs the game context; we don't carry it on the
            # pick dict. The MLB model is already used for the pick's fair_prob
            # (not Pinnacle devig), so divergence here is low — skip MC check.
            pass
    except Exception:
        sim = None
    _SIM_CACHE[key] = sim
    return sim


def _nfl_quick_sim(home_name, away_name, n):
    """Empirical-margin NFL sim for picks consensus. Mirrors _mc_run_simulation NFL branch."""
    try:
        import nfl_model, nfl_margin_dist, random
        state = nfl_model.get_final_state()
        final_elo = state.get("final_elo", {}) if state else {}
        h = nfl_model.abbr_from_name(home_name) or home_name
        a = nfl_model.abbr_from_name(away_name) or away_name
        h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
        a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
        diff = (h_elo + 65) - a_elo
        edge = diff / 25.0
        proj_margin = edge
        proj_total  = 45.0  # NFL avg total; we draw it independently of margin for V1
        rng = random.Random()
        h_wins = a_wins = 0
        margins = []
        for _ in range(n):
            m = nfl_margin_dist.sample_margin(proj_margin, rng, sport="nfl")
            margins.append(m)
            if m > 0: h_wins += 1
            elif m < 0: a_wins += 1
            else:
                # Rare 0 — coin flip for OT
                if rng.random() < 0.52: h_wins += 1
                else: a_wins += 1
        proj_h = (proj_total + proj_margin) / 2
        proj_a = (proj_total - proj_margin) / 2
        return {"home_win_pct": h_wins / n * 100,
                "away_win_pct": a_wins / n * 100,
                "draw_pct": 0.0,
                "expected_total": proj_total,
                "margins": margins,
                "proj_home_pts": proj_h,
                "proj_away_pts": proj_a}
    except Exception:
        return None


def _sim_prob_for_pick(sim, pick):
    """Return the sim's probability (0-1) for this pick's market, or None if
    we can't map the market to a sim output."""
    if not sim:
        return None
    market = (pick.get("market") or "")
    text = (pick.get("pick") or "")
    home = pick.get("home_team") or ""
    away = pick.get("away_team") or ""
    pick_norm = text.lower()
    home_norm = home.lower()
    away_norm = away.lower()

    # Moneyline (and 1H / 1P ML) — the sim's full-game probs are our best proxy
    # for 1H markets too (1H is correlated enough that gross divergence here
    # usually means something's off with the pick, not with the sim).
    if "ML" in market:
        if "draw" in pick_norm:
            return (sim.get("draw_pct") or 0) / 100.0
        if home_norm and (pick_norm == home_norm or home_norm in pick_norm):
            return (sim.get("home_win_pct") or 0) / 100.0
        if away_norm and (pick_norm == away_norm or away_norm in pick_norm):
            return (sim.get("away_win_pct") or 0) / 100.0
        return None

    # Totals
    if "Total" in market:
        import re as _re
        m = _re.search(r"(?i)(over|under)\s*([\d.]+)", text)
        if not m:
            return None
        side, line = m.group(1).lower(), float(m.group(2))
        totals = sim.get("totals") or {}
        # Soccer sim gives us Over 2.5 / Over 3.5; nearest-bucket fallback
        if abs(line - 2.5) < 0.01 and "over_2_5" in totals:
            return (totals["over_2_5"] / 100.0) if side == "over" else (1 - totals["over_2_5"] / 100.0)
        if abs(line - 3.5) < 0.01 and "over_3_5" in totals:
            return (totals["over_3_5"] / 100.0) if side == "over" else (1 - totals["over_3_5"] / 100.0)
        # Fallback: compare line to expected total — crude but better than nothing
        exp = sim.get("expected_total") or sim.get("mean_total")
        if exp is None:
            return None
        # Normal approximation around the expected total (σ ~ √exp for Poisson sports)
        import math as _m
        sigma = max(1.0, _m.sqrt(abs(exp)) * 1.3)
        from statistics import NormalDist
        z = (line - exp) / sigma
        p_over = 1 - NormalDist().cdf(z)
        return p_over if side == "over" else (1 - p_over)

    # BTTS
    if market == "BTTS":
        if "yes" in pick_norm:
            return (sim.get("btts_yes_pct") or 0) / 100.0
        if "no" in pick_norm:
            return (sim.get("btts_no_pct") or 1 - (sim.get("btts_yes_pct") or 0) / 100.0) / 100.0 if sim.get("btts_no_pct") else (1 - (sim.get("btts_yes_pct") or 0) / 100.0)
        return None

    # Spread / Puck Line / Run Line / 1H / 1P Spread — need per-sim margins
    # (which simulate_match doesn't surface in its summary). Skip the consensus
    # check for spreads; the line-shape filter already removes the worst noise.
    return None


def _apply_consensus(picks):
    """Attach sim probability + consensus_prob to each pick, drop divergent ones,
    and recompute EV and Kelly against the consensus number.

    The consensus prob (default: average of pricing-model fair + Monte Carlo
    sim) is what we actually believe will hit — so it's what gets compared to
    the book price for edge. ev_pct and kelly_pct are rewritten here to
    reflect that; the original pricing-model number stays in `pricing_prob`
    for the bulletin.

    If model_analytics.get_sport_weights() returns a non-default blend for a
    sport (the daily deep analysis found a weight shift that beats 50/50 by
    at least 5 Brier basis points over the last 90 days), that per-sport
    weight is used instead of 50/50. Keeps the models self-tuning without a
    code change on each shift.
    """
    _SIM_CACHE.clear()
    try:
        import model_analytics
        sport_weights = model_analytics.get_sport_weights()
        sport_min_prob = model_analytics.get_sport_min_prob()
        market_blacklist = model_analytics.get_market_blacklist()
    except Exception:
        sport_weights, sport_min_prob, market_blacklist = {}, {}, set()

    kept = []
    for p in picks:
        # Drop market types the analyzer flagged as deeply unprofitable.
        if p.get("market") in market_blacklist:
            continue
        # For NCAAF totals, extract the line from the pick text and pass to
        # the sim so it can anchor its scoring projection against the market
        # (fixes the Over bias where every total was 62-64%).
        market_total = None
        if p.get("sport_slug") == "ncaaf" and "Total" in (p.get("market") or ""):
            m = _re.search(r"(?i)(?:over|under)\s+([\d.]+)", p.get("pick") or "")
            if m:
                try: market_total = float(m.group(1))
                except ValueError: pass
        sim = _sim_for(p["sport_slug"], p.get("home_team"), p.get("away_team"),
                       market_total=market_total)
        sim_prob = _sim_prob_for_pick(sim, p)
        fair = p.get("fair_prob") or 0.0

        if sim_prob is None:
            # No MC coverage for this market (e.g. MLB bets path, which has
            # its own model). Still blend with market so MLB picks go through
            # the same 40/60 EV discipline as everything else.
            p["mc_prob"] = None
            p["divergence"] = None
            p["pricing_prob"] = fair
            pinnacle = p.get("pinnacle_prob")
            if pinnacle is not None and pinnacle > 0:
                blended = MODEL_WEIGHT * fair + MARKET_WEIGHT * pinnacle
                p["model_consensus"] = fair
                p["consensus_prob"] = blended
                p["fair_prob"] = blended
                dec = p.get("decimal") or 0.0
                if dec > 1.0:
                    push = p.get("push_prob", 0) or 0
                    p["ev_pct"] = (blended * dec - (1 - push)) * 100
                    b = dec - 1.0
                    q = 1.0 - blended
                    kelly = ((blended * b - q) / b) * 0.25 * 100.0 if b > 0 else 0.0
                    p["kelly_pct"] = max(0.0, kelly)
            else:
                p["consensus_prob"] = fair
            kept.append(p)
            continue

        p["mc_prob"] = sim_prob
        p["divergence"] = abs(sim_prob - fair)

        # Model consensus first: pricing + MC blended per-sport by analyzer.
        pw, mw = sport_weights.get(p.get("sport"), (0.5, 0.5))
        model_consensus = pw * fair + mw * sim_prob
        p["pricing_weight"] = pw
        p["mc_weight"] = mw
        p["pricing_prob"] = fair
        p["model_consensus"] = model_consensus

        # Final fair = 40% our model's view + 60% Pinnacle's devigged market
        # price. The market is sharp; our models earn their weight only
        # where they consistently beat the no-vig line. This also means EV
        # genuinely reflects a market disagreement, not model self-confidence.
        pinnacle = p.get("pinnacle_prob")
        if pinnacle is not None and pinnacle > 0:
            consensus = MODEL_WEIGHT * model_consensus + MARKET_WEIGHT * pinnacle
            p["pinnacle_prob"] = pinnacle
        else:
            consensus = model_consensus
        p["consensus_prob"] = consensus
        # fair_prob now = consensus so every downstream reader uses it
        p["fair_prob"] = consensus

        # Flag picks whose pricing/MC disagree 5-8pp — included but marked
        p["divergent_flag"] = (CONSENSUS_FLAG_DIVERGENCE <= p["divergence"] < CONSENSUS_MAX_DIVERGENCE)

        # Per-sport probability floor from the analyzer (default from MIN_FAIR_PROB)
        floor = sport_min_prob.get(p.get("sport"), MIN_FAIR_PROB)
        if consensus < floor:
            continue

        # Recompute EV and quarter-Kelly against consensus vs the book price.
        dec = p.get("decimal") or 0.0
        if dec > 1.0:
            push = p.get("push_prob", 0) or 0
            p["ev_pct"] = (consensus * dec - (1 - push)) * 100
            b = dec - 1.0
            q = 1.0 - consensus
            kelly = ((consensus * b - q) / b) * 0.25 * 100.0 if b > 0 else 0.0
            p["kelly_pct"] = max(0.0, kelly)

        if p["divergence"] > CONSENSUS_MAX_DIVERGENCE:
            continue  # the two signals disagree too much — don't recommend
        kept.append(p)
    return kept


def _expand_alt_line_picks(picks):
    """For each NFL/NCAAF spread pick, add alt-line variants at ±3 and ±6
    points off the main line. Fair prob is read off the empirical margin
    distribution; EV is scored against the base pick's book price adjusted
    for a per-point juice shift (approximate DK alt-line pricing).

    Each generated alt pick carries alt_line_flag=True so the UI badges it
    as 'ALT' and reminds the user to confirm the actual DK price before
    locking it in.
    """
    try:
        import nfl_margin_dist
    except Exception:
        return picks

    out = list(picks)
    for p in picks:
        slug = p.get("sport_slug")
        if slug not in ("nfl", "ncaaf"):
            continue
        if "Spread" not in (p.get("market") or ""):
            continue
        # Parse the current line and side from the pick text
        import re as _re
        m = _re.search(r"([+-]\d+(?:\.\d+)?)", p.get("pick") or "")
        if not m:
            continue
        try:
            base_line = float(m.group(1))
        except ValueError:
            continue
        base_dec = p.get("decimal") or 0.0
        if base_dec <= 1.0:
            continue
        side_name = p.get("home_team") if (p.get("home_team") or "").lower() in (p.get("pick") or "").lower() else p.get("away_team")
        is_home = side_name == p.get("home_team")

        # Projected margin: for the FAVORITE's side, main line implied prob is
        # roughly fair_prob; we back out projected margin by solving
        # p_margin_ge(x, −base_line) = pricing_prob for the home side. Easier:
        # use the pricing_prob as a starting point and solve numerically.
        pricing = p.get("pricing_prob") or p.get("consensus_prob") or 0.5
        proj_margin_home = _solve_margin_for_prob(pricing, base_line, is_home, slug)
        if proj_margin_home is None:
            continue

        for alt_shift in (-3.0, -6.0, 3.0, 6.0):
            alt_line = base_line + alt_shift
            if alt_line == base_line:
                continue
            # Fair prob at this alt line:
            #   home covers -alt_line  ⇔  margin > -alt_line
            #   away covers +alt_line  ⇔  margin <  alt_line
            # (Note: alt_line here is the SIDE's spread — negative = favored.)
            if is_home:
                fair = nfl_margin_dist.p_margin_ge(proj_margin_home, -alt_line + 0.001, sport=slug)
            else:
                fair = nfl_margin_dist.p_margin_le(proj_margin_home, alt_line - 0.001, sport=slug)

            if fair < 0.46 or fair > 0.90:
                continue  # outside useful band

            # Alt lines are shown INFORMATIONALLY: we can't price them
            # accurately without live DK alt-line odds, so we surface the
            # empirical fair prob and leave EV blank. User looks up the
            # actual DK price and decides.
            if fair < 0.60:  # only show meaningfully-confident alts
                continue

            # Direction label from the bettor's perspective: +X means the
            # bettor gets X MORE points of cushion vs the main line.
            if is_home:
                bettor_delta = base_line - alt_line  # home moving to shorter spread = easier
            else:
                bettor_delta = alt_line - base_line  # away moving to bigger +line = easier
            delta_tag = f"+{bettor_delta:.1f} pts easier" if bettor_delta > 0 else f"{bettor_delta:.1f} pts harder"
            new_line_label = f"{side_name} {'+' if alt_line >= 0 else ''}{alt_line}"
            new_pick = dict(p)
            new_pick.update({
                "id":         f"{p.get('id','')}_alt{alt_shift:+.0f}",
                "pick":       f"{new_line_label} (ALT · {delta_tag} vs main)",
                "fair_prob":  fair,
                "consensus_prob": fair,
                "pricing_prob":  fair,
                "mc_prob":    fair,
                "decimal":    None,     # no model price for alt
                "american":   None,
                "ev_pct":     None,     # bypass EV filter — informational
                "kelly_pct":  None,
                "alt_line_flag": True,
                "alt_shift":  alt_shift,
                "alt_delta":  bettor_delta,
                "base_line":  base_line,
                "book":       "verify at DK/FD",
                "strong":     False,
                "bulletin":   (f"Alt-line candidate from the empirical NFL/NCAAF "
                               f"margin distribution. Fair prob {fair*100:.1f}% "
                               f"at {new_line_label} ({delta_tag} vs main line "
                               f"{side_name} {'+' if base_line>=0 else ''}{base_line}). "
                               f"Not auto-priced — compare live DK alt odds to "
                               f"decide if the price beats the fair."),
            })
            out.append(new_pick)
    return out


def _solve_margin_for_prob(target_prob, base_line, is_home, slug):
    """Numerically back out the projected margin that produces target_prob at base_line.
    Returns the home-side projected margin (positive = home favored)."""
    try:
        import nfl_margin_dist
    except Exception:
        return None
    # Grid search: projected home margins from -21 to +21
    best = None
    best_err = 1e9
    for pm in range(-21, 22):
        if is_home:
            fair = nfl_margin_dist.p_margin_ge(pm, -base_line + 0.001, sport=slug)
        else:
            fair = nfl_margin_dist.p_margin_le(pm, base_line - 0.001, sport=slug)
        err = abs(fair - target_prob)
        if err < best_err:
            best_err = err
            best = pm
    return best


def _dec_to_american(dec):
    if dec is None or dec <= 1.0:
        return "—"
    if dec >= 2.0:
        return f"+{int(round((dec - 1) * 100))}"
    return str(int(round(-100 / (dec - 1))))


def collect_picks(date_str, today_only=True):
    """Return filtered picks ranked by model probability (not EV).

    The reference picks (Blacksmith Bets, The Syndicate) target confidence
    over value — they'll take a -150 (barely over 60% implied) favorite in a
    parlay if the probability is right, rather than hunt +400 longshots with
    nominally positive EV. We mirror that: filter on probability thresholds
    first, then rank by model probability descending.

    today_only: if True (default), drop picks whose game isn't on date_str in
    Eastern local time — the picks board is a daily board, not a "whatever
    Pinnacle has posted" board.
    """
    all_picks = []
    all_picks.extend(_mlb_bets(date_str))
    for sport in sports.SPORTS:
        if sport.get("dedicated"):
            continue
        all_picks.extend(_sport_bets(sport))

    # Filter to today's games only (Central local date), already-started filter
    if today_only:
        all_picks = [p for p in all_picks if _pick_is_on_date(p, date_str)]
    all_picks = [p for p in all_picks if not _is_started(p)]

    # Filter to confidence picks (prob + decimal + draw rule + line-shape)
    filtered = [p for p in all_picks if _passes_filter(p)]

    # Consensus pass: run the sport's MC sim, blend 40% model + 60% market,
    # drop picks where pricing vs MC diverge by more than CONSENSUS_MAX_DIVERGENCE.
    # Rewrites ev_pct against the blended consensus.
    filtered = _apply_consensus(filtered)

    # Alt-line expansion: for NFL/NCAAF spreads, add alt-line variants at
    # ±3 and ±6 points off the main line, priced from the empirical margin
    # distribution. Each alt pick is marked with alt_line_flag so the UI
    # can show "ALT" and remind the user to compare to the live DK price.
    filtered = _expand_alt_line_picks(filtered)

    # Final EV gate — a pick must beat the market by at least MIN_EV_PCT after
    # the 40/60 blend. Zero or negative EV picks don't belong on the board.
    filtered = [p for p in filtered if _passes_ev_filter(p)]

    # ONE PICK PER GAME for the MAIN board — picking both ML and Over on
    # the same game is double-dipping. Alt-line picks are exempt from this
    # rule because they're informational companions, not primary bets.
    main_picks = [p for p in filtered if not p.get("alt_line_flag")]
    alt_picks  = [p for p in filtered if p.get("alt_line_flag")]
    best_per_game = {}
    for p in main_picks:
        key = (p["sport_slug"], p.get("game", ""))
        current = best_per_game.get(key)
        if not current or (p.get("ev_pct") or 0) > (current.get("ev_pct") or 0):
            best_per_game[key] = p
    deduped = list(best_per_game.values()) + alt_picks

    # Rank: main picks first by EV desc; alt-line picks appended at the end
    # ordered by fair prob desc. Alts carry no EV so they shouldn't crowd
    # out real +EV plays, but they're useful once you've locked a side.
    main_sorted = [p for p in deduped if not p.get("alt_line_flag")]
    alt_sorted  = [p for p in deduped if p.get("alt_line_flag")]
    main_sorted.sort(key=lambda x: (-(x.get("ev_pct") or 0.0),
                                     -(x.get("consensus_prob") or 0.0)))
    alt_sorted.sort(key=lambda x: -(x.get("fair_prob") or 0.0))
    deduped = main_sorted[:MAX_PICKS] + alt_sorted[:12]

    # Tag strong tier: both ≥ 60% consensus AND ≥ 4% EV to earn the star.
    for p in deduped:
        base = p.get("consensus_prob") or p.get("fair_prob") or 0.0
        p["strong"] = (base >= STRONG_PROB_TIER) and ((p.get("ev_pct") or 0) >= 4.0)
        p["bulletin"] = generate_bulletin(p)
    return deduped


def generate_bulletin(pick):
    """Confidence-first 1-2 sentence rationale.

    Leads with model probability (what MC would simulate), then supporting
    EV + Kelly, then sport-specific color line about the model source.
    """
    # pricing_prob is the raw sport-model / Pinnacle-devig number; mc_prob is
    # the Monte Carlo sim; cons_pct is their average and the one we actually
    # bet into. For picks without MC coverage, pricing_prob is None and we
    # fall back to showing only the one number.
    pricing_pct = (pick.get("pricing_prob") or 0) * 100 if pick.get("pricing_prob") is not None else None
    mc_pct      = (pick.get("mc_prob") or 0) * 100 if pick.get("mc_prob") is not None else None
    cons_pct    = (pick.get("consensus_prob") or pick.get("fair_prob") or 0) * 100
    dec = pick.get("decimal") or 0
    market_implied = (1.0 / dec) * 100 if dec else 0

    conviction = "high-conviction" if cons_pct >= STRONG_PROB_TIER * 100 else "selective"
    ev = pick.get("ev_pct") or 0
    bits = []
    if mc_pct is not None and pricing_pct is not None:
        bits.append(
            f"Consensus of pricing model ({pricing_pct:.0f}%) and Monte Carlo "
            f"({mc_pct:.0f}%) lands at {cons_pct:.0f}% for {pick['pick']} "
            f"({conviction}); market price {pick.get('american','?')} "
            f"implies {market_implied:.0f}% — we treat the {cons_pct:.0f}% as "
            f"our fair when sizing EV."
        )
    else:
        bits.append(
            f"Model simulates {pick['pick']} to hit {cons_pct:.0f}% of the time "
            f"({conviction} pick) vs market's implied {market_implied:.0f}% at "
            f"{pick.get('american','?')}"
        )
    # Only call out EV when it's positive; otherwise just mention the book
    if ev > 0.1:
        bits.append(
            f"EV +{ev:.1f}% at {pick['book']}, quarter-Kelly "
            f"stake {pick['kelly_pct']:.1f}% of bankroll"
        )
    else:
        bits.append(
            f"Priced at {pick['book']} with no model-vs-market edge; "
            f"included on probability conviction"
        )
    base = ". ".join(bits) + "."
    slug = pick["sport_slug"]
    model_labels = {
        "mlb":    "MLB model: Elo + starting-pitcher ERA/WHIP + Poisson. "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "nfl":    "NFL model: team Elo blended with per-QB rating (12-season fit). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "ncaaf":  "NCAAF model: normal-distribution scoring from CFBD-fit Elo. "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "nhl":    "NHL model: Poisson goals + 5-season Elo fit (NHL.com season stats). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "epl":    "EPL model: 3-way Elo + Dixon-Coles (12-season fit). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "laliga": "La Liga model: 3-way Elo + Dixon-Coles (12-season fit). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "ligamx": "Liga MX model: 3-way Elo + Dixon-Coles (12-season fit). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "ucl":    "UCL: Dixon-Coles sim with EPL-prior goal rates (fallback). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "europa": "Europa: Dixon-Coles sim with EPL-prior goal rates (fallback). "
                  "Blended 40% model + 60% Pinnacle devig before EV.",
        "international": "International: Dixon-Coles sim with league-prior rates "
                         "(fallback). Blended 40% model + 60% Pinnacle devig before EV.",
        "ufc":    "UFC: Pinnacle devig only. Dedicated fighter model is a known gap.",
    }
    base += " " + model_labels.get(slug, "Model: Pinnacle devig.")
    if pick.get("divergent_flag"):
        base += " ⚠ Pricing model and Monte Carlo diverge 5-8pp — size down."
    return base
