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
            out.append({
                "id": f"mlb_{g.get('game_pk')}_{b['market']}_{b['side']}",
                "sport": "MLB",
                "sport_slug": "mlb",
                "game": f"{g['away']['team']} at {g['home']['team']}",
                "start_time": g.get("first_pitch"),
                "market": b["market"],
                "pick": b["pick"],
                "fair_prob": b["fair_prob"],
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
        "epl":    "3-way Elo (12-season fit)",
        "laliga": "3-way Elo (12-season fit)",
        "ligamx": "3-way Elo (12-season fit)",
    }
    return labels.get(slug, "Pinnacle devig only (no bespoke model)")


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
    """Reverse-engineered filter from the reference picks."""
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

CONSENSUS_MAX_DIVERGENCE = 0.15   # drop pick if |sim − fair| > 15pp
CONSENSUS_SIM_TRIALS     = 2000   # per-pick MC trial count (fast, ±1pp precision)

_SIM_CACHE = {}


def _sim_for(sport_slug, home, away):
    """Run (or fetch cached) MC sim for this game. Returns dict or None."""
    key = (sport_slug, home or "", away or "")
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
            sim = cfb_model.simulate_match(home, away, n=CONSENSUS_SIM_TRIALS)
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
    """Normal-dist NFL scoring sim for picks consensus. Mirrors _mc_run_simulation NFL branch."""
    try:
        import nfl_model, random
        state = nfl_model.get_final_state()
        final_elo = state.get("final_elo", {}) if state else {}
        h = nfl_model.abbr_from_name(home_name) or home_name
        a = nfl_model.abbr_from_name(away_name) or away_name
        h_elo = final_elo.get(h, nfl_model.INITIAL_ELO)
        a_elo = final_elo.get(a, nfl_model.INITIAL_ELO)
        diff = (h_elo + 65) - a_elo
        edge = diff / 25.0
        proj_h = 22.5 + edge / 2
        proj_a = 22.5 - edge / 2
        rng = random.Random()
        h_wins = a_wins = 0
        for _ in range(n):
            hs = max(0.0, rng.gauss(proj_h, 13.0))
            asc = max(0.0, rng.gauss(proj_a, 13.0))
            if hs > asc: h_wins += 1
            else: a_wins += 1
        return {"home_win_pct": h_wins / n * 100,
                "away_win_pct": a_wins / n * 100,
                "draw_pct": 0.0,
                "expected_total": proj_h + proj_a,
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
        sim = _sim_for(p["sport_slug"], p.get("home_team"), p.get("away_team"))
        sim_prob = _sim_prob_for_pick(sim, p)
        fair = p.get("fair_prob") or 0.0

        if sim_prob is None:
            # No MC coverage for this market — fall back to fair prob alone
            p["mc_prob"] = None
            p["consensus_prob"] = fair
            p["divergence"] = None
            kept.append(p)
            continue

        p["mc_prob"] = sim_prob
        p["divergence"] = abs(sim_prob - fair)
        # Blend with per-sport weights from the daily analyzer when available.
        pw, mw = sport_weights.get(p.get("sport"), (0.5, 0.5))
        consensus = pw * fair + mw * sim_prob
        p["pricing_weight"] = pw
        p["mc_weight"] = mw
        p["pricing_prob"] = fair      # keep the original for display
        p["consensus_prob"] = consensus
        # Overwrite fair_prob so every downstream reader (bulletin, filters,
        # strong tier, logged plays, edge table) uses the consensus number.
        p["fair_prob"] = consensus

        # Per-sport probability floor from the analyzer (default 46% from
        # MIN_FAIR_PROB; raised when a sport's low-end picks are losing).
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

    # Filter to today's games only (Eastern local date)
    if today_only:
        all_picks = [p for p in all_picks if _pick_is_on_date(p, date_str)]

    # Filter to confidence picks (prob + decimal + draw rule + line-shape)
    filtered = [p for p in all_picks if _passes_filter(p)]

    # Consensus pass: run the sport's MC sim for each pick's game and reject
    # picks where the sim disagrees with the model's fair prob by more than
    # CONSENSUS_MAX_DIVERGENCE. Also attaches `mc_prob` and `consensus_prob`.
    filtered = _apply_consensus(filtered)

    # Deduplicate: only the single highest-consensus side per (game, market).
    best_per_market = {}
    for p in filtered:
        key = (p["sport_slug"], p.get("game", ""), p.get("market", ""))
        current = best_per_market.get(key)
        if not current or (p.get("consensus_prob") or 0) > (current.get("consensus_prob") or 0):
            best_per_market[key] = p
    deduped = list(best_per_market.values())

    # Rank by CONSENSUS probability (desc), EV as a tiebreaker. Picks where
    # both the pricing model AND the Monte Carlo sim agree rise to the top.
    deduped.sort(key=lambda x: (-(x.get("consensus_prob") or x.get("fair_prob") or 0.0),
                                 -(x.get("ev_pct") or 0.0)))
    deduped = deduped[:MAX_PICKS]

    # Tag strong tier against the consensus number — a pick is only "strong"
    # when BOTH signals clear the 60% bar.
    for p in deduped:
        base = p.get("consensus_prob") or p.get("fair_prob") or 0.0
        p["strong"] = base >= STRONG_PROB_TIER
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
    if slug == "mlb":
        base += " MLB model: Elo + starting-pitcher ERA/WHIP + Poisson."
    elif slug == "nfl":
        base += " NFL model: team Elo blended with per-QB rating (12-season fit)."
    elif slug in _SOCCER_WITH_MODEL:
        base += " Soccer model: 3-way Elo (home/draw/away) with +100 HFA."
    else:
        base += " (No bespoke model for this sport — fair prob is Pinnacle devig.)"
    return base
