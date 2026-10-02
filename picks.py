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
            if b.get("ev_pct", 0) <= 0:
                continue
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
    try:
        import nfl_model
        state = nfl_model.get_or_run_multi_season_backtest()
        final_elo = state.get("final_elo", {}) if state else {}
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
        state = soccer_model.get_or_run_backtest(slug)
        final_elo = state.get("final_elo", {}) if state else {}
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
            if b.get("ev_pct", 0) <= 0:
                continue
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
MIN_DECIMAL        = 1.70   # don't show short-favorite picks (-150+)
SOCCER_DRAW_MAX_DEC = 3.70  # soccer draws only when market has them in reach
STRONG_PROB_TIER   = 0.60   # "strong" badge for high-conviction picks


def _is_soccer(slug):
    return slug in {"epl", "laliga", "ligamx", "ucl", "europa", "international"}


def _is_draw_pick(pick):
    """Detect a soccer draw bet (pick text contains 'Draw')."""
    return _is_soccer(pick.get("sport_slug", "")) and "draw" in pick.get("pick", "").lower()


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
    return True


def collect_picks(date_str):
    """Return filtered picks ranked by model probability (not EV).

    The reference picks (Blacksmith Bets, The Syndicate) target confidence
    over value — they'll take a -150 (barely over 60% implied) favorite in a
    parlay if the probability is right, rather than hunt +400 longshots with
    nominally positive EV. We mirror that: filter on probability thresholds
    first, then rank by model probability descending.
    """
    all_picks = []
    all_picks.extend(_mlb_bets(date_str))
    for sport in sports.SPORTS:
        if sport.get("dedicated"):
            continue
        all_picks.extend(_sport_bets(sport))

    # Filter to confidence picks
    filtered = [p for p in all_picks if _passes_filter(p)]

    # Rank by model probability (desc), with EV as a tiebreaker
    filtered.sort(key=lambda x: (-(x.get("fair_prob") or 0.0),
                                  -(x.get("ev_pct") or 0.0)))

    # Tag strong tier (MC-style conviction >= 60%)
    for p in filtered:
        p["strong"] = (p.get("fair_prob") or 0.0) >= STRONG_PROB_TIER
        p["bulletin"] = generate_bulletin(p)
    return filtered


def generate_bulletin(pick):
    """Confidence-first 1-2 sentence rationale.

    Leads with model probability (what MC would simulate), then supporting
    EV + Kelly, then sport-specific color line about the model source.
    """
    fair_pct = (pick.get("fair_prob") or 0) * 100
    dec = pick.get("decimal") or 0
    market_implied = (1.0 / dec) * 100 if dec else 0

    conviction = "high-conviction" if fair_pct >= STRONG_PROB_TIER * 100 else "selective"
    bits = []
    bits.append(
        f"Model simulates {pick['pick']} to hit {fair_pct:.0f}% of the time "
        f"({conviction} pick) vs market's implied {market_implied:.0f}% at "
        f"{pick.get('american','?')}"
    )
    bits.append(
        f"EV +{pick['ev_pct']:.1f}% at {pick['book']}, quarter-Kelly "
        f"stake {pick['kelly_pct']:.1f}% of bankroll"
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
