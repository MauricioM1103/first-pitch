#!/usr/bin/env python3
"""Daily deep analysis of logged picks.

Reads every graded pick we've snapshotted, computes:

  * per-sport + per-model (pricing / monte_carlo / consensus) hit rate,
    ROI, Brier score, and sample-size-weighted log loss
  * calibration bins: for picks we said would hit at 50-55%, 55-60%, 60-65%,
    etc. — what did they actually hit at? Diverges from the diagonal =
    miscalibrated model
  * which blend of (pricing, MC) scored best per sport and what weight
    shift that implies

Writes `logs/analysis_YYYY-MM-DD.json` once per day. picks.collect_picks
reads the latest analysis file for per-sport consensus weights so an
overperforming signal starts carrying more of the blend tomorrow —
conservatively (max shift ±0.15 from the default 0.5/0.5) and only when
the sample is big enough (>= 30 settled picks per sport).

Also exposes an insights list the /logged dashboard renders at the top:
natural-language flags like "NHL consensus is 5pp over-confident on 60%+
picks (84 settled)".
"""
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta

import plays_log
import log_persist

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MIN_SAMPLES_FOR_RECO     = 200  # don't trust a weight shift under this n
MIN_SAMPLES_FOR_CALIB    = 25   # per-bin threshold for calibration display
MIN_SAMPLES_FOR_FILTER   = 100  # min settled picks before applying a filter tweak
MAX_WEIGHT_SHIFT         = 0.15 # cap any single-sport blend adjustment
DEFAULT_WEIGHT           = 0.5  # pricing_prob = 0.5, mc_prob = 0.5 by default
LOOKBACK_DAYS            = 90


def analysis_file(date_str):
    return os.path.join(LOG_DIR, f"analysis_{date_str}.json")


def _load_recent_picks(lookback_days=LOOKBACK_DAYS, today=None):
    """Flatten all graded picks in the lookback window."""
    today = today or date.today()
    cutoff = today - timedelta(days=lookback_days)
    graded_all = []
    for ds in plays_log.all_logged_dates(limit=lookback_days + 10):
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            continue
        g = plays_log.read_graded(ds) or plays_log.read_picks(ds)
        if not g:
            continue
        for p in g.get("picks", []):
            graded_all.append(p)
    return graded_all


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _brier(prob, won):
    return (prob - (1.0 if won else 0.0)) ** 2


def _log_loss(prob, won):
    p = max(1e-6, min(1 - 1e-6, prob))
    return -math.log(p) if won else -math.log(1 - p)


def _settled(p):
    return p.get("result") in ("W", "L")


def _was_win(p):
    return p.get("result") == "W"


# ---------------------------------------------------------------------------
# Per-sport / per-model metrics
# ---------------------------------------------------------------------------

def model_metrics_per_sport(picks):
    """For each (sport, model) pair compute n, wins, win_pct, brier, log_loss,
    and units ROI. "model" is one of pricing / monte_carlo / consensus.
    """
    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "brier": 0.0,
                                   "ll": 0.0, "units": 0.0})
    for p in picks:
        if not _settled(p):
            continue
        won = _was_win(p)
        sport = p.get("sport") or "?"
        decimal = p.get("decimal") or 0
        profit = p.get("profit_u")
        if profit is None:
            profit = (decimal - 1.0) if won and decimal > 1 else (-1.0 if not won else 0.0)
        for model_key, prob_field in (
            ("pricing",   "pricing_prob"),
            ("monte_carlo", "mc_prob"),
            ("consensus", "consensus_prob"),
        ):
            prob = p.get(prob_field)
            if prob is None:
                continue
            b = buckets[(sport, model_key)]
            b["n"]     += 1
            b["wins"]  += 1 if won else 0
            b["brier"] += _brier(prob, won)
            b["ll"]    += _log_loss(prob, won)
            # ROI tracked only on consensus (the one we actually bet)
            if model_key == "consensus":
                b["units"] += profit
    rows = []
    for (sport, model), b in buckets.items():
        if not b["n"]:
            continue
        rows.append({
            "sport":     sport,
            "model":     model,
            "n":         b["n"],
            "wins":      b["wins"],
            "win_pct":   b["wins"] / b["n"] * 100,
            "brier":     b["brier"] / b["n"],
            "log_loss":  b["ll"]    / b["n"],
            "units":     b["units"],
            "roi_pct":   (b["units"] / b["n"] * 100) if b["n"] else 0.0,
        })
    rows.sort(key=lambda r: (r["sport"], r["model"]))
    return rows


def calibration_bins(picks, prob_field="consensus_prob"):
    """Group settled picks into 5-pt probability bins and compare expected
    vs actual hit rate. Shows whether the model is honest at each tier."""
    bins = defaultdict(lambda: {"n": 0, "wins": 0, "sum_prob": 0.0})
    for p in picks:
        if not _settled(p):
            continue
        prob = p.get(prob_field)
        if prob is None:
            continue
        # Bins: 0.45–0.50, 0.50–0.55, 0.55–0.60, 0.60–0.65, 0.65–0.70, 0.70+
        lo = min(0.70, max(0.45, round(prob * 20) / 20))
        bins[lo]["n"] += 1
        bins[lo]["wins"] += 1 if _was_win(p) else 0
        bins[lo]["sum_prob"] += prob
    out = []
    for lo in sorted(bins):
        b = bins[lo]
        out.append({
            "bin_lo":   lo,
            "bin_hi":   lo + 0.05,
            "n":        b["n"],
            "expected": b["sum_prob"] / b["n"] * 100,
            "actual":   b["wins"] / b["n"] * 100,
            "gap_pp":   (b["wins"] / b["n"] * 100) - (b["sum_prob"] / b["n"] * 100),
        })
    return out


# ---------------------------------------------------------------------------
# Recommended consensus weights per sport
# ---------------------------------------------------------------------------

def recommended_weights(picks):
    """Find the (pricing, mc) blend weight per sport that would have minimized
    Brier score over the lookback window. Capped at ±MAX_WEIGHT_SHIFT from
    the default 50/50 so we never go wild on a small sample.

    Returns {sport_name: {"pricing_w": w, "mc_w": 1-w, "n": n, "brier_at_rec": b,
             "brier_at_default": d, "improvement_bp": …}}.
    """
    by_sport = defaultdict(list)
    for p in picks:
        if not _settled(p):
            continue
        if p.get("pricing_prob") is None or p.get("mc_prob") is None:
            continue
        by_sport[p.get("sport") or "?"].append(p)

    out = {}
    weight_candidates = [DEFAULT_WEIGHT - MAX_WEIGHT_SHIFT + 0.05 * i
                         for i in range(int(MAX_WEIGHT_SHIFT * 2 / 0.05) + 1)]
    weight_candidates = [round(w, 2) for w in weight_candidates]
    for sport, ps in by_sport.items():
        if len(ps) < MIN_SAMPLES_FOR_RECO:
            continue
        best_w = DEFAULT_WEIGHT
        best_brier = float("inf")
        for w in weight_candidates:
            br = 0.0
            for p in ps:
                blended = w * p["pricing_prob"] + (1 - w) * p["mc_prob"]
                br += _brier(blended, _was_win(p))
            br /= len(ps)
            if br < best_brier:
                best_brier = br
                best_w = w
        # Also measure default blend brier for comparison
        def_br = 0.0
        for p in ps:
            blended = DEFAULT_WEIGHT * p["pricing_prob"] + (1 - DEFAULT_WEIGHT) * p["mc_prob"]
            def_br += _brier(blended, _was_win(p))
        def_br /= len(ps)
        out[sport] = {
            "pricing_w":        best_w,
            "mc_w":             round(1 - best_w, 2),
            "n":                len(ps),
            "brier_at_rec":     round(best_brier, 4),
            "brier_at_default": round(def_br, 4),
            "improvement_bp":   round((def_br - best_brier) * 10000, 1),
        }
    return out


# ---------------------------------------------------------------------------
# Picks-pipeline feedback levers
# ---------------------------------------------------------------------------

def recommended_min_fair_prob(picks):
    """Per-sport: find the consensus_prob floor that would have maximized ROI
    over the lookback window. If a sport's low-end picks (46-54%) have been
    losing but its 55%+ picks are winning, raise its floor. Only moves the
    floor in 1% steps up from 0.46; capped at 0.60 so we don't nuke a sport.

    Returns {sport: {"min_fair_prob": float, "roi_at_rec": float,
                     "roi_at_default": float, "n": int}} with entries only
    where the new floor actually improves ROI.
    """
    by_sport = defaultdict(list)
    for p in picks:
        if not _settled(p): continue
        if p.get("consensus_prob") is None: continue
        by_sport[p.get("sport") or "?"].append(p)

    DEFAULT_FLOOR = 0.46
    out = {}
    for sport, ps in by_sport.items():
        if len(ps) < MIN_SAMPLES_FOR_FILTER:
            continue
        best_floor = DEFAULT_FLOOR
        best_roi = _roi(ps, DEFAULT_FLOOR)
        default_roi = best_roi
        for floor in [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            kept = [p for p in ps if (p.get("consensus_prob") or 0) >= floor]
            if len(kept) < 10:
                continue
            r = _roi(ps, floor)
            if r > best_roi + 0.5:  # need meaningful improvement
                best_roi = r
                best_floor = floor
        if best_floor > DEFAULT_FLOOR:
            out[sport] = {
                "min_fair_prob":   best_floor,
                "roi_at_rec":      round(best_roi, 2),
                "roi_at_default":  round(default_roi, 2),
                "improvement_pp":  round(best_roi - default_roi, 2),
                "n":               len(ps),
            }
    return out


def _roi(picks, floor):
    kept = [p for p in picks if (p.get("consensus_prob") or 0) >= floor]
    if not kept:
        return 0.0
    units = sum((p.get("profit_u") or 0) for p in kept)
    return units / len(kept) * 100


def market_blacklist(picks):
    """Per-market: if a market type is deep underwater (ROI ≤ -8% over ≥ N
    settled picks), flag it so picks.collect_picks can drop it tomorrow.

    Returns {market_name: {"roi_pct": float, "n": int, "wins": int, "losses": int}}.
    """
    by_market = defaultdict(lambda: {"n": 0, "wins": 0, "losses": 0, "units": 0.0})
    for p in picks:
        if not _settled(p): continue
        key = p.get("market") or "?"
        b = by_market[key]
        b["n"] += 1
        if _was_win(p): b["wins"] += 1
        else: b["losses"] += 1
        b["units"] += p.get("profit_u") or 0
    out = {}
    for market, b in by_market.items():
        if b["n"] < MIN_SAMPLES_FOR_FILTER:
            continue
        roi = (b["units"] / b["n"]) * 100
        if roi <= -8:  # deeply unprofitable
            out[market] = {
                "roi_pct": round(roi, 2),
                "n":       b["n"],
                "wins":    b["wins"],
                "losses":  b["losses"],
            }
    return out


# ---------------------------------------------------------------------------
# Natural-language insights
# ---------------------------------------------------------------------------

def generate_insights(metrics_rows, calib, weights, floors=None, blacklist=None):
    """Return a list of short flags for the /logged dashboard."""
    insights = []
    floors = floors or {}
    blacklist = blacklist or {}

    # Sport × model: who's actually winning the consensus bets
    cons_rows = [r for r in metrics_rows if r["model"] == "consensus"]
    cons_rows.sort(key=lambda r: -r["roi_pct"])
    hot = [r for r in cons_rows if r["n"] >= 15 and r["roi_pct"] >= 5]
    cold = [r for r in cons_rows if r["n"] >= 15 and r["roi_pct"] <= -5]
    for r in hot[:3]:
        insights.append({
            "tone": "good",
            "text": f"{r['sport']}: consensus is +{r['roi_pct']:.1f}% ROI "
                    f"over {r['n']} picks ({r['wins']}-{r['n']-r['wins']}). Keep sizing.",
        })
    for r in cold[:3]:
        insights.append({
            "tone": "bad",
            "text": f"{r['sport']}: consensus is {r['roi_pct']:+.1f}% ROI "
                    f"over {r['n']} picks ({r['wins']}-{r['n']-r['wins']}). "
                    f"Model needs tuning or we should sit the sport out.",
        })

    # Calibration gaps on consensus
    big_gaps = [b for b in calib if b["n"] >= MIN_SAMPLES_FOR_CALIB
                and abs(b["gap_pp"]) >= 5]
    for b in big_gaps[:3]:
        direction = "over-confident" if b["gap_pp"] < 0 else "under-confident"
        insights.append({
            "tone": "warn",
            "text": f"Consensus at {b['bin_lo']*100:.0f}–{b['bin_hi']*100:.0f}% "
                    f"is {direction} by {abs(b['gap_pp']):.1f}pp "
                    f"({b['n']} picks: predicted {b['expected']:.1f}%, hit {b['actual']:.1f}%).",
        })

    # Which model is winning per sport (lower Brier = better calibration)
    model_by_sport = defaultdict(dict)
    for r in metrics_rows:
        model_by_sport[r["sport"]][r["model"]] = r
    for sport, by_model in model_by_sport.items():
        if "pricing" not in by_model or "monte_carlo" not in by_model:
            continue
        pr = by_model["pricing"]; mc = by_model["monte_carlo"]
        if pr["n"] < 20 or mc["n"] < 20:
            continue
        diff = pr["brier"] - mc["brier"]
        if abs(diff) < 0.01:
            continue
        winner = "Monte Carlo" if diff > 0 else "pricing model"
        insights.append({
            "tone": "info",
            "text": f"{sport}: {winner} is the more honest signal "
                    f"(Brier {min(pr['brier'],mc['brier']):.3f} vs "
                    f"{max(pr['brier'],mc['brier']):.3f}, n≈{min(pr['n'],mc['n'])}).",
        })

    # Weight recommendations
    for sport, w in weights.items():
        if w["improvement_bp"] <= 5:
            continue  # not meaningful
        lean = "Monte Carlo" if w["mc_w"] > w["pricing_w"] else "pricing model"
        insights.append({
            "tone": "info",
            "text": f"{sport} consensus will shift to "
                    f"{int(w['pricing_w']*100)}% pricing / {int(w['mc_w']*100)}% MC "
                    f"(leaning on {lean}; -{w['improvement_bp']:.1f}bp Brier over "
                    f"{w['n']} picks).",
        })

    # Per-sport floor adjustments
    for sport, f in floors.items():
        insights.append({
            "tone": "info",
            "text": f"{sport}: raising min probability floor from 46% to "
                    f"{int(f['min_fair_prob']*100)}% "
                    f"(+{f['improvement_pp']:.1f}pp ROI over {f['n']} picks). "
                    f"Low-confidence {sport} picks will stop making the board.",
        })

    # Market blacklist
    for market, b in blacklist.items():
        insights.append({
            "tone": "bad",
            "text": f"Market blacklist: {market} is {b['wins']}-{b['losses']} "
                    f"({b['roi_pct']:+.1f}% ROI over {b['n']} picks). Dropping "
                    f"it from recommendations until it recovers.",
        })

    return insights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(lookback_days=LOOKBACK_DAYS, today=None):
    today = today or date.today()
    today_str = today.isoformat()
    picks = _load_recent_picks(lookback_days, today)
    metrics = model_metrics_per_sport(picks)
    calib = calibration_bins(picks)
    weights = recommended_weights(picks)
    floors = recommended_min_fair_prob(picks)
    blacklist = market_blacklist(picks)
    insights = generate_insights(metrics, calib, weights, floors, blacklist)
    settled_n = sum(1 for p in picks if _settled(p))
    result = {
        "generated_at":   datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days":  lookback_days,
        "total_picks":    len(picks),
        "settled_picks":  settled_n,
        "metrics":        metrics,
        "calibration":    calib,
        "recommended_weights":  weights,
        "recommended_floors":   floors,
        "market_blacklist":     blacklist,
        "insights":       insights,
    }
    # Persist locally + GitHub
    plays_log.ensure_log_dir()
    text = json.dumps(result)
    try:
        with open(analysis_file(today_str), "w") as f:
            f.write(text)
    except OSError:
        pass
    try:
        log_persist.write_file(
            f"{log_persist.DEFAULT_LOGS_PATH}/analysis_{today_str}.json", text,
            message=f"log: analysis {today_str}",
        )
    except Exception:
        pass
    return result


def read_latest_analysis():
    """Return the newest analysis file we have. Runs today's if missing."""
    today = date.today()
    for offset in range(0, 7):
        d = (today - timedelta(days=offset)).isoformat()
        if os.path.exists(analysis_file(d)):
            try:
                with open(analysis_file(d)) as f:
                    return json.load(f)
            except Exception:
                continue
        # Try GitHub too
        try:
            text, _ = log_persist.read_file(
                f"{log_persist.DEFAULT_LOGS_PATH}/analysis_{d}.json"
            )
            if text:
                return json.loads(text)
        except Exception:
            pass
    return None


def ensure_today_analysis():
    """Trigger today's analysis if we haven't already. Called from /logged."""
    today_str = date.today().isoformat()
    if os.path.exists(analysis_file(today_str)):
        try:
            with open(analysis_file(today_str)) as f:
                return json.load(f)
        except Exception:
            pass
    # Check GitHub before regenerating
    try:
        text, _ = log_persist.read_file(
            f"{log_persist.DEFAULT_LOGS_PATH}/analysis_{today_str}.json"
        )
        if text:
            try:
                plays_log.ensure_log_dir()
                with open(analysis_file(today_str), "w") as f:
                    f.write(text)
            except OSError:
                pass
            return json.loads(text)
    except Exception:
        pass
    return run_analysis()


# ---------------------------------------------------------------------------
# Feedback into picks.collect_picks
# ---------------------------------------------------------------------------

def get_sport_weights():
    """Return {sport_name: (pricing_w, mc_w)} from the latest analysis, or {}."""
    a = read_latest_analysis()
    if not a:
        return {}
    out = {}
    for sport, w in (a.get("recommended_weights") or {}).items():
        # Only apply the shift if it would actually help by > 5 Brier basis points
        if (w.get("improvement_bp") or 0) >= 5:
            out[sport] = (w.get("pricing_w", DEFAULT_WEIGHT),
                          w.get("mc_w", DEFAULT_WEIGHT))
    return out


def get_sport_min_prob():
    """Per-sport minimum consensus probability from the latest analysis."""
    a = read_latest_analysis()
    if not a:
        return {}
    out = {}
    for sport, f in (a.get("recommended_floors") or {}).items():
        if f.get("improvement_pp", 0) > 0:
            out[sport] = f.get("min_fair_prob", 0.46)
    return out


def get_market_blacklist():
    """Markets the picks pipeline should skip tomorrow."""
    a = read_latest_analysis()
    if not a:
        return set()
    return set((a.get("market_blacklist") or {}).keys())


def active_adjustments_summary():
    """Short structured summary of what the analyzer has applied to today's picks."""
    weights = get_sport_weights()
    floors = get_sport_min_prob()
    blacklist = get_market_blacklist()
    return {
        "weights":   {s: {"pricing": round(w[0], 2), "mc": round(w[1], 2)}
                      for s, w in weights.items()},
        "floors":    {s: round(f, 2) for s, f in floors.items()},
        "blacklist": sorted(blacklist),
        "any":       bool(weights or floors or blacklist),
    }


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

_BG_LOCK = __import__("threading").Lock()
_BG_RUNNING = {"flag": False, "started_at": None}


def start_background_analysis():
    """Fire run_analysis() in a daemon thread. Idempotent — if one is already
    running, this is a no-op. Returns True if a new thread was launched.
    """
    import threading
    with _BG_LOCK:
        if _BG_RUNNING["flag"]:
            return False
        _BG_RUNNING["flag"] = True
        _BG_RUNNING["started_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _runner():
        try:
            run_analysis()
        except Exception as e:
            # Keep going — this is a best-effort background job
            print(f"[model_analytics] background analysis failed: {e}", flush=True)
        finally:
            with _BG_LOCK:
                _BG_RUNNING["flag"] = False

    t = threading.Thread(target=_runner, daemon=True, name="model-analytics")
    t.start()
    return True


def background_status():
    with _BG_LOCK:
        return dict(_BG_RUNNING)


if __name__ == "__main__":
    r = run_analysis()
    print(f"analysis: {r['total_picks']} picks, {r['settled_picks']} settled")
    print(f"insights ({len(r['insights'])}):")
    for i in r["insights"]:
        print(f"  [{i['tone']}] {i['text']}")
