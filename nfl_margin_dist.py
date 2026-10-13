#!/usr/bin/env python3
"""Empirical NFL/NCAAF margin distributions.

Review flagged: a normal distribution ignores how often NFL games land on
key numbers (3, 7, 10, 14). Smooth models misprice alt-spread picks like
Titans +7.5 and Browns +4.5 because they don't know that 3-point and
7-point margins are *way* over-represented versus a Gaussian.

This module builds an empirical margin distribution from the nflverse
historical cache (nfl_multi_backtest_cache.json → 2600+ games) and exposes:

    sample_margin(projected_margin, rng, sport="nfl")
        → one sampled margin, properly centered around the projection
          while preserving the empirical shape (field goal / TD spikes).

    p_margin_ge(projected_margin, line, sport="nfl")
        → P(margin ≥ line) analytically, for alt-line spread pricing.

Centering: we hold the empirical distribution with mean 0 and shift it by
(projected_margin − empirical_mean). This preserves the shape (key number
clumping) while anchoring the median to our model's view of who wins.
"""
import json
import math
import os
import random
import time
from collections import Counter


CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
NFL_CACHE_PATH = os.path.join(CACHE_DIR, "nfl_multi_backtest_cache.json")

# Lazy-loaded in-process caches
_NFL_MARGINS = None       # list of (home_score - away_score) ints, centered
_NFL_MEAN    = None       # original mean (so we can decenter when sampling)
_NFL_LOADED_AT = 0

_NCAAF_MARGINS = None
_NCAAF_MEAN    = None

# When nflverse history isn't accessible we fall back to a synthetic
# distribution that still clusters at the real key numbers. These weights
# are approximate historical frequencies for NFL regular-season margins.
_NFL_KEY_NUMBER_WEIGHTS = {
    0: 0.5, 1: 2.8, 2: 2.3, 3: 15.8, 4: 4.8, 5: 2.0, 6: 5.5,
    7: 10.2, 8: 2.8, 9: 2.4, 10: 7.5, 11: 2.4, 12: 2.0, 13: 3.0,
    14: 5.5, 15: 1.8, 16: 2.2, 17: 3.8, 18: 1.6, 19: 1.4, 20: 2.4,
    21: 2.3, 22: 0.9, 23: 1.0, 24: 1.6, 25: 0.6, 26: 0.5, 27: 0.9,
    28: 1.1, 29: 0.4, 30: 0.4, 31: 0.5, 34: 0.5, 35: 0.4,
}

# CFB has its own key-number shape: 3/7 spikes are smaller than NFL (worse
# PAT reliability, more 2-pt tries, more blowouts), 10/14 spikes are
# flatter, and the long right tail is fatter (variance ~17 pts vs ~13 NFL).
# Rescaling NFL margins by 1.4x moves the 7-point spike to 10 (7*1.4=9.8),
# which is wrong — we need CFB-shaped keys at CFB-shaped positions.
# Source: Pinnacle/Vegas analyst margins across 10 years of FBS games.
_NCAAF_KEY_NUMBER_WEIGHTS = {
    0: 0.6, 1: 1.3, 2: 1.5, 3: 7.3, 4: 2.5, 5: 1.8, 6: 2.8,
    7: 6.1, 8: 1.9, 9: 1.6, 10: 4.8, 11: 1.7, 12: 1.5, 13: 2.1,
    14: 4.5, 15: 1.3, 16: 1.6, 17: 3.3, 18: 1.4, 19: 1.3, 20: 2.4,
    21: 2.9, 22: 1.0, 23: 1.1, 24: 1.9, 25: 0.9, 26: 0.7, 27: 1.1,
    28: 1.6, 29: 0.5, 30: 0.5, 31: 0.6, 32: 0.5, 33: 0.4, 34: 0.5,
    35: 0.6, 38: 0.4, 42: 0.4, 45: 0.3, 49: 0.2, 55: 0.2,
}


def _load_nfl_margins():
    """Pull every scored game out of the NFL backtest cache."""
    global _NFL_MARGINS, _NFL_MEAN, _NFL_LOADED_AT
    now = time.time()
    # Reuse the in-process cache for an hour (cache file changes rarely)
    if _NFL_MARGINS and (now - _NFL_LOADED_AT) < 3600:
        return _NFL_MARGINS, _NFL_MEAN
    margins = []
    if os.path.exists(NFL_CACHE_PATH):
        try:
            with open(NFL_CACHE_PATH) as f:
                data = json.load(f)
            for p in data.get("predictions", []):
                hs = p.get("home_score"); as_ = p.get("away_score")
                if hs is None or as_ is None:
                    continue
                margins.append(int(hs) - int(as_))
        except Exception:
            margins = []
    if margins:
        mean = sum(margins) / len(margins)
        # Keep margins as integers; shift at sample/analytic time so the
        # key-number clumps stay on the integer grid instead of smearing
        # across 0.35/0.65 after centering.
        _NFL_MARGINS = margins
        _NFL_MEAN    = mean
    else:
        # Synthetic fallback from key-number weights, symmetrized
        pool = []
        for m, w in _NFL_KEY_NUMBER_WEIGHTS.items():
            count = max(1, int(round(w * 10)))
            pool.extend([m] * count)
            if m > 0:
                pool.extend([-m] * count)
        mean = 0.0
        _NFL_MARGINS = pool
        _NFL_MEAN    = mean
    _NFL_LOADED_AT = now
    return _NFL_MARGINS, _NFL_MEAN


def _load_ncaaf_margins():
    """CFB uses its own synthetic key-number distribution. We don't have a
    local CFB game cache (cfb_elo_cache.json stores Elo state, not scored
    predictions), so we materialize a pool from CFB-specific weights that
    preserve the real key-number spikes at their real positions."""
    global _NCAAF_MARGINS, _NCAAF_MEAN
    if _NCAAF_MARGINS is not None:
        return _NCAAF_MARGINS, _NCAAF_MEAN
    pool = []
    for m, w in _NCAAF_KEY_NUMBER_WEIGHTS.items():
        count = max(1, int(round(w * 20)))  # 20× scaling → ~2000-game pool
        pool.extend([m] * count)
        if m > 0:
            pool.extend([-m] * count)
    _NCAAF_MARGINS = pool
    _NCAAF_MEAN    = sum(pool) / len(pool) if pool else 0.0
    return _NCAAF_MARGINS, _NCAAF_MEAN


def _get_dist(sport):
    if sport == "ncaaf":
        return _load_ncaaf_margins()
    return _load_nfl_margins()


def sample_margin(projected_margin, rng=None, sport="nfl"):
    """Draw one margin from the empirical distribution, shifted so its
    median matches the model's projected margin."""
    rng = rng or random
    pool, mean = _get_dist(sport)
    if not pool:
        return int(round(rng.gauss(projected_margin, 13.0)))
    pick = pool[rng.randrange(len(pool))]
    # pool is integer-valued; shift so pool mean maps to projected_margin
    return int(round(pick - mean + projected_margin))


def p_margin_ge(projected_margin, line, sport="nfl"):
    """Analytic P(sim margin ≥ line) using the shifted empirical distribution."""
    pool, mean = _get_dist(sport)
    if not pool:
        sigma = 13.0 if sport == "nfl" else 15.0
        z = (line - projected_margin) / sigma
        return 1 - _phi(z)
    # pool[i] − mean + projected_margin ≥ line  ⇔  pool[i] ≥ line + mean − projected_margin
    threshold = line + mean - projected_margin
    hits = sum(1 for m in pool if m >= threshold)
    return hits / len(pool)


def p_margin_le(projected_margin, line, sport="nfl"):
    pool, mean = _get_dist(sport)
    if not pool:
        sigma = 13.0 if sport == "nfl" else 15.0
        z = (line - projected_margin) / sigma
        return _phi(z)
    threshold = line + mean - projected_margin
    hits = sum(1 for m in pool if m <= threshold)
    return hits / len(pool)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def key_number_weight(margin, sport="nfl"):
    """Debug helper: return the empirical mass at exactly this margin."""
    pool, _mean = _get_dist(sport)
    if not pool:
        return 0.0
    return sum(1 for m in pool if m == margin) / len(pool)


if __name__ == "__main__":
    pool, mean = _load_nfl_margins()
    print(f"NFL empirical margins: {len(pool)} games (mean {mean:+.2f})")
    counts = Counter(pool)
    total = len(pool)
    print("Key-number frequency (raw dist):")
    for k in (-14, -10, -7, -3, 0, 3, 7, 10, 14):
        pct = counts.get(k, 0) / total * 100 if total else 0
        print(f"  {k:+3d}:  {pct:5.2f}%")
    # Illustrate the alt-line win-prob curve for a 3-point favorite
    print("\nP(home covers X) for a 3-point favorite (projected margin +3):")
    for line in (-7.5, -3.5, -0.5, +2.5, +6.5, +10.5):
        p = p_margin_ge(3, line + 0.001)
        print(f"  home -{line:+.1f}: {p*100:5.1f}%")
