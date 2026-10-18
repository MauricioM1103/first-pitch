#!/usr/bin/env python3
"""Kalshi portfolio API — authenticated endpoints for the user's own
account. Pulls settlements, positions, and fills so /logged can show a
real-$ record for picks that actually hit Kalshi.

Separate from kalshi_odds.py (which just reads public market prices and
needs no auth). This module adds Ed25519 request signing per Kalshi's
newer API-key auth flow.

Credentials come from Render env / Secret Files (never the public
/settings form):
  KALSHI_API_KEY_ID       = the key UUID from Kalshi -> Account -> API Keys
  KALSHI_PRIVATE_KEY_PEM  = the full PEM text of the private key file you
                            downloaded when you created the key
Both are optional — if either is missing, every function here returns
empty / None without raising, so the rest of the site is unaffected.
"""
import base64
import json
import os
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_UA   = "cheeks-model/1.0 (+https://cheeks-model.onrender.com)"
_TIMEOUT_S = 10

# In-memory cache so we don't re-hit Kalshi on every /logged render.
_CACHE = {}
_CACHE_TTL_S = 10 * 60   # 10 min — portfolio settles slowly


# ---------------- credential loading ----------------

def _creds():
    """Return (api_key_id, private_key_pem) or (None, None) if either
    is missing. Checks environment variables first, then Render Secret
    Files at /etc/secrets/."""
    kid = os.environ.get("KALSHI_API_KEY_ID")
    pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
    if not kid:
        try:
            with open("/etc/secrets/KALSHI_API_KEY_ID") as f:
                kid = f.read().strip()
        except Exception:
            pass
    if not pem:
        try:
            with open("/etc/secrets/KALSHI_PRIVATE_KEY_PEM") as f:
                pem = f.read()
        except Exception:
            pass
    return (kid or None), (pem or None)


def is_configured():
    kid, pem = _creds()
    return bool(kid and pem)


# ---------------- Ed25519 signing ----------------
#
# Kalshi auth per https://trading-api.readme.io/reference/api-keys :
# message = str(unix_ms) + method + path     (query string excluded)
# signature = base64(ed25519_private_key.sign(message.encode()))
# Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
#
# We lazy-import `cryptography` so the module imports cleanly on dev
# machines without it. Any auth-needing call will raise if the pkg
# isn't installed.

_PRIVATE_KEY_CACHE = {"pem": None, "key": None}


def _load_private_key(pem):
    """Parse the PEM private key once per process. Returns an
    Ed25519PrivateKey or None if the pkg isn't available / parse fails."""
    if _PRIVATE_KEY_CACHE["pem"] == pem and _PRIVATE_KEY_CACHE["key"] is not None:
        return _PRIVATE_KEY_CACHE["key"]
    try:
        from cryptography.hazmat.primitives import serialization
    except Exception:
        return None
    try:
        key = serialization.load_pem_private_key(pem.encode(), password=None)
    except Exception:
        return None
    _PRIVATE_KEY_CACHE["pem"] = pem
    _PRIVATE_KEY_CACHE["key"] = key
    return key


def _signed_headers(method, path):
    kid, pem = _creds()
    if not (kid and pem):
        return None
    key = _load_private_key(pem)
    if key is None:
        return None
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    try:
        sig = base64.b64encode(key.sign(msg)).decode()
    except Exception:
        return None
    return {
        "KALSHI-ACCESS-KEY":       kid,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "User-Agent":              _UA,
        "Accept":                  "application/json",
    }


# ---------------- API calls ----------------

def _get(path, params=None):
    """Signed GET; returns parsed JSON dict or {} on any failure. `path`
    must start with /trade-api/v2/ so the signature matches the server
    canonicalization."""
    headers = _signed_headers("GET", path)
    if headers is None:
        return {}
    url = f"https://api.elections.kalshi.com{path}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def get_settlements(limit=200, min_ts=None):
    """Return list of settlement records. Each has `ticker`, `market_result`
    (yes/no), `yes_count`, `no_count`, `revenue` (cents, net P&L), and
    `settled_time`. Cached 10 min."""
    cache_key = ("settlements", limit, min_ts)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    out = []
    cursor = None
    # Paginate up to `limit` records total.
    while len(out) < limit:
        params = {"limit": min(200, limit - len(out))}
        if cursor:
            params["cursor"] = cursor
        if min_ts:
            params["min_ts"] = min_ts
        data = _get("/trade-api/v2/portfolio/settlements", params=params)
        batch = data.get("settlements") or []
        if not batch:
            break
        out.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
    _CACHE[cache_key] = (time.time(), out)
    return out


def get_positions():
    """Return list of open position records."""
    cache_key = ("positions",)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    data = _get("/trade-api/v2/portfolio/positions", params={"limit": 200})
    pos = (data.get("market_positions") or [])
    _CACHE[cache_key] = (time.time(), pos)
    return pos


def get_fills(limit=100, min_ts=None):
    """Return list of fill records (filled orders)."""
    cache_key = ("fills", limit, min_ts)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    params = {"limit": limit}
    if min_ts:
        params["min_ts"] = min_ts
    data = _get("/trade-api/v2/portfolio/fills", params=params)
    fills = data.get("fills") or []
    _CACHE[cache_key] = (time.time(), fills)
    return fills


def invalidate_cache():
    _CACHE.clear()


# ---------------- Analysis / aggregation ----------------
#
# P&L CALCULATION
# ---------------
# Kalshi's /portfolio/settlements `revenue` field is ambiguous — their
# docs are unclear whether it's gross payout or net profit, and the
# community has reported both interpretations. To be safe we IGNORE
# `revenue` and compute P&L ourselves from the primitive facts:
#
#   For each settlement:
#     payout_per_yes_contract = 100¢ if market_result == 'yes' else 0
#     payout_per_no_contract  = 100¢ if market_result == 'no'  else 0
#     payout = (yes_count * yes_payout) + (no_count * no_payout)
#
#   For each fill on that ticker:
#     if side == 'yes' and action == 'buy':   cost += price * count
#     if side == 'yes' and action == 'sell':  cost -= price * count   (close for cash)
#     if side == 'no'  and action == 'buy':   cost += price * count
#     if side == 'no'  and action == 'sell':  cost -= price * count
#
#   P&L (cents) = payout - cost_basis
#
# This is the Kalshi version of "payout minus wager" and matches what
# shows up on your account statement regardless of their internal
# `revenue` convention. We also expose the raw `revenue` field in the
# detail row so you can cross-check.
#
# Edge cases:
# - market_result == 'void' or 'all_no' (rare for sports): both yes
#   and no settle at 0, so payout = 0, P&L = -cost_basis (full loss
#   on both sides... but you shouldn't have both sides open).
# - Fills for a ticker returned 0 but settlement shows a position:
#   Kalshi might have purged old fills. Fall back to using
#   avg_entry_price if available on the settlement row.


def _fills_cost_basis_cents(fills, ticker):
    """Return (yes_cost, no_cost, yes_contracts, no_contracts) for a
    ticker. Costs include fees (Kalshi deducts a small fee per contract
    at trade time; the fill rows include a `fees` field in cents)."""
    yes_cost = 0
    no_cost = 0
    yes_qty = 0
    no_qty = 0
    for f in (fills or []):
        if f.get("ticker") != ticker:
            continue
        try:
            count = int(f.get("count") or 0)
            # Fill price is in cents (0-100). For a sell, we received
            # cents instead of paying them — treat as negative cost.
            # Kalshi reports `yes_price` for YES-side and `no_price`
            # for NO-side fills; whichever is present identifies the side.
            yes_price = f.get("yes_price")
            no_price = f.get("no_price")
            side = (f.get("side") or "").lower()
            action = (f.get("action") or "").lower()
            fees = int(f.get("fees") or 0)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        sign = -1 if action == "sell" else 1
        if side == "yes" or (yes_price is not None and side != "no"):
            price = int(yes_price if yes_price is not None else 0)
            yes_cost += sign * count * price + fees
            yes_qty  += sign * count
        else:
            price = int(no_price if no_price is not None else 0)
            no_cost += sign * count * price + fees
            no_qty  += sign * count
    return yes_cost, no_cost, yes_qty, no_qty


def _pnl_for_settlement(settlement, fills):
    """Return net P&L in cents for one settlement row, using fills for
    cost basis. Returns None if we can't compute (eg no fills for the
    ticker)."""
    ticker = settlement.get("ticker")
    if not ticker:
        return None
    result = (settlement.get("market_result") or "").lower()
    yes_count = int(settlement.get("yes_count") or 0)
    no_count  = int(settlement.get("no_count") or 0)
    # Payouts in cents.
    yes_payout_per = 100 if result == "yes" else 0
    no_payout_per  = 100 if result == "no"  else 0
    payout = (yes_count * yes_payout_per) + (no_count * no_payout_per)
    yes_cost, no_cost, _yq, _nq = _fills_cost_basis_cents(fills, ticker)
    cost = yes_cost + no_cost
    # If we have NO fills recorded for this ticker (fills pagination
    # may have cut it off), use Kalshi's own revenue field as a last
    # resort rather than reporting P&L=0.
    if cost == 0 and (yes_count or no_count):
        rev = settlement.get("revenue")
        try:
            return int(rev) if rev is not None else None
        except (TypeError, ValueError):
            return None
    return payout - cost


def summarize_recent(days=30):
    """Compute W-L-P + net $ over the last `days` days of settlements.
    Uses explicit (payout - cost_basis) calculation from fills — not
    Kalshi's ambiguous `revenue` field. Returns dict shaped like
    plays_log.summary so /logged can drop it into its tile template."""
    if not is_configured():
        return None
    cutoff = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    settlements = get_settlements(limit=500, min_ts=cutoff)
    if not settlements:
        return {"wins": 0, "losses": 0, "pushes": 0, "settled": 0,
                "pending": 0, "net_dollars": 0.0, "revenue_cents": 0,
                "n_trades": 0, "win_pct": 0.0, "roi_pct": 0.0,
                "cost_dollars": 0.0}
    # Pull fills covering the same window plus a buffer for late settles.
    fills = get_fills(limit=1000, min_ts=max(0, cutoff - 7 * 86400))

    w = l = pu = 0
    net_cents = 0
    gross_cost_cents = 0
    for s in settlements:
        pnl = _pnl_for_settlement(s, fills)
        if pnl is None:
            continue
        net_cents += pnl
        if pnl > 0:
            w += 1
        elif pnl < 0:
            l += 1
        else:
            pu += 1
        # Track cost basis for ROI
        yes_cost, no_cost, _, _ = _fills_cost_basis_cents(
            fills, s.get("ticker"))
        gross_cost_cents += max(0, yes_cost + no_cost)
    settled = w + l + pu
    hit_pct = (w / settled * 100) if settled else 0.0
    roi_pct = (net_cents / gross_cost_cents * 100) if gross_cost_cents else 0.0
    return {
        "wins":          w,
        "losses":        l,
        "pushes":        pu,
        "settled":       settled,
        "pending":       0,
        "net_dollars":   round(net_cents / 100.0, 2),
        "revenue_cents": net_cents,
        "n_trades":      len(settlements),
        "win_pct":       hit_pct,
        "roi_pct":       roi_pct,
        "cost_dollars":  round(gross_cost_cents / 100.0, 2),
    }


def recent_settlements_table(days=14, limit=50):
    """Rows for the /logged settlements table. Net $ computed from
    (payout - cost_basis) per the module header comment, with Kalshi's
    raw `revenue` field exposed for cross-check."""
    if not is_configured():
        return []
    cutoff = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    settlements = get_settlements(limit=limit, min_ts=cutoff)
    fills = get_fills(limit=500, min_ts=max(0, cutoff - 7 * 86400))
    out = []
    for s in settlements:
        pnl = _pnl_for_settlement(s, fills)
        raw_rev = s.get("revenue")
        try:
            raw_rev = int(raw_rev) if raw_rev is not None else None
        except (TypeError, ValueError):
            raw_rev = None
        out.append({
            "ticker":        s.get("ticker") or "",
            "market_result": s.get("market_result") or "",
            "yes_count":     int(s.get("yes_count") or 0),
            "no_count":      int(s.get("no_count") or 0),
            "net_dollars":   round((pnl or 0) / 100.0, 2) if pnl is not None else None,
            "raw_revenue_dollars": round(raw_rev / 100.0, 2) if raw_rev is not None else None,
            "settled_time":  s.get("settled_time") or "",
        })
    out.sort(key=lambda r: r.get("settled_time") or "", reverse=True)
    return out
