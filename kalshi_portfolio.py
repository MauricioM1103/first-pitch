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

P&L CALCULATION
---------------
Settlements come back with cost basis + fees already summed by Kalshi:
    yes_count_fp              = contracts of YES held at settlement (string "42.51")
    no_count_fp               = contracts of NO held at settlement
    yes_total_cost_dollars    = total $ paid for the YES side (string "19.257030")
    no_total_cost_dollars     = total $ paid for the NO side
    fee_cost                  = total trade fees paid (string dollars)
    revenue                   = GROSS payout at settlement in cents
                                = yes_count * (100 if yes won else 0)
                                + no_count * (100 if no won else 0)
    market_result             = "yes" / "no" / "scalar"

Net P&L (cents) = revenue - round(100 * yes_total_cost) - round(100 * no_total_cost) - round(100 * fee_cost)

We don't need to walk fills for cost basis anymore — Kalshi stamps the
settlement row with the per-side totals. Fills are still used in the
diagnostic view for drill-downs, but not in the aggregate.
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

# Fixed anchor for the Kalshi record. Everything settled before this
# date is excluded from the /logged tile and settlement table. Change
# this if you want to re-anchor the record (e.g. after a bankroll reset).
RECORD_START_DATE = "2026-08-18"

# In-memory cache so we don't re-hit Kalshi on every /logged render.
_CACHE = {}
_CACHE_TTL_S = 10 * 60   # 10 min — portfolio settles slowly


def _record_start_ts():
    """Unix timestamp (seconds) for the record anchor."""
    try:
        return int(datetime.strptime(RECORD_START_DATE, "%Y-%m-%d").timestamp())
    except Exception:
        return 0


def _settled_after_anchor(settlement):
    """True if the settlement row's `settled_time` is on/after the record
    anchor. Returns True (included) when the field is missing/unparseable
    so we don't silently drop data."""
    t = (settlement or {}).get("settled_time") or ""
    if not t:
        return True
    try:
        # settled_time example: "2026-10-18T06:49:06.986398Z"
        return t[:10] >= RECORD_START_DATE
    except Exception:
        return True


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

_PRIVATE_KEY_CACHE = {"pem": None, "key": None}


def _load_private_key(pem):
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
    """Return list of settlement records. Cached 10 min."""
    cache_key = ("settlements", limit, min_ts)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    out = []
    cursor = None
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
    cache_key = ("positions",)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    data = _get("/trade-api/v2/portfolio/positions", params={"limit": 200})
    pos = (data.get("market_positions") or [])
    _CACHE[cache_key] = (time.time(), pos)
    return pos


def get_fills(limit=100, min_ts=None):
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


# ---------------- Field helpers ----------------

def _dollars_to_cents(s):
    """Parse Kalshi's fixed-point dollar string into integer cents.
    "19.257030" -> 1926. Negative/None/invalid -> 0."""
    if s is None:
        return 0
    try:
        return int(round(float(str(s)) * 100))
    except (TypeError, ValueError):
        return 0


def _fp_to_float(s):
    """Parse Kalshi's fixed-point string (used for contract counts) into
    a float. "42.51" -> 42.51."""
    if s is None:
        return 0.0
    try:
        return float(str(s))
    except (TypeError, ValueError):
        return 0.0


# ---------------- Analysis / aggregation ----------------

def _pnl_for_settlement(settlement, _unused_fills=None):
    """Return net P&L in cents for one settlement row. Uses the cost
    basis + fees Kalshi stamps on the settlement itself, so fills are
    not consulted.

    Formula:
      pnl_cents = revenue
                - round(100 * yes_total_cost_dollars)
                - round(100 * no_total_cost_dollars)
                - round(100 * fee_cost)

    Returns None only if the settlement is missing a `revenue` field
    entirely (should never happen for a real row)."""
    try:
        revenue = settlement.get("revenue")
        if revenue is None:
            return None
        revenue = int(revenue)
    except (TypeError, ValueError):
        return None
    yes_cost_c = _dollars_to_cents(settlement.get("yes_total_cost_dollars"))
    no_cost_c  = _dollars_to_cents(settlement.get("no_total_cost_dollars"))
    fee_c      = _dollars_to_cents(settlement.get("fee_cost"))
    return revenue - yes_cost_c - no_cost_c - fee_c


def _settlement_cost_cents(settlement):
    """Total gross cost (contracts + fees) for this settlement in cents."""
    return (_dollars_to_cents(settlement.get("yes_total_cost_dollars"))
            + _dollars_to_cents(settlement.get("no_total_cost_dollars"))
            + _dollars_to_cents(settlement.get("fee_cost")))


def summarize_recent(days=None):
    """Compute W-L-P + net $ since the record anchor (RECORD_START_DATE).

    `days` is accepted for backward compat but ignored — the anchor is
    the source of truth so entries don't fall off a rolling window.
    """
    if not is_configured():
        return None
    cutoff = _record_start_ts()
    # Pull generously (365 days) so pagination doesn't cap before the
    # anchor is reached, then filter precisely by settled_time.
    settlements = get_settlements(limit=1000, min_ts=cutoff) or []
    settlements = [s for s in settlements if _settled_after_anchor(s)]
    if not settlements:
        return {"wins": 0, "losses": 0, "pushes": 0, "settled": 0,
                "pending": 0, "net_dollars": 0.0, "revenue_cents": 0,
                "n_trades": 0, "win_pct": 0.0, "roi_pct": 0.0,
                "cost_dollars": 0.0,
                "record_start": RECORD_START_DATE}

    w = l = pu = 0
    net_cents = 0
    gross_cost_cents = 0
    for s in settlements:
        pnl = _pnl_for_settlement(s)
        if pnl is None:
            continue
        net_cents += pnl
        if pnl > 0:
            w += 1
        elif pnl < 0:
            l += 1
        else:
            pu += 1
        gross_cost_cents += _settlement_cost_cents(s)
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
        "record_start":  RECORD_START_DATE,
    }


def recent_settlements_table(days=None, limit=50):
    """Rows for the /logged settlements table since RECORD_START_DATE.
    Net $ is explicit P&L (revenue - cost - fees). `raw_revenue_dollars`
    kept for cross-check.

    `days` is accepted for backward compat but ignored — the anchor is
    the source of truth."""
    if not is_configured():
        return []
    cutoff = _record_start_ts()
    settlements = get_settlements(limit=max(limit, 500), min_ts=cutoff) or []
    settlements = [s for s in settlements if _settled_after_anchor(s)]
    out = []
    for s in settlements:
        pnl = _pnl_for_settlement(s)
        raw_rev = s.get("revenue")
        try:
            raw_rev = int(raw_rev) if raw_rev is not None else None
        except (TypeError, ValueError):
            raw_rev = None
        yes_qty = _fp_to_float(s.get("yes_count_fp"))
        no_qty  = _fp_to_float(s.get("no_count_fp"))
        out.append({
            "ticker":        s.get("ticker") or "",
            "market_result": s.get("market_result") or "",
            "yes_count":     yes_qty,
            "no_count":      no_qty,
            "net_dollars":   round((pnl or 0) / 100.0, 2) if pnl is not None else None,
            "raw_revenue_dollars": round(raw_rev / 100.0, 2) if raw_rev is not None else None,
            "settled_time":  s.get("settled_time") or "",
        })
    out.sort(key=lambda r: r.get("settled_time") or "", reverse=True)
    if limit and len(out) > limit:
        out = out[:limit]
    return out


# ---------------- Fills helpers (kept for the /debug view) ----------------

def _fills_cost_basis_cents(fills, ticker):
    """DEPRECATED for the aggregate P&L (we use settlement row totals
    now) but kept so /debug/kalshi can replay fills side-by-side."""
    yes_cost = no_cost = 0
    yes_qty  = no_qty  = 0.0
    for f in (fills or []):
        tkr = f.get("ticker") or f.get("market_ticker") or ""
        if tkr != ticker:
            continue
        count = _fp_to_float(f.get("count_fp"))
        outcome = (f.get("outcome_side") or f.get("side") or "").lower()
        book    = (f.get("book_side") or "").lower()
        action  = (f.get("action") or "").lower()
        if book:
            # bid = you bought this outcome, ask = you sold
            sign = 1 if book == "bid" else -1
        elif action:
            sign = -1 if action == "sell" else 1
        else:
            sign = 1
        yes_px_c = _dollars_to_cents(f.get("yes_price_dollars"))
        no_px_c  = _dollars_to_cents(f.get("no_price_dollars"))
        fee_c    = _dollars_to_cents(f.get("fee_cost"))
        if outcome == "yes":
            yes_cost += int(round(sign * count * yes_px_c)) + fee_c
            yes_qty  += sign * count
        else:
            no_cost  += int(round(sign * count * no_px_c)) + fee_c
            no_qty   += sign * count
    return yes_cost, no_cost, yes_qty, no_qty
