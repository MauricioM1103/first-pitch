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

def summarize_recent(days=30):
    """Compute W-L-P + net $ over the last `days` days of settlements.
    Returns dict with the same shape plays_log.summary uses so /logged
    can render it with the same tile template.

    Dollar amounts come straight from Kalshi (cents → dollars). YES-settled
    positions where we were YES or NO-settled where we were NO = win. The
    opposite = loss. revenue field is signed net cents already so we can
    aggregate directly."""
    if not is_configured():
        return None
    # Fetch settlements from the last `days` days
    cutoff = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    rows = get_settlements(limit=500, min_ts=cutoff)
    if not rows:
        return {"wins": 0, "losses": 0, "pushes": 0, "settled": 0,
                "pending": 0, "net_dollars": 0.0, "revenue_cents": 0,
                "n_trades": 0}
    w = l = pu = 0
    net_cents = 0
    for s in rows:
        rev = s.get("revenue") or 0
        try:
            rev = int(rev)
        except (TypeError, ValueError):
            rev = 0
        net_cents += rev
        if rev > 0:
            w += 1
        elif rev < 0:
            l += 1
        else:
            pu += 1
    settled = w + l + pu
    hit_pct = (w / settled * 100) if settled else 0.0
    return {
        "wins":          w,
        "losses":        l,
        "pushes":        pu,
        "settled":       settled,
        "pending":       0,
        "net_dollars":   round(net_cents / 100.0, 2),
        "revenue_cents": net_cents,
        "n_trades":      len(rows),
        "win_pct":       hit_pct,
        "roi_pct":       0.0,  # Kalshi doesn't expose cost basis per settlement in one call
    }


def recent_settlements_table(days=14, limit=50):
    """Return rows for a table on /logged: [{ticker, settled_time, market_result,
    yes_count, no_count, net_dollars, when}, ...] newest first."""
    if not is_configured():
        return []
    cutoff = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    rows = get_settlements(limit=limit, min_ts=cutoff)
    out = []
    for s in rows:
        rev = s.get("revenue") or 0
        try:
            rev = int(rev)
        except (TypeError, ValueError):
            rev = 0
        settled_ts = s.get("settled_time") or ""
        out.append({
            "ticker":        s.get("ticker") or "",
            "market_result": s.get("market_result") or "",
            "yes_count":     s.get("yes_count") or 0,
            "no_count":      s.get("no_count") or 0,
            "net_dollars":   round(rev / 100.0, 2),
            "settled_time":  settled_ts,
        })
    # Sort newest first
    out.sort(key=lambda r: r.get("settled_time") or "", reverse=True)
    return out
