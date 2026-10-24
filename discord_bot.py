#!/usr/bin/env python3
"""Discord picks poster (Phase 31).

Posts the day's picks to a Discord channel at 12 AM CT (via /cron/daily-picks)
and sends live update images when picks are added or dropped as DO NOT BET.
Images are rendered server-side with Pillow — 1080x1920 vertical format
matching the social-slides template, 5 picks per PNG so the full day's slate
splits cleanly into 2 or 3 images depending on count.

Discord webhooks accept multipart file uploads up to 25 MB. A text-only
message with ~5 picks per image lands at ~80-120 KB per PNG — no risk of
hitting the limit.
"""
import io
import json
import os
import threading
import time
from datetime import date as _date, datetime, time as _time, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOT_PATH         = os.path.join(_BASE_DIR, "logs", "discord_last_snapshot.json")
_TOP_SNAP_PATH         = os.path.join(_BASE_DIR, "logs", "discord_top_n.json")
_LAST_ALERT_PATH       = os.path.join(_BASE_DIR, "logs", "discord_last_alert.json")
_DNB_SENT_PATH         = os.path.join(_BASE_DIR, "logs", "discord_dnb_sent.json")
_RESULT_SENT_PATH      = os.path.join(_BASE_DIR, "logs", "discord_result_sent.json")
_CANDIDATES_SENT_PATH  = os.path.join(_BASE_DIR, "logs", "discord_candidates_sent.json")

# Visual constants — matches the social-slides template.
_W, _H = 1080, 1920
_PICKS_PER_PNG = 9
# Phase 31 v2: three fixed slates per day, top 5 per slate, "Signal 3.5"
# effective threshold:
#   - Signal >= 4 picks always qualify (strong across most signals)
#   - Signal 3 picks qualify only if their EV cushion is strong enough
#     (>= _SIGNAL_3_MIN_EV). The marginal Signal 3 picks — those that
#     just barely cleared the composite — are the churn-prone edge, so
#     holding them to a stricter EV bar filters out the noise without
#     starving the feed on days without many Signal 4+ picks.
_TOP_N = 5
_MIN_SIGNAL_FOR_POST = 3
_SIGNAL_3_MIN_EV = 3.0  # percent

# EOD summary pagination.
# 9 cards per PNG mirrors the slate layout, 3 PNG max gives us up to
# 27 picks across the day. More than 27 very rarely happens; if it
# does we drop the lowest-signal pending picks past the cap.
_EOD_PER_PNG = 9
_EOD_MAX_PNGS = 3


_SPORT_FLOORS_PATH = os.path.join(_BASE_DIR, "logs",
                                   "sport_signal_floors.json")
_SPORT_FLOORS_CACHE = {"data": None, "ts": 0.0}
_SPORT_FLOORS_TTL = 300.0  # 5 min


def _load_sport_signal_floors():
    """Return {sport_slug: required_signal} map. Sports with a weak
    Signal 3 hit rate get their floor bumped to 4 so the Discord feed
    doesn't post picks from models that aren't performing."""
    import time as _t
    now = _t.time()
    if (_SPORT_FLOORS_CACHE["data"] is not None
            and (now - _SPORT_FLOORS_CACHE["ts"]) < _SPORT_FLOORS_TTL):
        return _SPORT_FLOORS_CACHE["data"]
    try:
        with open(_SPORT_FLOORS_PATH) as f:
            data = json.load(f) or {}
    except Exception:
        data = {}
    _SPORT_FLOORS_CACHE["data"] = data
    _SPORT_FLOORS_CACHE["ts"] = now
    return data


def _qualifies_for_discord(p):
    """Signal 3.5 gate with per-sport dynamic floors.

    Rules:
      - Signal >= 4 picks always qualify.
      - Signal 3 picks qualify only if EV >= _SIGNAL_3_MIN_EV AND the
        sport's historical hit rate at Signal 3 is healthy (floor set
        at 3 in _SPORT_FLOORS_PATH).
      - A sport whose Signal 3 picks have been hitting <48% long-term
        has its floor auto-bumped to 4 by nightly recompute, so even
        high-EV Signal 3 picks are blocked for that sport."""
    s = int(p.get("signal_score") or 0)
    if s >= 4:
        return True
    sport = (p.get("sport_slug") or p.get("sport") or "").lower()
    floors = _load_sport_signal_floors()
    required = int(floors.get(sport, _MIN_SIGNAL_FOR_POST))
    if required > 3:
        # Sport's Signal 3 is cold — require >= 4 for this sport.
        return False
    if s >= _MIN_SIGNAL_FOR_POST:
        try:
            return float(p.get("ev_pct") or 0.0) >= _SIGNAL_3_MIN_EV
        except (TypeError, ValueError):
            return False
    return False


def refresh_sport_signal_floors(graded_by_date):
    """Nightly recompute: scan last 30 days of graded picks, bump floor
    to 4 for any sport whose Signal 3 picks (current-filter only) have
    hit <48% on >=10 settled bets. Called from the EOD step."""
    buckets = {}  # sport -> [W, L]
    try:
        import plays_log
    except Exception:
        return {}
    for ds, picks in (graded_by_date or {}).items():
        for p in picks or []:
            if not plays_log._passes_current_filter(p):
                continue
            sig = int(p.get("signal_score") or 0)
            if sig != 3:
                continue
            r = p.get("result")
            if r not in ("W", "L"):
                continue
            sport = (p.get("sport") or "").lower()
            if not sport:
                continue
            buckets.setdefault(sport, [0, 0])
            buckets[sport][0 if r == "W" else 1] += 1
    floors = {}
    for sport, (w, l) in buckets.items():
        n = w + l
        if n < 10:
            continue
        hit = w / n
        if hit < 0.48:
            floors[sport] = 4
    try:
        os.makedirs(os.path.dirname(_SPORT_FLOORS_PATH), exist_ok=True)
        with open(_SPORT_FLOORS_PATH, "w") as f:
            json.dump(floors, f)
    except Exception:
        pass
    _SPORT_FLOORS_CACHE["data"] = floors
    _SPORT_FLOORS_CACHE["ts"] = time.time()
    return floors
# Hard DNB threshold — only alert when a posted pick's signal collapses
# this low. Marginal drift (Signal 4 -> 3) no longer fires DNBs.
_HARD_DNB_SIGNAL = 1
# Legacy constant some callers still reference. New flow uses slates.
_HALF_CUTOFF_HOUR = 15
_LIVE_ALERT_COOLDOWN_SEC = 2 * 3600

# Slate windows in Central Time. Each tuple is (start_hour, end_hour)
# for games starting in [start, end). Picks posted at the slate's own
# post time so you have ~1-2 hours to act before the earliest game.
_SLATE_WINDOWS = {
    1: (0,  12),   # Morning:   games before 12 PM CT
    2: (12, 18),   # Afternoon: games 12 PM - 6 PM CT
    3: (18, 28),   # Prime:     games 6 PM CT+ (28 = next-day 4 AM, catches 10 PM+)
}
_SLATE_LABELS = {
    0: "Day",           # fallback bucket when exact slate is unknown
    1: "Morning",
    2: "Afternoon",
    3: "Prime Time",
}
_SLATE_EMOJI = {
    0: ":bar_chart:",
    1: ":sun_with_face:",
    2: ":partly_sunny:",
    3: ":city_sunset:",
}
# Post times (hour, minute) in Central Time. Each ~1-2h before its
# slate's typical earliest game.
_SLATE_POST_TIMES = {
    1: (7,  0),    # 7:00 AM CT
    2: (12, 30),   # 12:30 PM CT
    3: (18, 0),    # 6:00 PM CT
}
# End-of-day summary post time (hour, minute, next_day_flag).
# 2 AM CT next day — gives late Prime Time games (NFL SNF, UFC main
# event, west-coast MLB) that end 11 PM-1 AM time to fully grade
# before the summary fires.
_EOD_POST_TIME = (2, 0, True)

# Colors (match the dark theme on the live site).
_BG         = (14, 18, 24)
_CARD       = (24, 29, 37)
_INK        = (240, 243, 248)
_MUTED      = (148, 163, 184)
_MUTED_2    = (100, 116, 139)
_RULE       = (49, 58, 73)
_GOOD       = (34, 197, 94)
_BAD        = (239, 68, 68)
_WARN       = (245, 158, 11)
_STAR       = (245, 158, 11)
_SIGNAL_3   = (56, 189, 248)
_LOCK       = (34, 197, 94)
_DNB        = (239, 68, 68)


def _font(size, bold=False):
    """Load a system font, falling back to the PIL default if nothing is
    installed. Render boxes aren't picky about typography — any clean
    sans-serif works."""
    candidates = []
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_text(draw, xy, text, font, fill=_INK, anchor=None, max_w=None):
    """Simple text draw with optional max-width truncation (adds ...)."""
    if max_w and font is not None:
        try:
            w = draw.textbbox((0, 0), text, font=font)[2]
            while w > max_w and len(text) > 3:
                text = text[:-2]
                w = draw.textbbox((0, 0), text + "...", font=font)[2]
            if w > max_w:
                text = text
            elif draw.textbbox((0, 0), text, font=font)[2] < max_w and text.endswith("..."):
                pass
            else:
                if text != "" and draw.textbbox((0, 0), text, font=font)[2] >= max_w:
                    text = text + "..."
        except Exception:
            pass
    if anchor:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    else:
        draw.text(xy, text, font=font, fill=fill)


def _draw_pill(draw, x, y, text, font, fill, bg):
    """Draw a rounded-rect pill with text inside. Returns (right_x, bottom_y)."""
    pad_x, pad_y = 16, 8
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 100, 20
    w, h = tw + pad_x * 2, th + pad_y * 2
    try:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    except Exception:
        draw.rectangle([x, y, x + w, y + h], fill=bg)
    draw.text((x + pad_x, y + pad_y - 2), text, font=font, fill=fill)
    return (x + w, y + h)


def _render_pick_card(canvas, draw, pick, y):
    """Render a single pick's compact card at vertical position y.
    Returns the y position AFTER the card (for stacking).
    Height is 170 so 9 cards fit on a 1920-tall slide."""
    card_h = 170
    pad = 36
    x0 = pad
    x1 = _W - pad

    is_dnb = bool(pick.get("do_not_bet"))
    is_whale = bool(pick.get("whale"))
    if is_dnb:
        border = _DNB
        border_width = 2
    elif is_whale:
        border = (56, 189, 248)  # cyan — WHALE pick stands out
        border_width = 3
    else:
        border = _RULE
        border_width = 1
    try:
        draw.rounded_rectangle([x0, y, x1, y + card_h], radius=14,
                               fill=_CARD, outline=border,
                               width=border_width)
    except Exception:
        draw.rectangle([x0, y, x1, y + card_h], fill=_CARD, outline=border)

    f_pick  = _font(30, bold=True)
    f_small = _font(20)
    f_chip  = _font(18, bold=True)
    f_mono  = _font(20)

    # Row 1: WHALE + LOCK + star + pick text + status chip (y+14)
    y_row = y + 14
    tx = x0 + 20
    is_whale = bool(pick.get("whale"))
    if is_whale:
        # Whale pill — bankroll-max tier. Filled cyan pill with white
        # text so it reads as a BADGE, not just a label. Stands out
        # against the dark card and the pick-text that follows.
        whale_text = "WHALE"
        try:
            bbox = draw.textbbox((0, 0), whale_text, font=f_chip)
            pw = bbox[2] - bbox[0] + 20
            ph = bbox[3] - bbox[1] + 10
        except Exception:
            pw, ph = 92, 30
        try:
            draw.rounded_rectangle(
                [tx, y_row - 4, tx + pw, y_row - 4 + ph],
                radius=14, fill=(56, 189, 248), outline=None)
        except Exception:
            draw.rectangle(
                [tx, y_row - 4, tx + pw, y_row - 4 + ph],
                fill=(56, 189, 248))
        draw.text((tx + 10, y_row - 2), whale_text, font=f_chip, fill=(10, 10, 15))
        tx += pw + 10
    if pick.get("signal_score") == 5:
        draw.text((tx, y_row + 4), "LOCK", font=f_chip, fill=_LOCK)
        tx += 72
    draw.text((tx, y_row + 2), "*", font=_font(28, bold=True), fill=_STAR)
    tx += 22
    pick_text = pick.get("pick") or ""
    if is_dnb:
        pick_text = f"[DO NOT BET] {pick_text}"
    _draw_text(draw, (tx, y_row), pick_text, f_pick,
               fill=(_DNB if is_dnb else _INK), max_w=x1 - tx - 160)

    # Right-aligned status chip
    chip_text = ""
    chip_bg = _CARD
    chip_fg = _INK
    if is_dnb:
        chip_text, chip_bg, chip_fg = "DO NOT BET", _DNB, _INK
    elif pick.get("result") == "W":
        chip_text, chip_bg, chip_fg = "WON", _GOOD, _INK
    elif pick.get("result") == "L":
        chip_text, chip_bg, chip_fg = "LOST", _BAD, _INK
    elif pick.get("result") == "P":
        chip_text, chip_bg, chip_fg = "PUSH", _MUTED_2, _INK
    elif pick.get("_row_status") == "live":
        chip_text, chip_bg, chip_fg = "LIVE", _WARN, _INK
    elif pick.get("_row_status") == "final":
        chip_text, chip_bg, chip_fg = "FINAL", _MUTED, _INK
    if chip_text:
        try:
            bbox = draw.textbbox((0, 0), chip_text, font=f_chip)
            cw = bbox[2] - bbox[0] + 24
            _draw_pill(draw, x1 - 20 - cw, y_row - 2, chip_text,
                       f_chip, chip_fg, chip_bg)
        except Exception:
            pass

    # Row 2: sport | matchup | market (y+58)
    y_row = y + 58
    sport = pick.get("sport_name") or pick.get("sport") or ""
    away  = pick.get("away_team") or pick.get("game_away") or ""
    home  = pick.get("home_team") or pick.get("game_home") or ""
    market = pick.get("market") or ""
    sub = f"{sport}  |  {away} @ {home}  |  {market}"
    _draw_text(draw, (x0 + 20, y_row), sub, f_small, fill=_MUTED,
               max_w=x1 - x0 - 40)

    # Row 3: Signal + EV pills + line move LEFT (y+95)
    y_row = y + 95
    tx = x0 + 20
    if pick.get("signal_score") is not None:
        rx, _b = _draw_pill(draw, tx, y_row,
                            f"Signal {pick['signal_score']}/5", f_chip,
                            fill=_SIGNAL_3, bg=(10, 30, 50))
        tx = rx + 8
    ev = pick.get("ev_pct")
    if ev is not None:
        try:
            ev_txt = f"+{float(ev):.1f}% EV"
            fg = _WARN if float(ev) > 15 else _GOOD
            rx, _b = _draw_pill(draw, tx, y_row, ev_txt, f_chip,
                                fill=fg, bg=(10, 40, 20))
            tx = rx + 8
        except (TypeError, ValueError):
            pass
    lm = pick.get("line_move_pp")
    if lm is not None and abs(lm) >= 0.001:
        arrow = "UP" if lm > 0 else "DN"
        _draw_text(draw, (tx, y_row + 4),
                   f"{arrow} {lm * 100:+.1f}pp", f_small,
                   fill=(_GOOD if lm > 0 else _BAD))

    # Row 4: kickoff time LEFT, odds RIGHT (stake suggestion removed
    # per user request — unit sizing stays as a /settings concept
    # rather than on every card).
    y_bottom = y + 138
    time_text = _format_pick_time(pick)
    _draw_text(draw, (x0 + 20, y_bottom), time_text, f_mono, fill=_MUTED)
    dec   = pick.get("decimal")
    book  = pick.get("book") or ""
    if dec:
        try:
            price_txt = f"@ {float(dec):.2f}  {book}"
            _draw_text(draw, (x1 - 20, y_bottom), price_txt,
                       f_mono, fill=_INK, anchor="rt")
        except (TypeError, ValueError):
            pass

    return y + card_h + 10


def render_picks_png(picks, title="Today's Picks", subtitle=""):
    """Render a batch of picks (max _PICKS_PER_PNG) to a 1080x1920 PNG.
    Returns bytes. Returns None if Pillow is unavailable.
    Layout: header 160px + 9 cards @ 180px (170 + 10 gap) = 1620px +
    footer 70px = 1850px total, fits in 1920."""
    if not _PIL_OK:
        return None
    canvas = Image.new("RGB", (_W, _H), _BG)
    draw = ImageDraw.Draw(canvas)

    # Header
    f_title = _font(56, bold=True)
    f_sub   = _font(24)
    draw.text((40, 32), title, font=f_title, fill=_INK)
    if subtitle:
        draw.text((40, 100), subtitle, font=f_sub, fill=_MUTED)

    # Divider
    draw.line([40, 148, _W - 40, 148], fill=_RULE, width=2)

    # Picks
    y = 168
    for p in picks[:_PICKS_PER_PNG]:
        y = _render_pick_card(canvas, draw, p, y)
        if y > _H - 80:
            break

    # Footer
    footer_y = _H - 50
    f_small = _font(18)
    draw.line([40, footer_y - 8, _W - 40, footer_y - 8],
              fill=_RULE, width=1)
    draw.text((40, footer_y), "cheeks-model.onrender.com",
              font=f_small, fill=_MUTED_2)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        draw.text((_W - 40, footer_y), stamp, font=f_small,
                  fill=_MUTED_2, anchor="rt")
    except Exception:
        draw.text((_W - 300, footer_y), stamp, font=f_small, fill=_MUTED_2)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# Rate-limit cooldown: when Discord (or Cloudflare in front of it)
# returns 429 we need to STOP sending to that webhook for a while or
# the limit window keeps renewing and we never recover. Shared across
# every send_* call via module-level state so a 429 on the slate PNG
# also stops the agenda text and vice versa. 15 min is generous for a
# typical Cloudflare 1015 window (usually clears in 1-10 min); if it's
# still 429 after that, next tick retries and extends again.
_RATE_LIMIT_COOLDOWN_UNTIL = 0.0
_RATE_LIMIT_COOLDOWN_SEC = 15 * 60


def _rate_limit_active():
    return time.time() < _RATE_LIMIT_COOLDOWN_UNTIL


def _trip_rate_limit(reason=""):
    global _RATE_LIMIT_COOLDOWN_UNTIL
    _RATE_LIMIT_COOLDOWN_UNTIL = time.time() + _RATE_LIMIT_COOLDOWN_SEC


def rate_limit_remaining_sec():
    """Expose for /debug/dispatch so operators can see the cooldown."""
    r = _RATE_LIMIT_COOLDOWN_UNTIL - time.time()
    return int(r) if r > 0 else 0


def send_discord_image(webhook_url, png_bytes, filename, message="",
                        return_id=False):
    """POST a PNG to a Discord webhook via multipart. Returns True on
    success (or the message_id when return_id=True — uses ?wait=true
    so Discord returns the created message so we can delete it later).
    Discord webhooks accept files up to ~25MB; our PNGs are typically
    ~80-120 KB.

    If a shared rate-limit cooldown is active (triggered when a prior
    call got HTTP 429), returns False without hitting the webhook so
    we don't extend the Cloudflare 1015 window."""
    if not webhook_url or not png_bytes:
        return False
    if _rate_limit_active():
        return False
    try:
        boundary = "----cheeksboundary" + str(int(time.time() * 1000))
        parts = []
        payload = {"content": message} if message else {}
        parts.append(f'--{boundary}\r\n'
                     f'Content-Disposition: form-data; name="payload_json"\r\n'
                     f'Content-Type: application/json\r\n\r\n'
                     f'{json.dumps(payload)}\r\n'.encode())
        parts.append(f'--{boundary}\r\n'
                     f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"\r\n'
                     f'Content-Type: image/png\r\n\r\n'.encode())
        parts.append(png_bytes)
        parts.append(f'\r\n--{boundary}--\r\n'.encode())
        body = b''.join(parts)
        url = webhook_url + ("?wait=true" if return_id else "")
        req = Request(url, data=body, method="POST", headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            # Discord webhooks reject requests without a User-Agent
            # with a 403; always include one.
            "User-Agent": "cheeks-model/1.0 (+https://cheeks-model.onrender.com)",
        })
        try:
            resp = urlopen(req, timeout=15)
            if return_id:
                try:
                    rb = json.loads(resp.read())
                    return rb.get("id")
                except Exception:
                    return True
                finally:
                    resp.close()
            else:
                resp.close()
                return True
        except HTTPError as _he:
            # HTTPError is raised on 4xx/5xx. Trip the cooldown on 429
            # so other calls in this process don't hammer the webhook.
            if getattr(_he, "code", 0) == 429:
                _trip_rate_limit("send_image_429")
            return False
    except Exception:
        return False


# ===========================================================
# Pick card PNG renderer (shipped 2026-10-24 v2)
# ===========================================================
# Renders a single pick as a ~1200x420 PNG — sport color sidebar,
# large pick label + odds, pills for Signal/Evidence/Tipoff/AI,
# chip row for firing signals, optional shop chip. Paired with the
# analysis text body in one Discord message so each pick lands as
# a clean visual card instead of a text wall.

_PICK_CARD_BG       = (43, 45, 49)      # Discord dark
_PICK_CARD_INK      = (255, 255, 255)
_PICK_CARD_INK_DIM  = (185, 187, 190)
_PICK_CARD_INK_FAINT = (120, 122, 128)
_PICK_CARD_PILL_BG  = (56, 58, 64)
_PICK_CARD_GOOD     = (87, 242, 135)
_PICK_CARD_LINE     = (60, 62, 68)

_PICK_CARD_SPORT_RGB = {
    "mlb":    (200, 16, 46),
    "nfl":    (1, 51, 105),
    "ncaaf":  (139, 34, 0),
    "nba":    (201, 8, 42),
    "nhl":    (0, 51, 160),
    "ufc":    (210, 10, 10),
    "epl":    (61, 25, 91),
    "laliga": (238, 135, 7),
    "ligamx": (0, 132, 61),
    "ucl":    (0, 30, 84),
    "europa": (255, 102, 0),
}


def _pc_font(size, bold=False):
    """Load a font — Segoe UI on Windows, DejaVu on Linux (Render)."""
    import sys as _sys
    if _sys.platform.startswith("win"):
        cands = ["C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
                 "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    else:
        cands = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                   else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for p in cands:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _pc_pill(draw, x, y, text, font, fg=_PICK_CARD_INK,
              bg=_PICK_CARD_PILL_BG, pad=(14, 7)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = tw + pad[0] * 2, th + pad[1] * 2
    try:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    except Exception:
        draw.rectangle([x, y, x + w, y + h], fill=bg)
    draw.text((x + pad[0], y + pad[1] - bbox[1]), text, font=font, fill=fg)
    return x + w


def _pc_fit_odds(draw, text, max_w=260, target_size=56, min_size=32):
    """Shrink odds font until it fits a max width. Returns (font, width)."""
    for size in (target_size, 48, 40, min_size):
        f = _pc_font(size, bold=True)
        try:
            w = draw.textbbox((0, 0), text, font=f)[2]
        except Exception:
            w = max_w
        if w <= max_w:
            return f, w
    f = _pc_font(min_size, bold=True)
    try:
        return f, draw.textbbox((0, 0), text, font=f)[2]
    except Exception:
        return f, max_w


def _pc_fit_pick_label(draw, text, max_w,
                        target_size=44, min_size=22):
    """Fit the pick label without truncation. Try shrinking the font
    through 44→40→36→32→28→24→22. If it STILL doesn't fit at 22pt,
    wrap into two lines at a natural word break. Never adds '...' —
    the line/total number must stay readable."""
    for size in (target_size, 40, 36, 32, 28, 24, min_size):
        f = _pc_font(size, bold=True)
        try:
            w = draw.textbbox((0, 0), text, font=f)[2]
        except Exception:
            w = max_w + 1
        if w <= max_w:
            return [text], f, size
    # Wrap into 2 lines on the latest word-break that fits
    f = _pc_font(min_size, bold=True)
    words = text.split()
    for i in range(len(words), 0, -1):
        head = " ".join(words[:i])
        try:
            if draw.textbbox((0, 0), head, font=f)[2] <= max_w:
                tail = " ".join(words[i:])
                return [head, tail], f, min_size
        except Exception:
            continue
    return [text], f, min_size


def render_pick_card_png(pick):
    """Return PNG bytes for one pick card. `pick` keys we read:
    sport_name, sport_slug, pick, decimal, signal_score, evidence_score,
    mins_to_tip, ai_verdict, best_book, best_price, best_price_delta_pct,
    reasons, start_time (ISO).

    v3 (2026-10-24): auto-shrinks + wraps the pick label so long lines
    like 'Over 237.5 (ALT · from main O/U 239.5)' or 'BTTS Yes & Over
    2.5 Goals' stay fully readable. Never truncates with '...'. The
    odds number also shrinks if the line gets very long."""
    if not _PIL_OK:
        return None
    W, H = 1200, 460
    canvas = Image.new("RGB", (W, H), _PICK_CARD_BG)
    draw   = ImageDraw.Draw(canvas, "RGBA")

    sport_slug = (pick.get("sport_slug") or "").lower()
    sidebar = _PICK_CARD_SPORT_RGB.get(sport_slug, (114, 137, 218))
    draw.rectangle([0, 0, 10, H], fill=sidebar)

    pad_x = 36
    y = 32
    sport = (pick.get("sport_name") or sport_slug or "?").upper()
    draw.text((pad_x, y), sport, font=_pc_font(20, bold=True),
               fill=_PICK_CARD_INK_DIM)
    y += 32

    # Odds FIRST (right-aligned) so we know how much width is left
    # for the pick label.
    dec = pick.get("decimal")
    odds_s = f"@ {dec:.2f}" if dec else "@ ?"
    f_odds, odds_w = _pc_fit_odds(draw, odds_s, max_w=260)

    # Pick label — fit to available width without truncation
    pick_label = pick.get("pick") or "?"
    max_w_label = W - pad_x - odds_w - 36  # 36px gap between label + odds
    lines, f_pick, pick_size = _pc_fit_pick_label(draw, pick_label,
                                                    max_w_label)
    line_h = pick_size + 6
    for i, line in enumerate(lines):
        draw.text((pad_x, y + i * line_h), line, font=f_pick,
                   fill=_PICK_CARD_INK)
    label_block_h = len(lines) * line_h
    # Odds vertically centered against the (possibly 2-line) label block
    odds_y = y + (label_block_h - pick_size) // 2 - 8
    draw.text((W - pad_x - odds_w, odds_y), odds_s, font=f_odds,
               fill=_PICK_CARD_GOOD)
    y += label_block_h + 20
    draw.rectangle([pad_x, y, W - pad_x, y + 1], fill=_PICK_CARD_LINE)
    y += 20

    # Pills row: Signal / Evidence / Tipoff / AI
    f_pill = _pc_font(22, bold=True)
    sig  = pick.get("signal_score")
    ev   = pick.get("evidence_score")
    mins = pick.get("mins_to_tip")
    px = pad_x
    sig_fg = _PICK_CARD_GOOD if (sig or 0) >= 4 else _PICK_CARD_INK
    px = _pc_pill(draw, px, y, f"Signal {sig}/5" if sig is not None else "Signal —",
                   f_pill, fg=sig_fg)
    px += 12
    px = _pc_pill(draw, px, y, f"Evidence {ev}/10" if ev is not None else "Evidence —",
                   f_pill, fg=_PICK_CARD_INK)
    px += 12
    # Relative tipoff
    if mins is not None:
        if mins < 60:
            tip_rel = f"Tip in {mins}m"
        else:
            tip_rel = f"Tip in {mins // 60}h{mins % 60}m"
    else:
        tip_rel = "Tipoff —"
    px = _pc_pill(draw, px, y, tip_rel, f_pill, fg=_PICK_CARD_INK_DIM)
    px += 12
    # Absolute tipoff time if start_time present
    st = pick.get("start_time") or ""
    tip_abs = ""
    if st:
        try:
            from datetime import datetime as _dt
            tz = None
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo("America/Chicago")
            except Exception:
                pass
            d = _dt.fromisoformat(st.replace("Z", "+00:00"))
            if tz is not None:
                d = d.astimezone(tz)
            tip_abs = d.strftime("%-I:%M %p CT") if sys_posix() else d.strftime("%#I:%M %p CT")
        except Exception:
            tip_abs = ""
    if tip_abs:
        px = _pc_pill(draw, px, y, tip_abs, f_pill, fg=_PICK_CARD_INK_DIM)
        px += 12
    # AI verdict
    aiv = pick.get("ai_verdict") or ""
    if aiv in ("Lean", "Strong Take"):
        ai_bg = (14, 165, 233, 60) if aiv == "Lean" else (34, 197, 94, 70)
        px = _pc_pill(draw, px, y, f"AI: {aiv}", f_pill,
                       fg=_PICK_CARD_INK, bg=ai_bg)
    y += 56

    # Signals-firing chip row
    reasons = pick.get("reasons") or []
    if reasons:
        f_chip = _pc_font(18)
        rx = pad_x
        for r in reasons:
            rx = _pc_pill(draw, rx, y, r, f_chip, fg=_PICK_CARD_INK_DIM,
                           bg=(45, 47, 52), pad=(10, 5))
            rx += 8
        y += 42

    # Shop chip
    bb = pick.get("best_book")
    bp = pick.get("best_price")
    bd = pick.get("best_price_delta_pct")
    if bb and bp:
        f_shop = _pc_font(20, bold=True)
        txt = f"$ Shop: {str(bb).upper()} {bp:.2f}"
        if bd:
            txt += f"  ({bd:+.1f}%)"
        _pc_pill(draw, pad_x, y, txt, f_shop,
                  fg=_PICK_CARD_GOOD, bg=(32, 60, 44))
        y += 42

    # Footer
    draw.text((pad_x, H - 28),
               "cheeks-model  ·  evidence-filtered pick",
               font=_pc_font(14), fill=_PICK_CARD_INK_FAINT)

    import io as _io
    buf = _io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def sys_posix():
    import sys as _s
    return not _s.platform.startswith("win")


def send_pick_card(webhook_url, pick, body_text, return_id=True):
    """Send a pick as PNG header + text analysis body in ONE Discord
    message. Returns the message_id on success (or True / False)."""
    png = render_pick_card_png(pick)
    if not png:
        # PIL missing — fall back to text-only
        return post_text(webhook_url, body_text, return_id=return_id)
    # Trim body if it would push the message over Discord's 2000-char
    # content cap.
    body = body_text or ""
    if len(body) > 1900:
        body = body[:1897] + "..."
    pid = (pick.get("id") or "pick").replace("/", "_")[:48]
    filename = f"{pid}.png"
    return send_discord_image(webhook_url, png, filename,
                                message=body, return_id=return_id)


def post_text(webhook_url, message, return_id=False):
    """POST a plain-text message to a Discord webhook. Returns True on
    success (or the message_id when return_id=True — uses ?wait=true
    to force Discord to return the created message so we can delete
    it later). Respects the shared 429 cooldown."""
    if not webhook_url or not message:
        return False
    if _rate_limit_active():
        return False
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError as _HE
        data = json.dumps({"content": message}).encode()
        url = webhook_url + ("?wait=true" if return_id else "")
        req = Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "User-Agent": "cheeks-model/1.0 (+https://cheeks-model.onrender.com)",
        })
        try:
            resp = urlopen(req, timeout=15)
            if return_id:
                try:
                    body = json.loads(resp.read())
                    return body.get("id")
                except Exception:
                    return True
                finally:
                    resp.close()
            else:
                resp.close()
                return True
        except _HE as _he:
            if getattr(_he, "code", 0) == 429:
                _trip_rate_limit("post_text_429")
            return False
    except Exception:
        return False


# Sport color sidebar for Discord embeds — picks a distinct color so
# viewers can tell at a glance which league the pick is for.
_SPORT_EMBED_COLORS = {
    "mlb":    0xC8102E,   # red
    "nfl":    0x013369,   # NFL blue
    "ncaaf":  0x8B2200,   # college maroon
    "nba":    0xC9082A,   # NBA red
    "nhl":    0x0033A0,   # NHL blue
    "ufc":    0xD20A0A,   # UFC red
    "epl":    0x3D195B,   # EPL purple
    "laliga": 0xEE8707,   # La Liga orange
    "ligamx": 0x00843D,   # Liga MX green
    "ucl":    0x001E54,   # Champions League dark blue
    "europa": 0xFF6600,   # Europa League orange
}


def post_embed(webhook_url, embed, content="", return_id=True):
    """POST a Discord embed (rich formatted card with color sidebar
    + fields). Returns the message_id on success so the caller can
    store it for later editing or deletion.

    `embed` is a dict shaped per Discord's embed object spec:
    https://discord.com/developers/docs/resources/channel#embed-object

    Example:
      post_embed(url, {
        "title":  "LIGA MX · BTTS Yes @ 1.71",
        "color":  0x00843D,
        "fields": [
          {"name": "Signal / Evidence", "value": "5/5 · 4/10", "inline": True},
          {"name": "AI verdict", "value": "Lean", "inline": True},
          {"name": "Matchup", "value": "...", "inline": False},
        ],
        "footer": {"text": "cheeks-model"},
      })
    """
    if not webhook_url or not embed:
        return False
    if _rate_limit_active():
        return False
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError as _HE
        payload = {"embeds": [embed]}
        if content:
            payload["content"] = content
        data = json.dumps(payload).encode()
        url = webhook_url + ("?wait=true" if return_id else "")
        req = Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json",
            "User-Agent": "cheeks-model/1.0 (+https://cheeks-model.onrender.com)",
        })
        try:
            resp = urlopen(req, timeout=15)
            if return_id:
                try:
                    body = json.loads(resp.read())
                    return body.get("id")
                except Exception:
                    return True
                finally:
                    resp.close()
            else:
                resp.close()
                return True
        except _HE as _he:
            if getattr(_he, "code", 0) == 429:
                _trip_rate_limit("post_embed_429")
            return False
    except Exception:
        return False


def delete_webhook_message(webhook_url, message_id):
    """Delete a message the webhook sent. Only works for messages
    whose IDs we previously captured (via post_text(return_id=True)
    or post_embed). Returns True on success."""
    if not webhook_url or not message_id:
        return False
    try:
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError as _HE
        url = webhook_url.rstrip("/") + f"/messages/{message_id}"
        req = Request(url, method="DELETE", headers={
            "User-Agent": "cheeks-model/1.0 (+https://cheeks-model.onrender.com)",
        })
        try:
            urlopen(req, timeout=15).close()
            return True
        except _HE as _he:
            # 404 = already deleted (fine), 429 = rate-limited
            if getattr(_he, "code", 0) == 429:
                _trip_rate_limit("delete_message_429")
            return False
    except Exception:
        return False


def sport_embed_color(sport_slug):
    """Return the embed sidebar color for a sport slug (hex int)."""
    return _SPORT_EMBED_COLORS.get((sport_slug or "").lower(), 0x7289da)


_INTRO_MESSAGE = (
    "**Cheeks Model — picks feed**\n\n"
    "Automated feed of the day's highest-conviction bets across NFL, MLB, "
    "NHL, NBA, soccer, UFC and more.\n\n"
    "**How it works now**\n"
    "- Picks post **individually**, timed to each sport's pre-game window "
    "(when lineups + injuries lock in) — no slate blast.\n"
    "- Only picks with an **Evidence Score ≥ 5** make it here. Score is "
    "built from Signal 4+ composite, favorable line move, Kalshi agreement, "
    "whale alignment, clean injury news, and AI endorsement.\n"
    "- Each post includes the matchup read, key factors, risk flags, and "
    "the best US-book price if DK / FD / MGM / Caesars beats Pinnacle.\n"
    "- Expected volume: **1–6 picks per day**. Quiet days are fine — only "
    "the strongest signals get through.\n\n"
    "Full dashboard: https://cheeks-model.onrender.com"
)


def send_intro(webhook_url):
    """Post the one-time welcome text. Call this once per channel so
    new viewers know what the feed does. Pin it in Discord for best
    results."""
    return post_text(webhook_url, _INTRO_MESSAGE)


# ------------------------------------------------------------------
# Channel wipe + re-intro (Phase 32).
# ------------------------------------------------------------------
# Discord webhooks CANNOT list or delete arbitrary channel history —
# only a Bot token can. We derive the channel_id from the webhook via
# `GET /webhooks/{id}/{token}` (no bot token needed for that), then use
# the bot token's REST permissions to bulk-delete recent messages and
# individually delete messages older than 14 days.
#
# If no bot token is provided we skip the wipe step and just post the
# fresh intro. The caller gets a result dict that explains what ran.
# ------------------------------------------------------------------

def _discord_api(method, path, bot_token=None, body=None, timeout=15):
    """Thin wrapper around the Discord REST API. Returns the parsed
    JSON on 2xx, or raises on error. `path` is the API path starting
    with '/'. `bot_token` is required for anything beyond webhook
    endpoints; webhook calls can pass None."""
    from urllib.request import Request, urlopen
    url = "https://discord.com/api/v10" + path
    headers = {
        "User-Agent": "cheeks-model/1.0 (+https://cheeks-model.onrender.com)",
        "Content-Type": "application/json",
    }
    if bot_token:
        headers["Authorization"] = f"Bot {bot_token}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}


def _channel_id_from_webhook(webhook_url):
    """Hit the webhook's own metadata endpoint to recover its
    channel_id. Returns None if the webhook URL is malformed or the
    call fails."""
    try:
        # Webhook URL: https://discord.com/api/webhooks/{id}/{token}
        tail = webhook_url.rstrip("/").split("/api/webhooks/", 1)[-1]
        wid, token = tail.split("/", 1)
    except Exception:
        return None
    try:
        meta = _discord_api("GET", f"/webhooks/{wid}/{token}")
        return meta.get("channel_id")
    except Exception:
        return None


def clear_channel_history(bot_token, channel_id, max_batches=20):
    """Delete every message in the channel. Uses bulk-delete for
    messages <14d old, individual DELETE for older. Returns a dict
    with counts.

    `max_batches` caps the number of 100-message pages we walk — each
    batch is one list + one bulk-delete call, so 20 batches = 2000
    messages max. Discord bot rate-limits are generous for this but
    we still cap it to avoid runaway loops."""
    if not bot_token or not channel_id:
        return {"ok": False, "error": "bot_token + channel_id required"}
    total_bulk = 0
    total_single = 0
    now_ms = int(time.time() * 1000)
    # Discord bulk-delete window is 14 days.
    cutoff_ms = now_ms - (14 * 24 * 60 * 60 * 1000)
    for _ in range(max_batches):
        try:
            msgs = _discord_api("GET",
                                 f"/channels/{channel_id}/messages?limit=100",
                                 bot_token=bot_token)
        except Exception as e:
            return {"ok": False, "error": f"list failed: {e}",
                    "deleted_bulk": total_bulk, "deleted_single": total_single}
        if not msgs:
            break
        recent = []
        old = []
        for m in msgs:
            mid = m.get("id")
            if not mid:
                continue
            # Discord IDs are snowflakes — timestamp is upper 42 bits
            # (shifted by 22) + Discord epoch (2015-01-01 00:00:00 UTC).
            try:
                ts_ms = (int(mid) >> 22) + 1420070400000
            except Exception:
                ts_ms = now_ms
            if ts_ms >= cutoff_ms:
                recent.append(mid)
            else:
                old.append(mid)
        # Bulk-delete the recent batch (needs 2-100 items; a single
        # recent message falls through to single-delete).
        if len(recent) >= 2:
            try:
                _discord_api("POST",
                              f"/channels/{channel_id}/messages/bulk-delete",
                              bot_token=bot_token,
                              body={"messages": recent})
                total_bulk += len(recent)
            except Exception:
                for mid in recent:
                    try:
                        _discord_api("DELETE",
                                      f"/channels/{channel_id}/messages/{mid}",
                                      bot_token=bot_token)
                        total_single += 1
                    except Exception:
                        continue
                    time.sleep(0.3)  # polite pacing on fallback path
        elif len(recent) == 1:
            try:
                _discord_api("DELETE",
                              f"/channels/{channel_id}/messages/{recent[0]}",
                              bot_token=bot_token)
                total_single += 1
            except Exception:
                pass
        # Individually delete old messages (bulk-delete rejects
        # anything > 14 days).
        for mid in old:
            try:
                _discord_api("DELETE",
                              f"/channels/{channel_id}/messages/{mid}",
                              bot_token=bot_token)
                total_single += 1
            except Exception:
                continue
            time.sleep(0.3)
        if len(msgs) < 100:
            break
    return {"ok": True,
            "deleted_bulk": total_bulk,
            "deleted_single": total_single,
            "total_deleted": total_bulk + total_single}


def delete_duplicate_messages(bot_token, channel_id, scan_limit=100,
                                content_prefix_chars=120, dry_run=False):
    """Scan the channel's recent history, group by content prefix,
    delete every duplicate but the OLDEST message in each group.

    Discord webhook messages are considered duplicates when their
    first `content_prefix_chars` characters match exactly — same pick
    posted twice ends up with the same header (sport, pick, price,
    evidence breakdown), so a prefix compare catches dupes without
    false-matching unrelated posts whose later bullets differ.

    Requires a Bot token with Manage Messages in the channel; without
    one this is a no-op. Returns a result dict with per-group counts.

    `dry_run=True` reports what WOULD be deleted without hitting
    Discord — useful for the first run after enabling the sweep."""
    if not bot_token or not channel_id:
        return {"ok": False, "error": "bot_token + channel_id required",
                "dry_run": dry_run}
    try:
        msgs = _discord_api("GET",
                             f"/channels/{channel_id}/messages?limit={min(100, scan_limit)}",
                             bot_token=bot_token)
    except Exception as e:
        return {"ok": False, "error": f"list failed: {e}", "dry_run": dry_run}

    groups = {}  # content_prefix → list[(ts_ms, msg_id)]
    for m in msgs or []:
        mid = m.get("id")
        content = (m.get("content") or "")
        if not mid or not content:
            continue
        try:
            ts_ms = (int(mid) >> 22) + 1420070400000
        except Exception:
            continue
        key = content[:content_prefix_chars].strip()
        groups.setdefault(key, []).append((ts_ms, mid))

    deleted = []
    kept = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        # Keep the OLDEST (smallest ts) — that's the "first" post
        items.sort(key=lambda t: t[0])
        kept.append({"preview": key[:60], "message_id": items[0][1]})
        for ts, mid in items[1:]:
            if dry_run:
                deleted.append({"preview": key[:60], "message_id": mid,
                                 "would_delete": True})
                continue
            try:
                _discord_api("DELETE",
                              f"/channels/{channel_id}/messages/{mid}",
                              bot_token=bot_token)
                deleted.append({"preview": key[:60], "message_id": mid})
            except Exception as e:
                deleted.append({"preview": key[:60], "message_id": mid,
                                 "error": str(e)})
            time.sleep(0.25)  # polite pacing

    return {"ok": True, "dry_run": dry_run,
            "scanned_count":  len(msgs or []),
            "dup_groups":     len([g for g in groups.values() if len(g) >= 2]),
            "deleted_count":  len([d for d in deleted if not d.get("error")]),
            "deleted":        deleted,
            "kept":           kept}


def reset_channel_and_reintro(webhook_url, bot_token=None):
    """Full reset: wipe channel history (if bot_token supplied), then
    post the fresh intro via the webhook. Returns a result dict the
    caller can hand straight back as JSON."""
    out = {"ok": False, "cleared": None, "intro_posted": False,
           "notes": []}
    if not webhook_url:
        out["notes"].append("No webhook URL configured.")
        return out
    channel_id = _channel_id_from_webhook(webhook_url)
    if not channel_id:
        out["notes"].append("Could not resolve channel_id from webhook URL.")
    # 1. Clear
    if bot_token and channel_id:
        try:
            out["cleared"] = clear_channel_history(bot_token, channel_id)
        except Exception as e:
            out["cleared"] = {"ok": False, "error": str(e)}
    else:
        out["notes"].append(
            "No bot token — skipping channel wipe. Add a Discord bot "
            "with MANAGE_MESSAGES in that channel to enable auto-wipe."
        )
    # 2. Post fresh intro
    out["intro_posted"] = send_intro(webhook_url)
    out["ok"] = out["intro_posted"] and (bot_token is None or
                                          (out["cleared"] or {}).get("ok", False))
    return out


def render_picks_png_tall(picks, title, subtitle=""):
    """Variable-height PNG — one card per pick, no clipping. Used for
    the daily top-N send and the live-update entrants PNG so the whole
    list fits in a single image rather than being split."""
    if not _PIL_OK:
        return None
    n = max(1, min(len(picks), _TOP_N))
    card_h = 170
    gap = 10
    header_h = 160
    footer_h = 70
    h = header_h + (n * card_h) + ((n - 1) * gap) + footer_h
    canvas = Image.new("RGB", (_W, h), _BG)
    draw = ImageDraw.Draw(canvas)

    f_title = _font(56, bold=True)
    f_sub   = _font(24)
    draw.text((40, 32), title, font=f_title, fill=_INK)
    if subtitle:
        draw.text((40, 100), subtitle, font=f_sub, fill=_MUTED)
    draw.line([40, 148, _W - 40, 148], fill=_RULE, width=2)

    y = 168
    for p in picks[:n]:
        y = _render_pick_card(canvas, draw, p, y)

    footer_y = h - 50
    f_small = _font(18)
    draw.line([40, footer_y - 8, _W - 40, footer_y - 8],
              fill=_RULE, width=1)
    draw.text((40, footer_y), "cheeks-model.onrender.com",
              font=f_small, fill=_MUTED_2)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        draw.text((_W - 40, footer_y), stamp, font=f_small,
                  fill=_MUTED_2, anchor="rt")
    except Exception:
        draw.text((_W - 300, footer_y), stamp, font=f_small, fill=_MUTED_2)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _safe_float(v):
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _risk_penalty(pick):
    """Return a soft penalty (added to the negative-signal sort key)
    to downrank picks with weather or injury risk flags. Doesn't
    exclude — just pushes them below equivalent-strength clean picks
    so they lose the top-5 battle to safer alternatives."""
    rf = pick.get("risk_flags") or {}
    penalty = 0.0
    w = rf.get("weather")
    if w == "high":
        penalty += 1.0   # equivalent to -1 Signal point
    elif w == "moderate":
        penalty += 0.4
    if rf.get("injury"):
        penalty += 0.6
    return penalty


def _rank_picks(picks):
    """Sort live (not do_not_bet) picks strongest-first for top-N.
    Tiebreakers: effective Signal (Signal - risk penalty) desc →
    line_move_pp desc (CLV favoring our side ranks higher) → EV desc →
    earliest start first."""
    live = [p for p in picks if not p.get("do_not_bet")]
    live.sort(key=lambda p: (
        -(int(p.get("signal_score") or 0)) + _risk_penalty(p),
        -_safe_float(p.get("line_move_pp")),
        -_safe_float(p.get("ev_pct")),
        p.get("start_time") or "",
    ))
    return live


def _market_bucket(market):
    """Rough market-type grouping used by diversification. Keeps the
    top-5 from being 5 spreads or 5 totals."""
    m = (market or "").lower()
    if "ml" in m or "moneyline" in m or "dnb" in m or "draw" in m:
        return "ML"
    if "spread" in m or "run line" in m or "puck line" in m or "alt" in m or "handicap" in m:
        return "SPREAD"
    if "team total" in m or "tt " in m or m.startswith("tt"):
        return "TT"
    if "total" in m or " over " in m or " under " in m or m.startswith("over") or m.startswith("under"):
        return "TOTAL"
    if "btts" in m or "both teams" in m:
        return "BTTS"
    if "correct" in m:
        return "CS"
    if "distance" in m:
        return "DIST"
    if "1h" in m or "first half" in m or "first-half" in m or "half time" in m:
        return "1H"
    return "OTHER"


def _diversify_by_market(ranked_picks, limit):
    """Round-robin across market buckets so no single market dominates
    the top-N. Preserves strongest-first within each bucket. If a
    bucket runs dry we just keep pulling from the strongest remaining."""
    buckets = {}
    for p in ranked_picks:
        b = _market_bucket(p.get("market"))
        buckets.setdefault(b, []).append(p)
    # Order buckets by the strongest pick inside each (so ML with a
    # Signal 5 outranks TOTAL with a Signal 3 on the first pass).
    def _bucket_rank(b):
        p = buckets[b][0]
        return (-(int(p.get("signal_score") or 0)), -_safe_float(p.get("ev_pct")))
    out = []
    while buckets and len(out) < limit:
        for b in sorted(buckets.keys(), key=_bucket_rank):
            if not buckets[b]:
                continue
            out.append(buckets[b].pop(0))
            if len(out) >= limit:
                break
        buckets = {b: lst for b, lst in buckets.items() if lst}
    return out


def _pick_start_ct(pick):
    """Return the pick's game-start datetime in Central Time, or None
    if we can't parse it."""
    from zoneinfo import ZoneInfo
    central = ZoneInfo("America/Chicago")
    st = pick.get("start_time")
    if not st:
        return None
    try:
        if st.endswith("Z"):
            st = st[:-1] + "+00:00"
        dt = datetime.fromisoformat(st)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(central)
    except Exception:
        return None


def _pick_hour_ct(pick):
    """Hour (0-23) in Central Time, or None if unparseable."""
    dt = _pick_start_ct(pick)
    return dt.hour if dt is not None else None


def _format_pick_time(pick):
    """Return a clear kickoff string like '7:30 PM CT' (plus
    countdown if present). Falls back to parsing start_time when
    time_abs is blank, and always suffixes CT so the TZ is obvious."""
    time_abs  = (pick.get("time_abs") or "").strip()
    countdown = (pick.get("countdown") or "").strip()
    # Prefer a parsed start_time so we always render the TZ.
    dt = _pick_start_ct(pick)
    if dt is not None:
        base = dt.strftime("%I:%M %p CT").lstrip("0")
    elif time_abs:
        # No parseable start_time — use what the UI gave us and tack
        # on CT if it doesn't already carry a TZ.
        has_tz = any(x in time_abs for x in ("CT", "ET", "PT", "UTC", "GMT"))
        base = time_abs if has_tz else f"{time_abs} CT"
    else:
        return "Time TBD"
    if countdown:
        return f"{base}  ·  {countdown}"
    return base


def _filter_half(picks, half, cutoff_hour=_HALF_CUTOFF_HOUR):
    """LEGACY. New code uses _filter_slate."""
    out = []
    for p in picks:
        h = _pick_hour_ct(p)
        if h is None:
            if half == 2:
                out.append(p)
            continue
        if half == 1 and h < cutoff_hour:
            out.append(p)
        elif half == 2 and h >= cutoff_hour:
            out.append(p)
    return out


def _filter_slate(picks, slate):
    """slate ∈ {1, 2, 3}: filter picks whose game starts in that
    slate's time window (CT). Picks without a parseable start_time
    fall into Prime Time (slate 3)."""
    start_h, end_h = _SLATE_WINDOWS[slate]
    out = []
    for p in picks:
        h = _pick_hour_ct(p)
        if h is None:
            if slate == 3:
                out.append(p)
            continue
        if start_h <= h < end_h:
            out.append(p)
    return out


def send_slate_picks(webhook_url, picks, date_label="", slate=1):
    """Post one tall PNG with the top-N (Signal >= 4) picks for the
    given slate (1 Morning / 2 Afternoon / 3 Prime Time).

    Returns {"sent": 0|1, "pick_ids": [...], "reason": "..."} so the
    caller can tell the difference between "no picks were in this
    slate's time window" (should NOT retry) and "post to Discord
    failed" (SHOULD retry on next tick)."""
    if not webhook_url or not picks:
        return {"sent": 0, "pick_ids": [], "reason": "no_webhook_or_no_board"}
    label = date_label or datetime.utcnow().strftime("%A, %B %d")
    slate_name = _SLATE_LABELS.get(slate, "Slate")
    emoji = _SLATE_EMOJI.get(slate, ":star:")

    pool_in_window = _filter_slate(picks, slate)
    # Morning slate special case: viewers want an early-day preview.
    # If no games start before 12 PM CT, don't go dark — fall back to
    # the full day's top-5 so there's always a morning drop. The
    # preamble below detects this via the `morning_fallback` flag and
    # relabels the window. For slates 2 and 3, an empty in-window pool
    # still returns the "no_picks_in_slate_window" terminal reason
    # since by then the morning slate already showed the day's top-5.
    morning_fallback = False
    if not pool_in_window:
        if slate == 1:
            pool_in_window = list(picks)
            morning_fallback = True
        else:
            return {"sent": 0, "pick_ids": [],
                    "reason": "no_picks_in_slate_window"}
    # Signal 3.5 gate — Signal 4+ always qualifies, Signal 3 only
    # with EV >= _SIGNAL_3_MIN_EV. _rank_picks drops DNB separately.
    pool = [p for p in pool_in_window if _qualifies_for_discord(p)]
    ranked = _rank_picks(pool)
    # Correlated-pick avoidance: at most 1 pick per game. If multiple
    # picks on the same matchup qualify, keep only the strongest.
    deduped = []
    seen_games = set()
    for p in ranked:
        key = (
            (p.get("sport_slug") or "").lower(),
            (p.get("home_team") or p.get("game_home") or "").lower(),
            (p.get("away_team") or p.get("game_away") or "").lower(),
        )
        if key in seen_games:
            continue
        seen_games.add(key)
        deduped.append(p)
    # Market-type diversification: don't let the top-5 be 5 spreads.
    # Round-robin across market buckets (ML / SPREAD / TOTAL / etc.)
    # Keeps variance across market types so one market's cold streak
    # can't cost the whole slate.
    top = _diversify_by_market(deduped, _TOP_N)
    if not top:
        # Stay quiet if nothing clears the Signal >= 4 bar — better than
        # posting an empty slide or dropping to Signal 3 noise. Caller
        # should treat this as "nothing to send", not a transient
        # failure to retry.
        return {"sent": 0, "pick_ids": [],
                "reason": "nothing_cleared_signal_3_5_gate"}

    hours = sorted([h for h in (_pick_hour_ct(p) for p in top) if h is not None])
    if hours:
        h = hours[0]
        earliest = f"{h % 12 or 12}:00 {'AM' if h < 12 else 'PM'} CT"
    else:
        earliest = "later today"

    start_h, end_h = _SLATE_WINDOWS[slate]
    window_desc = {
        1: "games starting before 12 PM CT",
        2: "games starting 12 PM - 6 PM CT",
        3: "games starting 6 PM CT or later",
    }[slate]
    if morning_fallback:
        window_desc = ("today's full board — no games before 12 PM CT, "
                       "so this morning's drop is the full day's top-" + str(_TOP_N))
    whale_count = sum(1 for p in top if p.get("whale"))
    whale_line = ""
    if whale_count:
        whale_line = (
            f"\n:whale: **{whale_count} whale pick"
            f"{'s' if whale_count != 1 else ''}** today — "
            "bankroll-max tier (short-price, high-prob, confirmed Signal). "
            "Flagged with a cyan WHALE pill in the slide."
        )
    # Count picks where our model prob diverges >= 10pp from Pinnacle
    # devigged. These often indicate a model artifact, a stale market
    # price, or late-breaking info Pinnacle has baked in but we don't.
    # Not a disqualification — just a warning you can scan for before
    # placing.
    def _is_outlier(p):
        try:
            fp = float(p.get("fair_prob") or 0.0)
            pp = float(p.get("pinnacle_prob") or 0.0)
        except (TypeError, ValueError):
            return False
        if fp <= 0 or pp <= 0:
            return False
        return abs(fp - pp) * 100 >= 10.0
    outlier_count = sum(1 for p in top if _is_outlier(p))
    outlier_line = ""
    if outlier_count:
        outlier_line = (
            f"\n:warning: **{outlier_count} model-outlier pick"
            f"{'s' if outlier_count != 1 else ''}** — model and Pinnacle "
            "disagree by >= 10pp on these. Could be a model artifact or "
            "info Pinnacle has that we don't. Verify the current line."
        )
    preamble = (
        f"{emoji} **{slate_name} picks — {label}**\n\n"
        f"Top {len(top)} Signal 3.5+ picks for {window_desc} "
        f"(earliest kickoff ~{earliest}). "
        "This list is **locked for the day** — only hard DO NOT BET "
        "reversals or final results will post here until the next slate."
        + whale_line
        + outlier_line
    )
    # If a prior slate posted earlier today and some of its picks have
    # already graded, surface the running record as context.
    prior_recap = _prior_slate_recap(slate)
    if prior_recap:
        preamble += f"\n\n{prior_recap}"
    post_text(webhook_url, preamble)

    png = render_picks_png_tall(
        top,
        title=f"{slate_name} — Top {len(top)}",
        subtitle=label,
    )
    if not png:
        return {"sent": 0, "pick_ids": [], "reason": "png_render_failed"}
    posted_ids = [p.get("id") for p in top if p.get("id")]
    ok = send_discord_image(webhook_url, png, f"slate{slate}.png")
    if ok:
        return {"sent": 1, "pick_ids": posted_ids}
    # The actual HTTP POST to Discord failed (401/404/429/network).
    # Caller should retry this on the next tick.
    return {"sent": 0, "pick_ids": posted_ids,
            "reason": "discord_post_failed"}


def diff_posted_hard_dnb(star_picks, posted_slate_ids):
    """Return [(pick, slate_label), ...] for POSTED picks whose signal
    has collapsed to <= _HARD_DNB_SIGNAL (true market reversal, not
    marginal drift). Deduped via logs/discord_dnb_sent.json so each
    pick fires at most once.

    posted_slate_ids: {slate_num: [ids]} dict."""
    sent = _load_dnb_sent()
    pick_by_id = {p.get("id"): p for p in star_picks}
    out = []
    for slate_num, ids in (posted_slate_ids or {}).items():
        label = _SLATE_LABELS.get(slate_num, "Slate")
        for pid in ids:
            p = pick_by_id.get(pid)
            if not p or pid in sent:
                continue
            # Fire only on genuine reversal: either explicit DNB flag
            # OR signal dropped very low. Keeps edge-of-cutoff drift
            # from spamming (Signal 4 -> 3 no longer alerts).
            sig = int(p.get("signal_score") or 0)
            if p.get("do_not_bet") or sig <= _HARD_DNB_SIGNAL:
                out.append((p, label))
                sent.add(pid)
    if out:
        _write_dnb_sent(sent)
    return out


def send_eod_summary(webhook_url, picks_by_slate, date_label=""):
    """End-of-day summary. picks_by_slate = {slate_num: [picks]} where
    each pick has been graded (result + profit_u). Shows the WHOLE
    DAY's record across every slate, paginated into 1-3 PNGs (up to
    _EOD_PER_PNG cards per image, _EOD_MAX_PNGS images total).

    Pagination: wins first (strongest signal first), then losses
    (worst first — biggest units bled), then pushes/pending. Keeps
    the top-of-list visually dense with the positives, which is what
    the viewer most wants to see at EOD.
    """
    if not webhook_url or not picks_by_slate:
        return 0
    label = date_label or datetime.utcnow().strftime("%A, %B %d")

    # Flatten + dedup by id in case the same pick appears on multiple
    # slate pick_id lists (shouldn't happen, defensive).
    #
    # Skip picks flagged do_not_bet. These are picks that WERE posted to
    # Discord earlier but have since been retracted (signal collapse,
    # decimal < 1.50 after move, AI Avoid verdict, line move adverse,
    # etc.). The user already got a DNB alert for them, so including
    # them in the day summary would double-count: once as a loss/win in
    # the record and once as "wait, you told me not to bet that."
    # Shipped 2026-10-22 after a Liverpool 1H Under flip-to-DNB got
    # shown as a win in the EOD summary even though it was retracted.
    all_picks = []
    seen = set()
    per_slate_rec = {}
    for s, picks in picks_by_slate.items():
        if not picks:
            continue
        w = l = pu = pend_s = 0
        u = 0.0
        for p in picks:
            pid = p.get("id")
            if pid and pid in seen:
                continue
            if p.get("do_not_bet"):
                continue
            if pid:
                seen.add(pid)
            all_picks.append(p)
            r = p.get("result")
            if r == "W":
                w += 1
            elif r == "L":
                l += 1
            elif r == "P":
                pu += 1
            else:
                pend_s += 1
            try:
                u += float(p.get("profit_u") or 0.0)
            except (TypeError, ValueError):
                pass
        per_slate_rec[s] = (w, l, pu, u, pend_s)

    if not all_picks:
        return 0

    tw = sum(1 for p in all_picks if p.get("result") == "W")
    tl = sum(1 for p in all_picks if p.get("result") == "L")
    tp = sum(1 for p in all_picks if p.get("result") == "P")
    tu = 0.0
    for p in all_picks:
        try:
            tu += float(p.get("profit_u") or 0.0)
        except (TypeError, ValueError):
            pass
    settled = tw + tl + tp
    pend = len(all_picks) - settled
    hit = f"{(tw / settled * 100):.0f}%" if settled else "n/a"

    # Preamble text (unchanged shape — just the headline + per-slate
    # breakdown)
    lines = [
        f":bar_chart: **Day summary — {label}**",
        "",
        f"Total: **{tw}-{tl}-{tp}**  ·  {hit} hit on {settled} settled  ·  **{tu:+.2f}u**",
    ]
    if pend:
        lines.append(f"_({pend} still pending — will update on re-run.)_")
    lines.append("")
    for s, (w, l, pu, u, pending) in sorted(per_slate_rec.items()):
        sname = _SLATE_LABELS.get(s, f"Slate {s}")
        emoji = _SLATE_EMOJI.get(s, ":star:")
        line = f"{emoji} **{sname}:** {w}-{l}-{pu}  ·  {u:+.2f}u"
        if pending:
            line += f"  ·  _{pending} pending_"
        lines.append(line)
    post_text(webhook_url, "\n".join(lines))

    # Pagination.
    # Order: wins (signal desc) → losses (|profit_u| desc, biggest
    # bleeders first) → pushes → pending.
    def _profit(p):
        try:
            return float(p.get("profit_u") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    wins    = [p for p in all_picks if p.get("result") == "W"]
    losses  = [p for p in all_picks if p.get("result") == "L"]
    pushes  = [p for p in all_picks if p.get("result") == "P"]
    pending_picks = [p for p in all_picks
                     if p.get("result") not in ("W", "L", "P")]
    wins.sort(key=lambda p: -(int(p.get("signal_score") or 0)))
    losses.sort(key=_profit)                        # most-negative first
    pushes.sort(key=lambda p: -(int(p.get("signal_score") or 0)))
    pending_picks.sort(key=lambda p: -(int(p.get("signal_score") or 0)))
    ordered = wins + losses + pushes + pending_picks

    PER_PNG = _EOD_PER_PNG
    MAX_PNGS = _EOD_MAX_PNGS
    total_pages = max(1, min(MAX_PNGS, (len(ordered) + PER_PNG - 1) // PER_PNG))
    sent_count = 0
    for page_idx in range(total_pages):
        chunk = ordered[page_idx * PER_PNG:(page_idx + 1) * PER_PNG]
        if not chunk:
            break
        if total_pages > 1:
            title = f"Day Summary — {tw}-{tl}-{tp}  ({page_idx + 1}/{total_pages})"
        else:
            title = f"Day Summary — {tw}-{tl}-{tp}"
        subtitle = f"{label}  ·  {tu:+.2f}u"
        png = render_picks_png_tall(chunk, title=title, subtitle=subtitle)
        if not png:
            continue
        fname = f"eod_summary_{page_idx + 1}.png" if total_pages > 1 else "eod_summary.png"
        if send_discord_image(webhook_url, png, fname):
            sent_count += 1
    return sent_count


# Keep send_daily_picks + diff_posted_dnb as aliases for backward
# compatibility with any callers that haven't migrated yet.
def send_daily_picks(webhook_url, picks, date_label="", half=None):
    """DEPRECATED — maps half -> slate (half 1 → slate 1, half 2 → merged
    slate 2+3). Prefer send_slate_picks."""
    if half == 1:
        r = send_slate_picks(webhook_url, picks, date_label, slate=1)
        return r.get("sent", 0)
    if half == 2:
        # Legacy half 2 = afternoon + prime time; pick whichever has picks.
        for s in (2, 3):
            r = send_slate_picks(webhook_url, picks, date_label, slate=s)
            if r.get("sent"):
                return r["sent"]
        return 0
    r = send_slate_picks(webhook_url, picks, date_label, slate=1)
    return r.get("sent", 0)


def _load_last_alert_ts():
    try:
        with open(_LAST_ALERT_PATH) as f:
            return float(json.load(f).get("ts", 0))
    except Exception:
        return 0.0


def _write_last_alert_ts(ts):
    try:
        os.makedirs(os.path.dirname(_LAST_ALERT_PATH), exist_ok=True)
        with open(_LAST_ALERT_PATH, "w") as f:
            json.dump({"ts": ts}, f)
    except Exception:
        pass


def diff_top_n(star_picks):
    """DEPRECATED — kept so legacy callers don't break. Returns [] and
    just refreshes the baseline snapshot. New entry-detection goes
    through diff_candidates (scoped against posted top-10s)."""
    ranked = _rank_picks(star_picks)
    _write_top_snapshot([p.get("id") for p in ranked[:_TOP_N] if p.get("id")])
    return []


def diff_candidates(star_picks, posted_half1=None, posted_half2=None):
    """Return up to _ENTRANT_MAX_PER_ALERT picks that could still join
    a posted top-10 later today. Rules:
      - signal_score >= 3
      - not do_not_bet
      - game not yet started (_row_status == 'upcoming' or absent)
      - NOT already in a posted top-10 (either half)
    Sorted strongest first. The caller decides whether to actually
    send based on the once-per-day dedup."""
    posted = set(posted_half1 or []) | set(posted_half2 or [])
    out = []
    for p in _rank_picks(star_picks):
        pid = p.get("id")
        if not pid or pid in posted:
            continue
        if (p.get("signal_score") or 0) < 3:
            continue
        if p.get("_row_status") and p.get("_row_status") != "upcoming":
            continue
        out.append(p)
        if len(out) >= _ENTRANT_MAX_PER_ALERT:
            break
    return out


def _today_ct_str():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Chicago")).date().isoformat()


def _load_dnb_sent():
    """Set of pick IDs already alerted for DNB today. Resets at day
    change in CT so yesterday's picks don't block today's alerts."""
    try:
        with open(_DNB_SENT_PATH) as f:
            data = json.load(f) or {}
        if data.get("date") != _today_ct_str():
            return set()
        return set(data.get("ids") or [])
    except Exception:
        return set()


def _write_dnb_sent(ids):
    try:
        os.makedirs(os.path.dirname(_DNB_SENT_PATH), exist_ok=True)
        with open(_DNB_SENT_PATH, "w") as f:
            json.dump({"date": _today_ct_str(), "ids": sorted(ids)}, f)
    except Exception:
        pass


def _load_result_sent():
    try:
        with open(_RESULT_SENT_PATH) as f:
            data = json.load(f) or {}
        if data.get("date") != _today_ct_str():
            return set()
        return set(data.get("ids") or [])
    except Exception:
        return set()


def _write_result_sent(ids):
    try:
        os.makedirs(os.path.dirname(_RESULT_SENT_PATH), exist_ok=True)
        with open(_RESULT_SENT_PATH, "w") as f:
            json.dump({"date": _today_ct_str(), "ids": sorted(ids)}, f)
    except Exception:
        pass


def diff_newly_graded(picks_source, posted_pairs, wins_only=True):
    """Return [(pick, slate_label), ...] for POSTED picks that just
    graded and haven't been alerted today.

    picks_source: iterable of pick dicts (from the live board or from
                  plays_log.grade_all for today).
    posted_pairs: list of (pick_id, slate_label) tuples.
    wins_only:    True (default) — only fire alerts for wins (W).
                  Losses and pushes stay quiet; the EOD summary carries
                  them in bulk. Set False to also surface L/P.

    Skips picks that are currently flagged `do_not_bet` — once a pick
    flips to DNB after posting (signal collapsed or line moved against
    us) we don't claim the result as ours, since the user was told
    not to place that bet.

    Dedup via logs/discord_result_sent.json so each pick fires exactly
    one result alert per day."""
    sent = _load_result_sent()
    pick_by_id = {p.get("id"): p for p in picks_source if p.get("id")}
    accept = ("W",) if wins_only else ("W", "L", "P")
    out = []
    for pid, slate_label in posted_pairs:
        p = pick_by_id.get(pid)
        if not p:
            continue
        if p.get("do_not_bet"):
            # Pick flipped to DNB after posting — don't announce
            # its result as a win/loss we're owning.
            continue
        r = p.get("result")
        if r in accept and pid not in sent:
            out.append((p, slate_label))
            sent.add(pid)
    if out:
        _write_result_sent(sent)
    return out


def _format_scoreline(pick):
    """Return '{away} {sa}-{sh} {home}' if the pick has final-score
    row attributes populated by the grader. '' if not."""
    try:
        sa = pick.get("row_score_away")
        sh = pick.get("row_score_home")
        if sa is None or sh is None:
            return ""
        away = pick.get("away_team") or pick.get("game_away") or ""
        home = pick.get("home_team") or pick.get("game_home") or ""
        if not away or not home:
            return f"{int(sa)}-{int(sh)}"
        return f"{away} {int(sa)}-{int(sh)} {home}"
    except Exception:
        return ""


def send_result_alerts(webhook_url, graded_pairs):
    """One single-pick PNG per newly-graded WIN. Losses and pushes
    stay quiet — only wins get a live alert (quiet feed for the user).
    Each winning pick fires exactly once; the EOD summary still shows
    the whole day's record including L/P.

    `graded_pairs` is expected to be prefiltered to wins already (via
    diff_newly_graded(wins_only=True)) but we re-check here as a
    safety net so an upstream call with the default (W/L/P) can't
    sneak a loss PNG through."""
    if not webhook_url or not graded_pairs:
        return 0
    sent = 0
    stamp = datetime.utcnow().strftime("%H:%M UTC")
    for pick, slate_label in graded_pairs:
        r = pick.get("result")
        if r != "W":
            continue
        try:
            units = float(pick.get("profit_u") or 0.0)
            units_txt = f"  ·  {units:+.2f}u"
        except (TypeError, ValueError):
            units_txt = ""
        score = _format_scoreline(pick)
        score_line = f"  ·  Final: {score}" if score else ""
        png = render_picks_png_tall(
            [pick],
            title=f"WON — {slate_label}",
            subtitle=(f"{pick.get('pick') or ''}{units_txt}"
                      f"{score_line}  ·  {stamp}"),
        )
        if png:
            msg = (f":white_check_mark: **{slate_label} pick** — **WON**"
                   f"{units_txt}{(' | ' + score) if score else ''}")
            if send_discord_image(webhook_url, png, "result_W.png", msg):
                sent += 1
    return sent


def _prior_slate_recap(current_slate):
    """When slate N is about to post, read the dispatch schedule to
    pull slate (N-1) and (N-2)'s posted pick ids, look them up in the
    day's graded picks via plays_log, and produce a one-line record
    recap ('Slate 1 so far: 2-1-0, +0.8u') for the preamble. Returns
    '' if no prior slate posted today or we can't read plays_log."""
    try:
        import plays_log as _pl
        base_dir = _BASE_DIR
        with open(os.path.join(base_dir, "logs", "discord_schedule.json")) as f:
            sched = json.load(f) or {}
    except Exception:
        return ""
    slates = sched.get("slates") or {}
    # Collect prior slates' pick_ids
    prior = []
    for s in range(1, current_slate):
        meta = slates.get(str(s)) or {}
        if meta.get("sent") and meta.get("pick_ids"):
            prior.append((s, meta["pick_ids"]))
    if not prior:
        return ""
    try:
        graded_by_date = _pl.grade_all(limit_dates=2) or {}
    except Exception:
        return ""
    today_ct = _today_ct_str()
    today_picks = graded_by_date.get(today_ct) or []
    by_id = {p.get("id"): p for p in today_picks if p.get("id")}
    lines = []
    for s, ids in prior:
        w = l = pu = pend = 0
        u = 0.0
        for pid in ids:
            p = by_id.get(pid)
            if not p:
                pend += 1
                continue
            r = p.get("result")
            if r == "W":
                w += 1
            elif r == "L":
                l += 1
            elif r == "P":
                pu += 1
            else:
                pend += 1
            try:
                u += float(p.get("profit_u") or 0.0)
            except (TypeError, ValueError):
                pass
        name = _SLATE_LABELS.get(s, f"Slate {s}")
        settled = w + l + pu
        bits = [f"{name} so far: **{w}-{l}-{pu}**"]
        if settled:
            bits.append(f"{u:+.2f}u")
        if pend:
            bits.append(f"{pend} pending")
        lines.append("  ·  ".join(bits))
    return ":bar_chart: " + " | ".join(lines)


def diff_posted_dnb(star_picks, posted_half1, posted_half2):
    """Return [(pick, half_label), ...] for POSTED top-10 picks that
    just flipped to DO NOT BET and haven't been alerted today. Dedupes
    via logs/discord_dnb_sent.json so each DNB fires at most once."""
    posted_half1 = set(posted_half1 or [])
    posted_half2 = set(posted_half2 or [])
    sent = _load_dnb_sent()
    pick_by_id = {p.get("id"): p for p in star_picks}
    out = []
    for pid in posted_half1:
        p = pick_by_id.get(pid)
        if p and p.get("do_not_bet") and pid not in sent:
            out.append((p, "First Half"))
            sent.add(pid)
    for pid in posted_half2:
        p = pick_by_id.get(pid)
        if p and p.get("do_not_bet") and pid not in sent:
            out.append((p, "Second Half"))
            sent.add(pid)
    if out:
        _write_dnb_sent(sent)
    return out


# Max new entrants surfaced per live-update message. User wants a
# very quiet channel — only alert when a short list of candidates
# could replace a posted top-10 pick before game time.
_ENTRANT_MAX_PER_ALERT = 2


def _pick_replacement_for(dnb_pick, star_picks, exclude_ids):
    """Find the highest-signal upcoming pick on today's live board
    that could step in for a posted pick that just flipped to DNB.

    Rules:
      - must clear the Discord qualification bar (_qualifies_for_discord)
      - must NOT be do_not_bet itself
      - must NOT already be in a posted slate (exclude_ids)
      - must NOT be an alt-line pick (replacements are standalone bets)
      - game must not have started yet (_row_status upcoming OR absent)
      - sort by signal_score desc, then EV desc, pick top one
      - returns None if nothing qualifies — caller falls back to plain
        DNB alert without a replacement.
    """
    if not star_picks:
        return None
    candidates = []
    for p in star_picks:
        pid = p.get("id")
        if not pid or pid in exclude_ids or pid == dnb_pick.get("id"):
            continue
        if p.get("do_not_bet") or p.get("model_only"):
            continue
        if p.get("alt_line_flag") or p.get("source") == "alt":
            continue
        if p.get("_row_status") and p.get("_row_status") != "upcoming":
            continue
        if not _qualifies_for_discord(p):
            continue
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (
        -(int(p.get("signal_score") or 0)),
        -_safe_float(p.get("ev_pct")),
    ))
    return candidates[0]


def send_change_alert(webhook_url, entrants=None, dnb_alerts=None,
                      star_picks=None, exclude_ids=None):
    """Post live updates.

    DNB alerts fire with a REPLACEMENT: for each posted pick that
    flipped to DO NOT BET we find the highest-signal upcoming pick
    on today's live board that isn't already in a posted slate, and
    send ONE combined message naming the dropped pick and the
    replacement. If no replacement qualifies, we fall back to the
    plain DNB message so the user still knows the pick is dead.

    Entrants (new picks that could join the top N later today) are
    capped at _ENTRANT_MAX_PER_ALERT and rate-limited to one message
    every _LIVE_ALERT_COOLDOWN_SEC.
    """
    if not webhook_url:
        return 0
    sent = 0
    stamp = datetime.utcnow().strftime("%H:%M UTC")
    star_picks = star_picks or []
    exclude_ids = set(exclude_ids or [])

    # --- DNB alerts WITH REPLACEMENT (per-pick-id dedup upstream) ---
    dnb_alerts = dnb_alerts or []
    for dnb_pick, label in dnb_alerts:
        replacement = _pick_replacement_for(
            dnb_pick, star_picks,
            exclude_ids | {p.get("id") for p, _ in dnb_alerts if p.get("id")})
        # Build the one combined PNG. If we have a replacement we
        # stack both picks on the card (dropped on top, replacement
        # below). If no replacement, just the dropped pick.
        if replacement:
            pngs_payload = [dnb_pick, replacement]
            title = f"DNB: {dnb_pick.get('pick') or ''}  →  REPLACEMENT"
            subtitle = (f"{label} slate pick dropped; "
                        f"replacement is the strongest upcoming pick on today's live board — {stamp}")
            fname = "dnb_with_replacement.png"
            try:
                _r_sig = int(replacement.get("signal_score") or 0)
                _r_ev  = float(replacement.get("ev_pct") or 0.0)
            except Exception:
                _r_sig, _r_ev = 0, 0.0
            msg = (f":no_entry: **{label} pick → DO NOT BET**  "
                   f"({dnb_pick.get('pick') or ''})\n"
                   f":arrow_right: **Replacement**: {replacement.get('pick') or ''}  "
                   f"(Signal {_r_sig}/5, "
                   f"{'+' if _r_ev >= 0 else ''}{_r_ev:.1f}% EV, "
                   f"@ {replacement.get('decimal') or '?'}, "
                   f"{replacement.get('book') or '?'})")
        else:
            pngs_payload = [dnb_pick]
            title = f"DNB: {dnb_pick.get('pick') or ''}"
            subtitle = (f"{label} slate pick dropped; "
                        f"no eligible replacement on today's live board — {stamp}")
            fname = "dnb.png"
            msg = (f":no_entry: **{label} pick → DO NOT BET**  "
                   f"({dnb_pick.get('pick') or ''})\n"
                   f"No replacement available from today's live board.")
        png = render_picks_png_tall(pngs_payload, title=title, subtitle=subtitle)
        if png and send_discord_image(webhook_url, png, fname, msg):
            sent += 1

    # --- Entrant alerts (once per day, capped at 2 picks per message)
    entrants = (entrants or [])[:_ENTRANT_MAX_PER_ALERT]
    if entrants and not _candidates_sent_today():
        n = len(entrants)
        png = render_picks_png_tall(
            entrants,
            title=f"+ Top {_TOP_N} candidate{'s' if n > 1 else ''}",
            subtitle=f"{n} new pick{'s' if n > 1 else ''} could replace "
                     f"a posted pick before game time — {stamp}",
        )
        if png:
            msg = (f":star: **Candidate{'s' if n > 1 else ''}** — "
                   f"{n} new pick{'s' if n > 1 else ''} could join the "
                   f"top {_TOP_N} later today")
            if send_discord_image(webhook_url, png, "entrants.png", msg):
                sent += 1
                _mark_candidates_sent()
    return sent


def _candidates_sent_today():
    """Have we posted the once-per-day entrant/candidate alert already?"""
    try:
        with open(_CANDIDATES_SENT_PATH) as f:
            data = json.load(f) or {}
        return data.get("date") == _today_ct_str()
    except Exception:
        return False


def _mark_candidates_sent():
    try:
        os.makedirs(os.path.dirname(_CANDIDATES_SENT_PATH), exist_ok=True)
        with open(_CANDIDATES_SENT_PATH, "w") as f:
            json.dump({"date": _today_ct_str()}, f)
    except Exception:
        pass


# -------------- Change detection snapshot --------------
#
# Previously this held a {id: {signal_score, do_not_bet}} map for the
# whole slate and alerted on any addition or DO NOT BET flip — noisy
# because the Signal 3 bar is low enough that lots of picks pop in and
# out as prices move. The current flow uses only `_load_top_snapshot`
# + `_write_top_snapshot` (a plain list of the current top-N ids) and
# alerts only when a NEW id enters the top-N. The old per-pick file is
# kept read/writable for backward compat (nothing writes to it now).

def _load_top_snapshot():
    """Return the prior top-N pick id list, or [] if none."""
    try:
        with open(_TOP_SNAP_PATH) as f:
            return json.load(f) or []
    except Exception:
        return []


def _write_top_snapshot(ids):
    try:
        os.makedirs(os.path.dirname(_TOP_SNAP_PATH), exist_ok=True)
        with open(_TOP_SNAP_PATH, "w") as f:
            json.dump(list(ids or []), f)
    except Exception:
        pass


# -------------- Dynamic schedule / agenda / results --------------

_HALF_FLOOR_HOUR_1 = 4   # earliest clock-hour the first-half post may fire
_HALF_FLOOR_MIN_1  = 30  # :30
_LEAD_HOURS        = 2   # how many hours before earliest game to post picks
_RESULTS_WAIT_HRS  = 4   # wait this long after latest game START to post results


def send_agenda(webhook_url, slates, eod_at_iso, date_label):
    """Post the day's schedule as a text message at 12:15 AM CT so
    viewers know when picks will land. v2 slate layout — 3 fixed CT
    posts + EOD summary."""
    if not webhook_url:
        return 0

    def _fmt_iso(iso):
        if not iso:
            return "—"
        try:
            dt = datetime.fromisoformat(iso)
            return dt.strftime("%I:%M %p CT").lstrip("0")
        except Exception:
            return "—"

    lines = [
        f":calendar_spiral: **Today's schedule — {date_label}**",
        "",
        ":sun_with_face: **Morning slate** — posting at "
        f"**{_fmt_iso((slates.get('1') or {}).get('post_at_iso'))}**  "
        "(top 5, games starting before 12 PM CT)",
        ":partly_sunny: **Afternoon slate** — posting at "
        f"**{_fmt_iso((slates.get('2') or {}).get('post_at_iso'))}**  "
        "(top 5, games 12 PM - 6 PM CT)",
        ":city_sunset: **Prime Time slate** — posting at "
        f"**{_fmt_iso((slates.get('3') or {}).get('post_at_iso'))}**  "
        "(top 5, games 6 PM CT or later)",
        ":bar_chart: **End-of-day summary** — ~"
        f"**{_fmt_iso(eod_at_iso)}**  "
        "(one slide with all picks + their W/L/P + units)",
        "",
        "Each slate is locked once posted. Hard DO NOT BET reversals "
        "(signal collapse or scratch) will post as single-pick alerts.",
        "",
        "Dashboard: https://cheeks-model.onrender.com",
    ]
    return 1 if post_text(webhook_url, "\n".join(lines)) else 0


