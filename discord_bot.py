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
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOT_PATH = os.path.join(_BASE_DIR, "logs", "discord_last_snapshot.json")

# Visual constants — matches the social-slides template.
_W, _H = 1080, 1920
_PICKS_PER_PNG = 5

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
    Returns the y position AFTER the card (for stacking)."""
    card_h = 220
    pad = 40
    x0 = pad
    x1 = _W - pad

    is_dnb = bool(pick.get("do_not_bet"))
    border = _DNB if is_dnb else _RULE
    try:
        draw.rounded_rectangle([x0, y, x1, y + card_h], radius=18, fill=_CARD, outline=border, width=2 if is_dnb else 1)
    except Exception:
        draw.rectangle([x0, y, x1, y + card_h], fill=_CARD, outline=border)

    # Row 1: star + pick text + status chip
    y_row = y + 20
    f_pick  = _font(34, bold=True)
    f_small = _font(22)
    f_chip  = _font(20, bold=True)
    f_mono  = _font(22)

    tx = x0 + 24
    if pick.get("signal_score") == 5:
        draw.text((tx, y_row + 2), "LOCK", font=f_chip, fill=_LOCK)
        tx += 90
    draw.text((tx, y_row + 2), "*", font=_font(32, bold=True), fill=_STAR)
    tx += 24
    pick_text = pick.get("pick") or ""
    if is_dnb:
        pick_text = f"[DO NOT BET] {pick_text}"
    _draw_text(draw, (tx, y_row), pick_text, f_pick,
               fill=(_DNB if is_dnb else _INK), max_w=x1 - tx - 180)

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
            cw = bbox[2] - bbox[0] + 32
            _draw_pill(draw, x1 - 24 - cw, y_row - 2, chip_text, f_chip, chip_fg, chip_bg)
        except Exception:
            pass

    # Row 2: sport + matchup
    y_row = y + 76
    sport = pick.get("sport_name") or pick.get("sport") or ""
    away  = pick.get("away_team") or pick.get("game_away") or ""
    home  = pick.get("home_team") or pick.get("game_home") or ""
    market = pick.get("market") or ""
    sub = f"{sport}  |  {away} @ {home}  |  {market}"
    _draw_text(draw, (x0 + 24, y_row), sub, f_small, fill=_MUTED, max_w=x1 - x0 - 48)

    # Row 3: Signal + EV + line move + price
    y_row = y + 128
    tx = x0 + 24
    if pick.get("signal_score") is not None:
        rx, _ = _draw_pill(draw, tx, y_row, f"Signal {pick['signal_score']}/5", f_chip,
                           fill=_SIGNAL_3, bg=(10, 30, 50))
        tx = rx + 10
    ev = pick.get("ev_pct")
    if ev is not None:
        try:
            ev_txt = f"+{float(ev):.1f}% EV"
            bg = _CARD
            fg = _WARN if float(ev) > 15 else _GOOD
            rx, _ = _draw_pill(draw, tx, y_row, ev_txt, f_chip, fill=fg, bg=(10, 40, 20))
            tx = rx + 10
        except (TypeError, ValueError):
            pass
    lm = pick.get("line_move_pp")
    if lm is not None and abs(lm) >= 0.001:
        arrow = "UP" if lm > 0 else "DN"
        _draw_text(draw, (tx, y_row + 4), f"{arrow} {lm * 100:+.1f}pp", f_small,
                   fill=(_GOOD if lm > 0 else _BAD))

    # Row 4: start time + odds
    y_row = y + 172
    tx = x0 + 24
    time_abs = pick.get("time_abs") or ""
    countdown = pick.get("countdown") or ""
    time_text = f"{countdown}  {time_abs}".strip() if countdown else time_abs
    _draw_text(draw, (tx, y_row), time_text, f_mono, fill=_MUTED_2)
    dec = pick.get("decimal")
    book = pick.get("book") or ""
    if dec:
        try:
            price_txt = f"@ {float(dec):.2f}  {book}"
            _draw_text(draw, (x1 - 24, y_row), price_txt, f_mono, fill=_INK, anchor="rt")
        except (TypeError, ValueError):
            pass

    return y + card_h + 14


def render_picks_png(picks, title="Today's Picks", subtitle=""):
    """Render a batch of picks (max _PICKS_PER_PNG) to a 1080x1920 PNG.
    Returns bytes. Returns None if Pillow is unavailable."""
    if not _PIL_OK:
        return None
    canvas = Image.new("RGB", (_W, _H), _BG)
    draw = ImageDraw.Draw(canvas)

    # Header
    f_title = _font(72, bold=True)
    f_sub   = _font(28)
    draw.text((40, 48), title, font=f_title, fill=_INK)
    if subtitle:
        draw.text((40, 134), subtitle, font=f_sub, fill=_MUTED)

    # Divider
    draw.line([40, 190, _W - 40, 190], fill=_RULE, width=2)

    # Picks
    y = 220
    for p in picks[:_PICKS_PER_PNG]:
        y = _render_pick_card(canvas, draw, p, y)
        if y > _H - 240:
            break

    # Footer
    footer_y = _H - 90
    f_small = _font(22)
    draw.line([40, footer_y - 10, _W - 40, footer_y - 10], fill=_RULE, width=1)
    draw.text((40, footer_y), "cheeks-model.onrender.com",
              font=f_small, fill=_MUTED_2)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        draw.text((_W - 40, footer_y), stamp, font=f_small, fill=_MUTED_2, anchor="rt")
    except Exception:
        draw.text((_W - 300, footer_y), stamp, font=f_small, fill=_MUTED_2)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def send_discord_image(webhook_url, png_bytes, filename, message=""):
    """POST a PNG to a Discord webhook via multipart. Returns True on
    success. Discord webhooks accept files up to ~25MB; our PNGs are
    typically ~80-120 KB."""
    if not webhook_url or not png_bytes:
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
        req = Request(webhook_url, data=body, method="POST", headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })
        urlopen(req, timeout=10).close()
        return True
    except Exception:
        return False


def send_daily_picks(webhook_url, picks, date_label=""):
    """Chunk picks into _PICKS_PER_PNG groups, render each, send to
    Discord with a header message. Returns count of PNGs sent."""
    if not webhook_url or not picks:
        return 0
    chunks = [picks[i:i + _PICKS_PER_PNG] for i in range(0, len(picks), _PICKS_PER_PNG)]
    total = len(chunks)
    sent = 0
    for i, chunk in enumerate(chunks, 1):
        png = render_picks_png(
            chunk,
            title=f"Today's Picks ({i}/{total})",
            subtitle=date_label or datetime.utcnow().strftime("%A, %B %d"),
        )
        if not png:
            continue
        msg = f"**Today's picks** ({len(picks)} total)" if i == 1 else ""
        if send_discord_image(webhook_url, png, f"picks_{i}_of_{total}.png", msg):
            sent += 1
    return sent


def send_change_alert(webhook_url, added=None, dropped=None):
    """Send a change-notification PNG when picks are added or flagged
    DO NOT BET. Separate PNGs per direction so the user can tell at a
    glance which happened."""
    if not webhook_url:
        return 0
    sent = 0
    added = added or []
    dropped = dropped or []
    if added:
        png = render_picks_png(
            added[:_PICKS_PER_PNG],
            title=f"+ New Pick{'s' if len(added) > 1 else ''}",
            subtitle=f"Added {len(added)} pick{'s' if len(added) > 1 else ''} just now",
        )
        if png and send_discord_image(webhook_url, png, "added.png",
                                       f":warning: **{len(added)} new pick{'s' if len(added) > 1 else ''}** added to the board"):
            sent += 1
    if dropped:
        png = render_picks_png(
            dropped[:_PICKS_PER_PNG],
            title=f"X DO NOT BET",
            subtitle=f"{len(dropped)} pick{'s' if len(dropped) > 1 else ''} fell off the filter",
        )
        if png and send_discord_image(webhook_url, png, "dropped.png",
                                       f":no_entry: **{len(dropped)} pick{'s' if len(dropped) > 1 else ''}** now DO NOT BET"):
            sent += 1
    return sent


# -------------- Change detection snapshot --------------

def _load_snapshot():
    """Return the previous picks snapshot {id: {signal_score, do_not_bet}}
    or {} if none."""
    try:
        with open(_SNAPSHOT_PATH) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_snapshot(snap):
    try:
        os.makedirs(os.path.dirname(_SNAPSHOT_PATH), exist_ok=True)
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump(snap, f)
    except Exception:
        pass


def diff_picks(star_picks):
    """Compare current star_picks against the last snapshot. Returns
    (added, dropped) where:
      added   = picks in current that weren't in snapshot
      dropped = picks that newly flipped to do_not_bet since last snap
    After returning, writes a fresh snapshot."""
    prev = _load_snapshot()
    current = {}
    added, dropped = [], []
    for p in star_picks:
        pid = p.get("id")
        if not pid:
            continue
        is_dnb = bool(p.get("do_not_bet"))
        current[pid] = {
            "signal_score": p.get("signal_score"),
            "do_not_bet":   is_dnb,
        }
        prev_entry = prev.get(pid)
        if prev_entry is None:
            # New pick we've never seen; skip from "added" if it's
            # a saved pick already — only alert on genuinely new ones
            # that just qualified for Signal >= 3.
            if not p.get("_from_saved"):
                added.append(p)
        elif is_dnb and not prev_entry.get("do_not_bet"):
            dropped.append(p)
    _write_snapshot(current)
    return added, dropped
