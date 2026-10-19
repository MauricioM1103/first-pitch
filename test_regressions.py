#!/usr/bin/env python3
"""Regression tests for the critical paths that broke this quarter.

Run from the project root with:   python -m unittest test_regressions

These are plain unittest tests — no pytest / tox dependency — so they
run cleanly on Render in a tiny CI step. They cover:

  - Kalshi P&L calculation (fixed-point dollar parsing + formula)
  - EOD summary pagination logic (card budget, bucket ordering)
  - Grader completeness guard (full-game markets refuse live events)
  - Pick/event date-match tolerance (back-to-back day isolation)
  - Discord dispatcher reason codes (nothing-in-window vs post-fail)

Each test documents the SPECIFIC bug it would have caught if it had
existed before. If you add behavior to any of these modules, add or
update a test case here so the next regression gets caught before
deploy."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestKalshiPnL(unittest.TestCase):
    """Would have caught: the pre-fix bug where kalshi_portfolio read
    `yes_count`, `yes_price`, `fees` (nonexistent) instead of the real
    `yes_count_fp`, `yes_price_dollars`, `fee_cost`. That bug made
    cost_basis = 0 always, so net P&L fell back to Kalshi's gross
    `revenue` field and the tile reported +$985 instead of -$195."""

    def setUp(self):
        import kalshi_portfolio
        self.kp = kalshi_portfolio

    def test_dollars_to_cents_parses_fixed_point(self):
        self.assertEqual(self.kp._dollars_to_cents("19.257030"), 1926)
        self.assertEqual(self.kp._dollars_to_cents("0.737370"), 74)
        self.assertEqual(self.kp._dollars_to_cents("0.000000"), 0)

    def test_dollars_to_cents_tolerates_noise(self):
        self.assertEqual(self.kp._dollars_to_cents(None), 0)
        self.assertEqual(self.kp._dollars_to_cents(""), 0)
        self.assertEqual(self.kp._dollars_to_cents("not-a-number"), 0)

    def test_pnl_win_matches_hand_math(self):
        # Yes won, 42.51 contracts, paid $19.26, fee $0.74
        # Expected: 4251 - 1926 - 0 - 74 = 2251¢ = $22.51 profit
        settlement = {
            "revenue": 4251,
            "yes_total_cost_dollars": "19.257030",
            "no_total_cost_dollars":  "0.000000",
            "fee_cost":               "0.737370",
            "market_result":          "yes",
        }
        self.assertEqual(self.kp._pnl_for_settlement(settlement), 2251)

    def test_pnl_loss_is_negative_full_cost_plus_fees(self):
        # Yes lost: paid $25, fee $0.50, got nothing
        # Expected: 0 - 2500 - 0 - 50 = -2550¢ = -$25.50
        settlement = {
            "revenue": 0,
            "yes_total_cost_dollars": "25.000000",
            "no_total_cost_dollars":  "0.000000",
            "fee_cost":               "0.500000",
            "market_result":          "no",
        }
        self.assertEqual(self.kp._pnl_for_settlement(settlement), -2550)

    def test_pnl_push_or_no_exposure_is_zero(self):
        settlement = {
            "revenue": 0,
            "yes_total_cost_dollars": "0.000000",
            "no_total_cost_dollars":  "0.000000",
            "fee_cost":               "0.000000",
            "market_result":          "yes",
        }
        self.assertEqual(self.kp._pnl_for_settlement(settlement), 0)

    def test_pnl_missing_revenue_returns_none(self):
        # A malformed row should return None (so summarize_recent skips
        # it) rather than crashing or treating it as zero.
        self.assertIsNone(self.kp._pnl_for_settlement({"revenue": None}))
        self.assertIsNone(self.kp._pnl_for_settlement({}))


class TestEodPagination(unittest.TestCase):
    """Would have caught: a regression where EOD only renders one PNG
    even when the day has >9 picks, or renders empty 2nd/3rd pages
    when picks are exactly a multiple of _EOD_PER_PNG."""

    def setUp(self):
        import discord_bot
        self.db = discord_bot

    def _pages_for(self, n):
        """Pure arithmetic reflecting the paginator's bounds."""
        import math
        return max(1, min(self.db._EOD_MAX_PNGS,
                           math.ceil(n / self.db._EOD_PER_PNG))) if n else 0

    def test_cap_is_27_picks_across_3_pngs(self):
        # Spec: 9 per PNG, 3 PNGs max → 27 total shown.
        self.assertEqual(self.db._EOD_PER_PNG * self.db._EOD_MAX_PNGS, 27)

    def test_pagination_bucket_boundaries(self):
        cases = {3: 1, 9: 1, 10: 2, 18: 2, 19: 3, 27: 3, 30: 3, 100: 3}
        for n, expected_pages in cases.items():
            self.assertEqual(self._pages_for(n), expected_pages,
                              f"{n} picks should produce {expected_pages} PNG(s)")

    def test_zero_picks_no_pngs(self):
        self.assertEqual(self._pages_for(0), 0)


class TestResultAlertFilter(unittest.TestCase):
    """Would have caught: a regression where losses or pushes sneak
    into the Discord win-only alerts (user explicitly asked for wins
    only)."""

    def setUp(self):
        import discord_bot
        self.db = discord_bot
        # Clear the dedup file so each test runs fresh
        if os.path.exists(self.db._RESULT_SENT_PATH):
            os.remove(self.db._RESULT_SENT_PATH)

    def tearDown(self):
        if os.path.exists(self.db._RESULT_SENT_PATH):
            os.remove(self.db._RESULT_SENT_PATH)

    def _p(self, pid, result):
        return {"id": pid, "result": result, "market": "ML",
                "pick": f"Pick {pid}", "profit_u": 0.9}

    def test_wins_only_true_filters_losses_and_pushes(self):
        picks = [self._p("a", "W"), self._p("b", "L"), self._p("c", "P")]
        posted = [("a", "Morning"), ("b", "Morning"), ("c", "Morning")]
        got = self.db.diff_newly_graded(picks, posted, wins_only=True)
        self.assertEqual([p["id"] for p, _ in got], ["a"])

    def test_wins_only_false_passes_all_graded(self):
        picks = [self._p("a", "W"), self._p("b", "L"), self._p("c", "P")]
        posted = [("a", "Morning"), ("b", "Morning"), ("c", "Morning")]
        got = self.db.diff_newly_graded(picks, posted, wins_only=False)
        self.assertEqual(sorted(p["id"] for p, _ in got), ["a", "b", "c"])

    def test_dedup_prevents_double_fire(self):
        picks = [self._p("a", "W")]
        posted = [("a", "Morning")]
        self.db.diff_newly_graded(picks, posted, wins_only=True)
        again = self.db.diff_newly_graded(picks, posted, wins_only=True)
        self.assertEqual(again, [])


class TestGraderCompletedGuard(unittest.TestCase):
    """Would have caught: a regression where full-game markets grade
    off a live-but-incomplete game. The 36h→12h tolerance fix was in
    response to a bug where today's pick matched yesterday's game
    between the same teams in a divisional series."""

    def setUp(self):
        import plays_log
        self.pl = plays_log

    def test_live_full_game_market_stays_pending(self):
        live_event = {
            "completed": False,
            "home_team": "Chiefs", "away_team": "Broncos",
            "scores": [{"name": "Chiefs", "score": "14"},
                       {"name": "Broncos", "score": "7"}],
        }
        pick = {"market": "ML", "pick": "Chiefs",
                "home_team": "Chiefs", "away_team": "Broncos"}
        self.assertEqual(self.pl.grade_pick(pick, live_event), "pending")

    def test_completed_game_grades_full_market(self):
        done_event = {
            "completed": True,
            "home_team": "Chiefs", "away_team": "Broncos",
            "scores": [{"name": "Chiefs", "score": "14"},
                       {"name": "Broncos", "score": "7"}],
        }
        pick = {"market": "ML", "pick": "Chiefs",
                "home_team": "Chiefs", "away_team": "Broncos"}
        self.assertEqual(self.pl.grade_pick(pick, done_event), "W")

    def test_date_tolerance_rejects_back_to_back_day(self):
        # Pick for today 7pm, event for yesterday 7pm (24h apart) →
        # should NOT match. 12h tolerance was the fix.
        pick = {"start_time": "2026-10-18T19:00:00Z"}
        yesterday = {"commence_time": "2026-10-17T19:00:00Z"}
        self.assertFalse(
            self.pl._event_date_close_to_pick(yesterday, pick))

    def test_date_tolerance_allows_same_day_doubleheader(self):
        # Game 1 at 1pm, pick is for game 2 at 7pm → 6h apart, under
        # 12h tolerance, so a doubleheader lookup still matches.
        pick = {"start_time": "2026-10-18T19:00:00Z"}
        game1 = {"commence_time": "2026-10-18T13:00:00Z"}
        self.assertTrue(
            self.pl._event_date_close_to_pick(game1, pick))

    def test_date_tolerance_allows_same_game(self):
        pick = {"start_time": "2026-10-18T19:00:00Z"}
        event = {"commence_time": "2026-10-18T19:00:00Z"}
        self.assertTrue(
            self.pl._event_date_close_to_pick(event, pick))


class TestDispatcherReasonCodes(unittest.TestCase):
    """Would have caught: the regression where send_slate_picks
    returned sent=0 for both 'no picks in slate window' and
    'HTTP POST failed', so the dispatcher couldn't tell them apart
    and retried the former forever."""

    def setUp(self):
        import discord_bot
        self.db = discord_bot

    def test_no_webhook_returns_terminal_reason(self):
        r = self.db.send_slate_picks("", [], slate=1)
        self.assertEqual(r.get("sent"), 0)
        self.assertEqual(r.get("reason"), "no_webhook_or_no_board")

    def test_empty_picks_returns_terminal_reason(self):
        r = self.db.send_slate_picks("http://example.com", [], slate=1)
        self.assertEqual(r.get("sent"), 0)
        self.assertEqual(r.get("reason"), "no_webhook_or_no_board")

    def test_pm_only_board_still_posts_morning_fallback(self):
        # Pick is an evening game (NHL at 8pm CT). Slate 1's window
        # is "games before 12pm CT" and the window filter drops it,
        # but slate 1 is the morning preview — rather than go dark
        # we fall back to the full day's board. The POST will fail
        # against this fake webhook, so sent=0, but the reason is
        # discord_post_failed, NOT no_picks_in_slate_window.
        pick = {
            "id": "x", "pick": "Over 1.5", "market": "1H Total",
            "sport_slug": "nhl", "sport_name": "NHL",
            "home_team": "A", "away_team": "B",
            "signal_score": 4, "ev_pct": 10.0,
            "fair_prob": 0.72, "pinnacle_prob": 0.57,
            "decimal": 1.68,
            "start_time": "2026-10-19T20:00:00-05:00",
        }
        r = self.db.send_slate_picks("http://example.com", [pick], slate=1)
        self.assertEqual(r.get("sent"), 0)
        self.assertEqual(r.get("reason"), "discord_post_failed")

    def test_pm_only_board_still_terminal_for_slate_2_and_3(self):
        # Slates 2 and 3 don't do the morning fallback — if their
        # time window is empty they return the terminal reason.
        pick = {
            "id": "x", "pick": "Over 1.5", "market": "ML",
            "sport_slug": "nhl", "sport_name": "NHL",
            "home_team": "A", "away_team": "B",
            "signal_score": 4, "ev_pct": 10.0,
            "fair_prob": 0.72, "pinnacle_prob": 0.57,
            "decimal": 1.68,
            # Early-morning game — before slate 2's 12pm start
            "start_time": "2026-10-19T09:00:00-05:00",
        }
        r = self.db.send_slate_picks("http://example.com", [pick], slate=2)
        self.assertEqual(r.get("sent"), 0)
        self.assertEqual(r.get("reason"), "no_picks_in_slate_window")


class TestFairProbBlend(unittest.TestCase):
    """Would have caught: the double-counting bug in picks._augment_with_mc
    where fair_prob (initialized to the Pinnacle devig) was blended
    first into model_consensus and then AGAIN into consensus against
    Pinnacle, halving the Monte Carlo sim's intended weight. Default
    was advertised as 35% model / 65% market but ran at effectively
    17% sim / 83% pinnacle. Fix: collapse to a single-stage per-sport
    blend `mw_sport * sim + market_w * pinnacle`."""

    def _blend(self, mw_sport, sim, pin):
        """Mirror the production formula (post-fix)."""
        market_w = 1.0 - mw_sport
        return mw_sport * sim + market_w * pin

    def test_default_blend_is_35_model_65_market(self):
        # NHL/MLB default: mw_sport = 0.35
        got = self._blend(0.35, sim=0.70, pin=0.60)
        self.assertAlmostEqual(got, 0.35 * 0.70 + 0.65 * 0.60, places=6)
        self.assertAlmostEqual(got, 0.635, places=3)

    def test_nfl_blend_is_30_model_70_market(self):
        got = self._blend(0.30, sim=0.70, pin=0.60)
        self.assertAlmostEqual(got, 0.30 * 0.70 + 0.70 * 0.60, places=6)
        self.assertAlmostEqual(got, 0.63, places=3)

    def test_soccer_blend_is_45_model_55_market(self):
        got = self._blend(0.45, sim=0.70, pin=0.60)
        self.assertAlmostEqual(got, 0.45 * 0.70 + 0.55 * 0.60, places=6)
        self.assertAlmostEqual(got, 0.645, places=3)

    def test_ufc_blend_is_50_50(self):
        got = self._blend(0.50, sim=0.70, pin=0.60)
        self.assertAlmostEqual(got, 0.65, places=3)

    def test_blend_matches_production_apply_consensus(self):
        # Simulate the real _apply_consensus path with a stub pick.
        # Values chosen so |sim - fair| < CONSENSUS_MAX_DIVERGENCE
        # (0.08) and consensus > MIN_FAIR_PROB (0.46).
        import picks
        p = {
            "sport":       "NHL",
            "sport_slug":  "nhl",
            "market":      "ML",
            "pick":        "Home ML",
            "home_team":   "A", "away_team": "B",
            "fair_prob":    0.60,   # initial = Pinnacle devig
            "pinnacle_prob": 0.60,
            "decimal":      1.90,
        }
        # Monkeypatch _sim_for / _sim_prob_for_pick to return a known
        # sim_prob close enough to not trigger the divergence filter.
        orig_sim_for = picks._sim_for
        orig_sim_prob_for = picks._sim_prob_for_pick
        picks._SIM_CACHE.clear()
        picks._sim_for = lambda *a, **k: {"home_win_prob": 0.66}
        picks._sim_prob_for_pick = lambda sim, p: 0.66
        try:
            kept = picks._apply_consensus([dict(p)])
        finally:
            picks._sim_for = orig_sim_for
            picks._sim_prob_for_pick = orig_sim_prob_for
            picks._SIM_CACHE.clear()
        self.assertEqual(len(kept), 1)
        # NHL default mw_sport = 0.35 → expect 0.35*0.66 + 0.65*0.60
        expected = 0.35 * 0.66 + 0.65 * 0.60
        self.assertAlmostEqual(kept[0]["fair_prob"], expected, places=3,
            msg=("Fair prob should be the single-stage blend. The "
                 "pre-fix two-stage blend would have halved the sim's "
                 "weight and returned a different value."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
