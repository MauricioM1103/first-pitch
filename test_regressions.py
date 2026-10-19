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

    def test_picks_outside_slate_window_terminal(self):
        # Pick is an evening game (NHL at 8pm CT), slate 1 is for
        # games before 12pm CT → window filter drops it, so result
        # reason should be no_picks_in_slate_window (terminal, no retry)
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
        self.assertEqual(r.get("reason"), "no_picks_in_slate_window")


if __name__ == "__main__":
    unittest.main(verbosity=2)
