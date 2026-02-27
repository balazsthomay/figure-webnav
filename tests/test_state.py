"""Tests for state module."""

from __future__ import annotations

import time

import pytest

from webnav.state import StateTracker
from tests.conftest import make_mock_page


class TestStateTracker:
    def test_initial_state(self):
        tracker = StateTracker()
        assert tracker.current_step == 1
        assert tracker.steps_completed == 0
        assert tracker.steps_failed == 0

    def test_begin_step(self):
        tracker = StateTracker()
        tracker.begin_step(5)
        assert tracker.current_step == 5
        assert tracker.retry_count == 0

    def test_step_elapsed(self):
        tracker = StateTracker()
        tracker.begin_step(1)
        # Should be very small (< 0.1s)
        assert tracker.step_elapsed() < 0.1

    def test_total_elapsed(self):
        tracker = StateTracker()
        assert tracker.total_elapsed() < 0.1

    def test_time_remaining(self):
        tracker = StateTracker()
        assert tracker.time_remaining() > 289  # ~290s budget

    def test_is_over_budget(self):
        tracker = StateTracker(max_time=0.0)
        assert tracker.is_over_budget()

    def test_is_not_over_budget(self):
        tracker = StateTracker()
        assert not tracker.is_over_budget()

    def test_is_step_timed_out(self):
        tracker = StateTracker(max_step_time=0.0)
        tracker.begin_step(1)
        assert tracker.is_step_timed_out()

    def test_is_not_step_timed_out(self):
        tracker = StateTracker()
        tracker.begin_step(1)
        assert not tracker.is_step_timed_out()

    def test_stuck_detection(self):
        tracker = StateTracker(max_retries=3)
        assert not tracker.is_stuck()
        tracker.increment_retry()
        tracker.increment_retry()
        assert not tracker.is_stuck()
        tracker.increment_retry()
        assert tracker.is_stuck()

    def test_mark_completed(self):
        tracker = StateTracker()
        tracker.mark_completed()
        assert tracker.steps_completed == 1

    def test_mark_failed(self):
        tracker = StateTracker()
        tracker.mark_failed()
        assert tracker.steps_failed == 1

    def test_summary(self):
        tracker = StateTracker()
        tracker.mark_completed()
        tracker.mark_completed()
        tracker.mark_failed()
        s = tracker.summary()
        assert s["steps_completed"] == 2
        assert s["steps_failed"] == 1
        assert "total_time" in s


class TestDetectStepFromUrl:
    @pytest.mark.asyncio
    async def test_parses_step_from_url(self):
        tracker = StateTracker()
        page = make_mock_page(url="https://example.com/step7?version=2")
        step = await tracker.detect_step_from_url(page)
        assert step == 7

    @pytest.mark.asyncio
    async def test_fallback_to_current(self):
        tracker = StateTracker()
        tracker.current_step = 3
        page = make_mock_page(url="https://example.com/")
        step = await tracker.detect_step_from_url(page)
        assert step == 3

    @pytest.mark.asyncio
    async def test_check_advancement(self):
        tracker = StateTracker()
        page = make_mock_page(url="https://example.com/step6?version=2")
        assert await tracker.check_advancement(page, 5) is True
        assert await tracker.check_advancement(page, 6) is False

    @pytest.mark.asyncio
    async def test_check_advancement_final_step_completion_page(self):
        """Final step: URL navigated away from /step30 → should be True."""
        tracker = StateTracker(total_steps=30)
        page = make_mock_page(url="https://example.com/complete")
        assert await tracker.check_advancement(page, 30) is True

    @pytest.mark.asyncio
    async def test_check_advancement_final_step_still_on_step(self):
        """Final step: URL still at /step30 → should be False (not yet advanced)."""
        tracker = StateTracker(total_steps=30)
        page = make_mock_page(url="https://example.com/step30?version=2")
        assert await tracker.check_advancement(page, 30) is False

    @pytest.mark.asyncio
    async def test_check_advancement_final_step_results_page(self):
        """Final step: URL navigated to results → should be True."""
        tracker = StateTracker(total_steps=30)
        page = make_mock_page(url="https://example.com/results?score=30")
        assert await tracker.check_advancement(page, 30) is True

    @pytest.mark.asyncio
    async def test_check_advancement_non_final_step_no_advance(self):
        """Non-final step: URL without step pattern → fallback to current_step."""
        tracker = StateTracker(total_steps=30)
        tracker.current_step = 15
        page = make_mock_page(url="https://example.com/loading")
        # detect_step_from_url returns current_step (15), 15 > 15 = False
        assert await tracker.check_advancement(page, 15) is False
