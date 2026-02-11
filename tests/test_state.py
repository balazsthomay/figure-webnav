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
