"""Tests for page_cleaner module."""

from __future__ import annotations

import pytest

from webnav.page_cleaner import CLEANUP_JS, clean_page
from tests.conftest import make_mock_page


@pytest.mark.asyncio
async def test_clean_page_calls_evaluate_twice():
    """Cleaner should run twice (to catch re-renders)."""
    page = make_mock_page()
    await clean_page(page)
    assert page.evaluate.call_count == 2


def test_cleanup_js_is_iife():
    """JS should be a self-invoking function."""
    assert CLEANUP_JS.strip().startswith("(")
    assert CLEANUP_JS.strip().endswith(")();")


def test_cleanup_js_hides_overlays():
    """JS should hide z-index based overlays (non-destructive, no remove)."""
    assert "zIndex" in CLEANUP_JS or "z-index" in CLEANUP_JS.lower()
    assert "display" in CLEANUP_JS


def test_cleanup_js_removes_fixed_overlays():
    """JS should target fixed position overlays."""
    assert "fixed" in CLEANUP_JS


def test_cleanup_js_scrolls_to_top():
    assert "scrollTo(0, 0)" in CLEANUP_JS


def test_cleanup_js_handles_pointer_events():
    """JS should disable pointer-events on smaller overlays."""
    assert "pointer-events" in CLEANUP_JS
