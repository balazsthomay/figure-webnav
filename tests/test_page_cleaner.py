"""Tests for page_cleaner module."""

from __future__ import annotations

import pytest

from webnav.page_cleaner import CLEANUP_JS, _QUICK_CLEAN_JS, clean_page, quick_clean
from tests.conftest import make_mock_page


@pytest.mark.asyncio
async def test_clean_page_calls_evaluate_once():
    """Cleaner runs once (double-run removed — React re-renders are async)."""
    page = make_mock_page()
    await clean_page(page)
    assert page.evaluate.call_count == 1


@pytest.mark.asyncio
async def test_quick_clean_calls_evaluate_once():
    """quick_clean batches reset + clean into a single evaluate call."""
    page = make_mock_page()
    await quick_clean(page)
    assert page.evaluate.call_count == 1


def test_quick_clean_js_contains_reset_and_cleanup():
    """quick_clean JS should contain both reset and cleanup logic."""
    assert "data-wnav-noise" in _QUICK_CLEAN_JS
    assert "zIndex" in _QUICK_CLEAN_JS or "z-index" in _QUICK_CLEAN_JS.lower()


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


def test_cleanup_js_does_not_scroll_to_top():
    """Scroll-to-top was removed to avoid breaking scroll-to-reveal challenges."""
    assert "scrollTo(0, 0)" not in CLEANUP_JS


def test_cleanup_js_targets_fixed_only():
    """JS should only hide position:fixed overlays, not absolute."""
    assert "s.position === 'fixed'" in CLEANUP_JS
    # Should NOT target absolute-positioned elements (they may be challenge content)
    assert "s.position === 'absolute'" not in CLEANUP_JS
