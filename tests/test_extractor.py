"""Tests for extractor module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from webnav.extractor import (
    find_code, _extract_candidates, _flatten_tree_text, _ADVANCED_SCAN_JS,
)
from tests.conftest import make_mock_page


class TestExtractCandidates:
    def test_finds_code_in_text(self):
        text = "Your code is: AB3F9X"
        codes = _extract_candidates(text)
        assert "AB3F9X" in codes

    def test_multiple_codes(self):
        text = "AB3F9X and XY12CD"
        codes = _extract_candidates(text)
        assert len(codes) == 2

    def test_no_codes(self):
        text = "No code here at all"
        assert _extract_candidates(text) == []

    def test_filters_false_positives(self):
        text = "SUBMIT BUTTON REVEAL AB3F9X"
        codes = _extract_candidates(text)
        assert "SUBMIT" not in codes
        assert "BUTTON" not in codes
        assert "REVEAL" not in codes
        assert "AB3F9X" in codes

    def test_lowercase_not_matched(self):
        text = "ab3f9x"
        assert _extract_candidates(text) == []


class TestFlattenTreeText:
    def test_flattens_nested(self):
        tree = {
            "name": "root",
            "children": [
                {"name": "child1"},
                {"name": "child2", "children": [{"name": "grandchild"}]},
            ],
        }
        text = _flatten_tree_text(tree)
        assert "root" in text
        assert "child1" in text
        assert "grandchild" in text


class TestFindCodeConsolidated:
    """Tests that find_code() uses a single consolidated evaluate call."""

    @pytest.mark.asyncio
    async def test_single_evaluate_call(self):
        """find_code() should make exactly ONE page.evaluate() call (not 7+)."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code is None
        assert page.evaluate.call_count == 1

    @pytest.mark.asyncio
    async def test_passes_skip_set_to_js(self):
        """find_code() passes used_codes as list to the JS function."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        page.frames = [MagicMock()]
        await find_code(page, used_codes={"AB3F9X", "XY45AB"})
        args = page.evaluate.call_args[0]
        assert len(args) == 2
        assert isinstance(args[1], list)
        assert set(args[1]) == {"AB3F9X", "XY45AB"}


class TestFindCode:
    """Tests for find_code() — now uses a single consolidated evaluate call."""

    @pytest.mark.asyncio
    async def test_finds_code(self):
        """Consolidated scan returns a code."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value="XY45AB")
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code == "XY45AB"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_code(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code is None

    @pytest.mark.asyncio
    async def test_skips_used_codes_via_js(self):
        """JS scan filters used codes — returns None when only match is used."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        page.frames = [MagicMock()]
        code = await find_code(page, used_codes={"XY45AB"})
        assert code is None

    @pytest.mark.asyncio
    async def test_returns_non_used_code(self):
        """JS scan returns a code that is NOT in the used set."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value="Z9K3LP")
        page.frames = [MagicMock()]
        code = await find_code(page, used_codes={"XY45AB"})
        assert code == "Z9K3LP"

    @pytest.mark.asyncio
    async def test_frame_fallback(self):
        """Falls back to Playwright frame search when main scan finds nothing."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        frame = MagicMock()
        frame.evaluate = AsyncMock(return_value="Code is: K9M3PL")
        page.frames = [MagicMock(), frame]  # main + 1 iframe
        code = await find_code(page)
        assert code == "K9M3PL"

    @pytest.mark.asyncio
    async def test_frame_fallback_skips_used_codes(self):
        """Frame search respects the used_codes skip set."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        frame = MagicMock()
        frame.evaluate = AsyncMock(return_value="Code is: K9M3PL")
        page.frames = [MagicMock(), frame]
        code = await find_code(page, used_codes={"K9M3PL"})
        assert code is None

    @pytest.mark.asyncio
    async def test_frame_exception_handled(self):
        """Frame evaluate errors are caught gracefully."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)
        frame = MagicMock()
        frame.evaluate = AsyncMock(side_effect=Exception("detached"))
        page.frames = [MagicMock(), frame]
        code = await find_code(page)
        assert code is None


class TestAdvancedScanJS:
    """Verify _ADVANCED_SCAN_JS is a valid JS string constant."""

    def test_js_constant_is_nonempty_string(self):
        assert isinstance(_ADVANCED_SCAN_JS, str)
        assert len(_ADVANCED_SCAN_JS) > 100

    def test_js_contains_key_sections(self):
        assert "Split content" in _ADVANCED_SCAN_JS or "split" in _ADVANCED_SCAN_JS.lower()
        assert "ROT13" in _ADVANCED_SCAN_JS or "rot13" in _ADVANCED_SCAN_JS.lower()
        assert "template" in _ADVANCED_SCAN_JS.lower()
        assert "noscript" in _ADVANCED_SCAN_JS.lower()
        assert "styleSheets" in _ADVANCED_SCAN_JS or "cssRules" in _ADVANCED_SCAN_JS
        assert "fromCharCode" in _ADVANCED_SCAN_JS
