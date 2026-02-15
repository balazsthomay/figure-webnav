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


class TestFindCode:
    @pytest.mark.asyncio
    async def test_finds_code_in_inner_text(self):
        page = make_mock_page(inner_text="Your code is: XY45AB")
        code = await find_code(page)
        assert code == "XY45AB"

    @pytest.mark.asyncio
    async def test_finds_code_via_hidden_scan(self):
        """Code not in inner_text but found by JS hidden scan."""
        page = make_mock_page(inner_text="no codes here")
        # evaluate calls: 1st=green_code(None), 2nd=hidden_scan("AB3F9X")
        page.evaluate = AsyncMock(side_effect=[None, "AB3F9X"])
        code = await find_code(page)
        assert code == "AB3F9X"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_code(self):
        page = make_mock_page(inner_text="nothing here")
        # Both evaluate calls (green_code + hidden_scan) return None
        page.evaluate = AsyncMock(return_value=None)
        code = await find_code(page)
        assert code is None

    @pytest.mark.asyncio
    async def test_finds_code_via_js_fallback(self):
        page = make_mock_page(inner_text="nothing here")
        # Green code check returns None, hidden scan returns the code
        page.evaluate = AsyncMock(side_effect=[None, "QW3R7Y"])
        code = await find_code(page)
        assert code == "QW3R7Y"

    @pytest.mark.asyncio
    async def test_skips_used_codes(self):
        page = make_mock_page(inner_text="Your code is: XY45AB")
        code = await find_code(page, used_codes={"XY45AB"})
        assert code is None

    @pytest.mark.asyncio
    async def test_finds_next_code_when_first_used(self):
        page = make_mock_page(inner_text="Codes: XY45AB and Z9K3LP")
        code = await find_code(page, used_codes={"XY45AB"})
        assert code == "Z9K3LP"

    @pytest.mark.asyncio
    async def test_finds_base64_encoded_code(self):
        """Code not in text or hidden scan, but found via Base64 decode."""
        page = make_mock_page(inner_text="no codes here")
        # evaluate calls: green_code=None, hidden_scan=None, shadow=None,
        # iframe_recursive=None, base64_scan="AB3F9X"
        page.evaluate = AsyncMock(side_effect=[None, None, None, None, "AB3F9X"])
        # Playwright frames — only main frame (no iframes to search)
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code == "AB3F9X"

    @pytest.mark.asyncio
    async def test_base64_skips_used_codes(self):
        """Base64-decoded code is skipped if already used."""
        page = make_mock_page(inner_text="no codes here")
        # evaluate calls: green_code=None, hidden_scan=None, shadow=None,
        # iframe_recursive=None, base64_scan="AB3F9X", script_scan=None, advanced=None
        page.evaluate = AsyncMock(side_effect=[None, None, None, None, "AB3F9X", None, None])
        page.frames = [MagicMock()]
        code = await find_code(page, used_codes={"AB3F9X"})
        assert code is None

    @pytest.mark.asyncio
    async def test_finds_code_in_script_tag(self):
        """Code found in inline script tag."""
        page = make_mock_page(inner_text="no codes here")
        # evaluate calls: green=None, hidden=None, shadow=None,
        # iframe_recursive=None, base64=None, script_scan="K9M3PL"
        page.evaluate = AsyncMock(
            side_effect=[None, None, None, None, None, "K9M3PL"]
        )
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code == "K9M3PL"

    @pytest.mark.asyncio
    async def test_finds_code_via_advanced_scan(self):
        """Code found by advanced scan (split/encoding/template etc.)."""
        page = make_mock_page(inner_text="no codes here")
        # evaluate calls: green=None, hidden=None, shadow=None,
        # iframe_recursive=None, base64=None, script=None, advanced="QR7X2M"
        page.evaluate = AsyncMock(
            side_effect=[None, None, None, None, None, None, "QR7X2M"]
        )
        page.frames = [MagicMock()]
        code = await find_code(page)
        assert code == "QR7X2M"

    @pytest.mark.asyncio
    async def test_advanced_scan_skips_used_codes(self):
        """Advanced scan result is skipped if already used."""
        page = make_mock_page(inner_text="no codes here")
        # All scans return None except advanced which returns a used code
        page.evaluate = AsyncMock(
            side_effect=[None, None, None, None, None, None, "QR7X2M"]
        )
        page.frames = [MagicMock()]
        code = await find_code(page, used_codes={"QR7X2M"})
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
