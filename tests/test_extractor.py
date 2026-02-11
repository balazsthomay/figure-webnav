"""Tests for extractor module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from webnav.extractor import find_code, _extract_candidates, _flatten_tree_text
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
