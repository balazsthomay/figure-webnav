"""Tests for browser module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webnav.browser import BASE_URL, BrowserController


class TestBrowserControllerInit:
    def test_default_headless(self):
        bc = BrowserController()
        assert bc.headless is True

    def test_headed_mode(self):
        bc = BrowserController(headless=False)
        assert bc.headless is False


class TestPageProperty:
    def test_raises_when_not_started(self):
        bc = BrowserController()
        with pytest.raises(RuntimeError, match="Browser not started"):
            _ = bc.page

    def test_returns_page_when_set(self):
        bc = BrowserController()
        mock_page = MagicMock()
        bc._page = mock_page
        assert bc.page is mock_page


class TestBrowserMethods:
    """Test browser methods using a pre-set mock page."""

    def _make_controller(self) -> tuple[BrowserController, MagicMock]:
        bc = BrowserController()
        page = MagicMock()
        page.goto = AsyncMock()
        page.url = "https://example.com/step1"
        page.inner_text = AsyncMock(return_value="hello")
        page.screenshot = AsyncMock(return_value=b"png")
        page.evaluate = AsyncMock(return_value="result")
        page.wait_for_timeout = AsyncMock()
        mock_loc = MagicMock()
        mock_loc.aria_snapshot = AsyncMock(return_value="- heading 'Test'")
        page.locator = MagicMock(return_value=mock_loc)
        bc._page = page
        return bc, page

    @pytest.mark.asyncio
    async def test_goto(self):
        bc, page = self._make_controller()
        await bc.goto("https://example.com")
        page.goto.assert_called_with(
            "https://example.com", wait_until="domcontentloaded", timeout=10_000
        )

    @pytest.mark.asyncio
    async def test_goto_step(self):
        bc, page = self._make_controller()
        await bc.goto_step(5)
        page.goto.assert_called_with(
            f"{BASE_URL}/step5?version=3",
            wait_until="domcontentloaded",
            timeout=10_000,
        )

    @pytest.mark.asyncio
    async def test_current_url(self):
        bc, page = self._make_controller()
        url = await bc.current_url()
        assert url == "https://example.com/step1"

    @pytest.mark.asyncio
    async def test_aria_snapshot(self):
        bc, page = self._make_controller()
        result = await bc.aria_snapshot()
        assert result == "- heading 'Test'"

    @pytest.mark.asyncio
    async def test_evaluate_js(self):
        bc, page = self._make_controller()
        result = await bc.evaluate_js("1 + 1")
        page.evaluate.assert_called_with("1 + 1")

    @pytest.mark.asyncio
    async def test_screenshot(self):
        bc, page = self._make_controller()
        result = await bc.screenshot()
        assert result == b"png"

    @pytest.mark.asyncio
    async def test_inner_text(self):
        bc, page = self._make_controller()
        result = await bc.inner_text("body")
        page.inner_text.assert_called_with("body")

    @pytest.mark.asyncio
    async def test_wait_for_timeout(self):
        bc, page = self._make_controller()
        await bc.wait_for_timeout(500)
        page.wait_for_timeout.assert_called_with(500)


class TestHandleDialog:
    @pytest.mark.asyncio
    async def test_dismisses_dialog(self):
        dialog = MagicMock()
        dialog.dismiss = AsyncMock()
        await BrowserController._handle_dialog(dialog)
        dialog.dismiss.assert_called_once()


class TestBaseUrl:
    def test_base_url(self):
        assert "netlify.app" in BASE_URL
