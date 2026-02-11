"""Tests for executor module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from webnav.dispatcher import Action
from webnav.executor import (
    run,
    submit_code,
    _extract_key_sequence,
    _parse_key_tokens,
)
from tests.conftest import make_mock_page


class TestRunAction:
    @pytest.mark.asyncio
    async def test_click_action(self):
        page = make_mock_page()
        action = Action(type="click", selector="button:has-text('Reveal')")
        result = await run(page, action)
        assert result is True
        page.locator.assert_called()

    @pytest.mark.asyncio
    async def test_scroll_action(self):
        page = make_mock_page()
        action = Action(type="scroll", amount=600)
        result = await run(page, action)
        assert result is True
        page.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_wait_action(self):
        page = make_mock_page()
        action = Action(type="wait", amount=0)  # 0 for speed in tests
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_press_action(self):
        page = make_mock_page()
        action = Action(type="press", value="Enter")
        result = await run(page, action)
        assert result is True
        page.keyboard.press.assert_called_with("Enter")

    @pytest.mark.asyncio
    async def test_js_action(self):
        page = make_mock_page()
        action = Action(type="js", value="document.title")
        result = await run(page, action)
        assert result is True
        page.evaluate.assert_called_with("document.title")

    @pytest.mark.asyncio
    async def test_fill_action(self):
        page = make_mock_page()
        action = Action(type="fill", selector="input", value="test123")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_hover_action(self):
        page = make_mock_page()
        action = Action(type="hover", selector="button:has-text('Hover')")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        page = make_mock_page()
        action = Action(type="unknown")
        result = await run(page, action)
        assert result is False

    @pytest.mark.asyncio
    async def test_click_with_multiple_attempts(self):
        page = make_mock_page()
        action = Action(type="click", selector="button:has-text('Reveal')", amount=3)
        result = await run(page, action)
        assert result is True


class TestScrollAction:
    @pytest.mark.asyncio
    async def test_scroll_incremental(self):
        page = make_mock_page()
        action = Action(type="scroll", amount=250)
        result = await run(page, action)
        assert result is True
        # Should have been called multiple times (100px steps)
        assert page.evaluate.call_count >= 2


class TestSelectAction:
    @pytest.mark.asyncio
    async def test_select_action(self):
        page = make_mock_page()
        action = Action(type="select", value="Option A")
        result = await run(page, action)
        assert result is True


class TestSubmitCode:
    @pytest.mark.asyncio
    async def test_submit_fills_and_clicks(self):
        page = make_mock_page()
        result = await submit_code(page, "AB3F9X")
        assert result is True

    @pytest.mark.asyncio
    async def test_submit_handles_missing_textbox(self):
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        result = await submit_code(page, "AB3F9X")
        assert result is False

    @pytest.mark.asyncio
    async def test_submit_uses_enter_when_no_submit_button(self):
        """Falls back to pressing Enter when no submit button is visible."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc

        call_count = 0

        async def conditional_visible(timeout=500):
            nonlocal call_count
            call_count += 1
            # First call succeeds (textbox), subsequent calls for submit buttons fail
            if call_count <= 1:
                return True
            raise PlaywrightTimeout("timeout")

        mock_loc.is_visible = AsyncMock(side_effect=conditional_visible)
        result = await submit_code(page, "AB3F9X")
        # Should still succeed — falls back to Enter key
        assert result is True

    @pytest.mark.asyncio
    async def test_submit_exception_returns_false(self):
        page = make_mock_page()
        page.locator = MagicMock(side_effect=Exception("broken"))
        result = await submit_code(page, "AB3F9X")
        assert result is False


class TestJsFallbackCode:
    @pytest.mark.asyncio
    async def test_js_stores_fallback_code(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value="fallback_code:XY9Z3K")
        action = Action(type="js", value="some_js()")
        result = await run(page, action)
        assert result is True
        assert action._found_code == "XY9Z3K"

    @pytest.mark.asyncio
    async def test_js_no_fallback_code(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value="some other result")
        action = Action(type="js", value="some_js()")
        await run(page, action)
        assert not hasattr(action, "_found_code") or action._found_code is None


class TestClickFallbacks:
    @pytest.mark.asyncio
    async def test_click_falls_back_to_get_by_role(self):
        """When locator finds no visible element, try get_by_role."""
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.count = AsyncMock(return_value=1)
        mock_loc.nth = MagicMock(return_value=mock_loc)
        mock_loc.is_visible = AsyncMock(return_value=False)

        role_loc = MagicMock()
        role_loc.first = role_loc
        role_loc.click = AsyncMock()
        page.get_by_role = MagicMock(return_value=role_loc)

        action = Action(type="click", selector="button:has-text('Reveal')")
        result = await run(page, action)
        assert result is True
        page.get_by_role.assert_called_with("button", name="Reveal")

    @pytest.mark.asyncio
    async def test_click_falls_back_to_get_by_text(self):
        """When get_by_role fails, try get_by_text."""
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.count = AsyncMock(return_value=1)
        mock_loc.nth = MagicMock(return_value=mock_loc)
        mock_loc.is_visible = AsyncMock(return_value=False)

        role_loc = MagicMock()
        role_loc.first = role_loc
        role_loc.click = AsyncMock(side_effect=Exception("no role match"))
        page.get_by_role = MagicMock(return_value=role_loc)

        text_loc = MagicMock()
        text_loc.first = text_loc
        text_loc.click = AsyncMock()
        page.get_by_text = MagicMock(return_value=text_loc)

        action = Action(type="click", selector="button:has-text('Reveal')")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_click_no_has_text_returns_false(self):
        """When selector has no has-text pattern and all locators fail."""
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.count = AsyncMock(return_value=0)
        mock_loc.nth = MagicMock(return_value=mock_loc)

        action = Action(type="click", selector=".some-class")
        result = await run(page, action)
        assert result is False


class TestFillAction:
    @pytest.mark.asyncio
    async def test_fill_tries_fallback_selectors(self):
        """When primary selector fails, tries input[type='text'] etc."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = make_mock_page()
        call_count = 0

        async def conditional_visible(timeout=500):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise PlaywrightTimeout("timeout")
            return True

        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(side_effect=conditional_visible)

        action = Action(type="fill", selector=".custom-input", value="hello")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_fill_no_selector_tries_defaults(self):
        page = make_mock_page()
        action = Action(type="fill", value="hello")
        result = await run(page, action)
        assert result is True


class TestHoverAction:
    @pytest.mark.asyncio
    async def test_hover_tries_text_selectors(self):
        """When primary selector fails, tries text='Hover here'."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = make_mock_page()
        call_count = 0

        async def conditional_visible(timeout=500):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise PlaywrightTimeout("timeout")
            return True

        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(side_effect=conditional_visible)

        action = Action(type="hover", selector=".custom-hover")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_hover_all_fail(self):
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(side_effect=PlaywrightTimeout("timeout"))

        action = Action(type="hover", selector=".nonexistent")
        result = await run(page, action)
        assert result is False


class TestDragFillAction:
    @pytest.mark.asyncio
    async def test_drag_fill_js_succeeds(self):
        """JS strategy fills all slots."""
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=6)
        action = Action(type="drag_fill")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_drag_fill_js_fails_playwright_succeeds(self):
        """JS returns <6, Playwright drag_to fills rest."""
        page = make_mock_page()
        eval_calls = 0

        async def mock_eval(script):
            nonlocal eval_calls
            eval_calls += 1
            if eval_calls <= 1:
                return 3  # JS: only 3 filled
            return 6  # After Playwright: all filled

        page.evaluate = AsyncMock(side_effect=mock_eval)
        mock_loc = page.locator.return_value
        mock_loc.count = AsyncMock(return_value=3)
        mock_loc.nth = MagicMock(return_value=mock_loc)
        mock_loc.is_visible = AsyncMock(return_value=True)
        mock_loc.drag_to = AsyncMock()

        action = Action(type="drag_fill")
        result = await run(page, action)
        assert result is True

    @pytest.mark.asyncio
    async def test_drag_fill_mouse_simulation(self):
        """Falls back to mouse simulation when other methods fail."""
        page = make_mock_page()
        eval_calls = 0

        async def mock_eval(script):
            nonlocal eval_calls
            eval_calls += 1
            if eval_calls <= 1:
                return 0  # JS: none filled
            if eval_calls <= 3:
                return 0  # After Playwright attempts
            # Mouse layout query
            if "pieces" in str(script) and "slots" in str(script):
                return {
                    "pieces": [{"x": 100, "y": 100}],
                    "slots": [{"x": 300, "y": 300}],
                }
            return 6  # Final check

        page.evaluate = AsyncMock(side_effect=mock_eval)
        mock_loc = page.locator.return_value
        mock_loc.count = AsyncMock(return_value=0)
        page.mouse = MagicMock()
        page.mouse.move = AsyncMock()
        page.mouse.down = AsyncMock()
        page.mouse.up = AsyncMock()

        action = Action(type="drag_fill")
        result = await run(page, action)
        # We just check it doesn't crash; actual result depends on mock eval sequence
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_drag_fill_exception(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("broken"))
        action = Action(type="drag_fill")
        result = await run(page, action)
        assert result is False


class TestScrollElementAction:
    @pytest.mark.asyncio
    async def test_scroll_element_found(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value={"x": 100, "y": 200})
        page.mouse = MagicMock()
        page.mouse.move = AsyncMock()
        page.mouse.wheel = AsyncMock()

        action = Action(type="scroll_element")
        result = await run(page, action)
        assert result is True
        page.mouse.wheel.assert_called()

    @pytest.mark.asyncio
    async def test_scroll_element_not_found(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(return_value=None)

        action = Action(type="scroll_element")
        result = await run(page, action)
        assert result is False

    @pytest.mark.asyncio
    async def test_scroll_element_exception(self):
        page = make_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("broken"))

        action = Action(type="scroll_element")
        result = await run(page, action)
        assert result is False


class TestCanvasDrawAction:
    @pytest.mark.asyncio
    async def test_canvas_draw_success(self):
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(return_value=True)
        mock_loc.bounding_box = AsyncMock(
            return_value={"x": 50, "y": 50, "width": 400, "height": 300}
        )
        page.mouse = MagicMock()
        page.mouse.move = AsyncMock()
        page.mouse.down = AsyncMock()
        page.mouse.up = AsyncMock()

        action = Action(type="canvas_draw")
        result = await run(page, action)
        assert result is True
        assert page.mouse.down.call_count == 5  # 5 strokes

    @pytest.mark.asyncio
    async def test_canvas_draw_not_visible(self):
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(return_value=False)

        action = Action(type="canvas_draw")
        result = await run(page, action)
        assert result is False

    @pytest.mark.asyncio
    async def test_canvas_draw_no_bounding_box(self):
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(return_value=True)
        mock_loc.bounding_box = AsyncMock(return_value=None)

        action = Action(type="canvas_draw")
        result = await run(page, action)
        assert result is False

    @pytest.mark.asyncio
    async def test_canvas_draw_exception(self):
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.is_visible = AsyncMock(side_effect=Exception("broken"))

        action = Action(type="canvas_draw")
        result = await run(page, action)
        assert result is False


class TestKeySequenceAction:
    @pytest.mark.asyncio
    async def test_key_sequence_reads_from_page(self):
        page = make_mock_page(
            inner_text="Keyboard Sequence Challenge\nRequired sequence: ArrowUp ArrowDown\nPress keys"
        )
        action = Action(type="key_sequence")
        result = await run(page, action)
        assert result is True
        assert page.keyboard.press.call_count == 2

    @pytest.mark.asyncio
    async def test_key_sequence_no_keys_found(self):
        page = make_mock_page(inner_text="Nothing useful here")
        action = Action(type="key_sequence")
        result = await run(page, action)
        assert result is False

    @pytest.mark.asyncio
    async def test_key_sequence_with_arrows_in_text(self):
        page = make_mock_page(inner_text="Required sequence:\n↑\n↓\n→")
        action = Action(type="key_sequence")
        result = await run(page, action)
        assert result is True
        assert page.keyboard.press.call_count == 3

    @pytest.mark.asyncio
    async def test_key_sequence_exception(self):
        page = make_mock_page()
        page.inner_text = AsyncMock(side_effect=Exception("broken"))
        action = Action(type="key_sequence")
        result = await run(page, action)
        assert result is False


class TestExtractKeySequence:
    def test_basic_sequence(self):
        text = "Keyboard Challenge\nRequired sequence: ArrowUp ArrowDown ArrowLeft"
        keys = _extract_key_sequence(text)
        assert keys == ["ArrowUp", "ArrowDown", "ArrowLeft"]

    def test_sequence_with_arrows(self):
        text = "Sequence: ↑ ↓ → ←"
        keys = _extract_key_sequence(text)
        assert "ArrowUp" in keys
        assert "ArrowDown" in keys

    def test_multiline_sequence(self):
        text = "Required sequence:\nArrowUp\nArrowDown\nEnter"
        keys = _extract_key_sequence(text)
        assert len(keys) == 3

    def test_no_sequence(self):
        assert _extract_key_sequence("Nothing here") == []

    def test_empty_text(self):
        assert _extract_key_sequence("") == []


class TestParseKeyTokens:
    def test_arrow_symbols(self):
        keys = _parse_key_tokens("↑ ↓")
        assert "ArrowUp" in keys
        assert "ArrowDown" in keys

    def test_word_tokens(self):
        keys = _parse_key_tokens("ArrowUp ArrowDown Enter")
        assert keys == ["ArrowUp", "ArrowDown", "Enter"]

    def test_lowercase_mapped(self):
        keys = _parse_key_tokens("up down left right")
        assert keys == ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]

    def test_single_letters(self):
        keys = _parse_key_tokens("a b c")
        assert keys == ["a", "b", "c"]

    def test_empty(self):
        assert _parse_key_tokens("") == []

    def test_quoted_tokens(self):
        keys = _parse_key_tokens("'enter' 'space'")
        assert keys == ["Enter", "Space"]


class TestSelectActionFail:
    @pytest.mark.asyncio
    async def test_select_exception(self):
        page = make_mock_page()
        mock_loc = page.locator.return_value
        mock_loc.first = mock_loc
        mock_loc.select_option = AsyncMock(side_effect=Exception("no select"))
        action = Action(type="select", value="Option X")
        result = await run(page, action)
        assert result is False


class TestActionException:
    @pytest.mark.asyncio
    async def test_top_level_exception_returns_false(self):
        """Any exception in the action handler is caught and returns False."""
        page = make_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("broken"))
        action = Action(type="scroll", amount=100)
        result = await run(page, action)
        assert result is False
