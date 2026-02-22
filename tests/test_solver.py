"""Tests for solver module (LLM interface)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webnav.perception import PageState, ElementInfo
from webnav.solver import Solver, _parse_actions, _instruction_hints


class TestParseActions:
    def test_parses_click_with_element_index(self):
        raw = '[{"action": "click", "element": 3}]'
        actions = _parse_actions(raw)
        assert len(actions) == 1
        assert actions[0].type == "click"
        assert actions[0].element == 3

    def test_parses_fill_with_element_and_text(self):
        raw = '[{"action": "fill", "element": 7, "text": "hello"}]'
        actions = _parse_actions(raw)
        assert len(actions) == 1
        assert actions[0].type == "fill"
        assert actions[0].element == 7
        assert actions[0].value == "hello"

    def test_parses_scroll_action(self):
        raw = '[{"action": "scroll", "amount": 500}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "scroll"
        assert actions[0].amount == 500

    def test_parses_hover_with_duration(self):
        raw = '[{"action": "hover", "element": 5, "duration": 3.5}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "hover"
        assert actions[0].element == 5
        assert actions[0].duration == 3.5

    def test_parses_drag_with_elements(self):
        raw = '[{"action": "drag", "from_element": 2, "to_element": 8}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "drag"
        # from_element is parsed as "element" field via item.get("element")
        # to_element is parsed directly
        assert actions[0].to_element == 8

    def test_parses_wait_with_seconds(self):
        raw = '[{"action": "wait", "seconds": 5}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "wait"
        assert actions[0].amount == 5

    def test_parses_key_sequence(self):
        raw = '[{"action": "key_sequence", "keys": "ArrowUp ArrowDown"}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "key_sequence"
        assert actions[0].value == "ArrowUp ArrowDown"

    def test_parses_multiple_actions(self):
        raw = '[{"action": "click", "element": 0}, {"action": "scroll", "amount": 600}]'
        actions = _parse_actions(raw)
        assert len(actions) == 2

    def test_strips_code_fences(self):
        raw = '```json\n[{"action": "click", "element": 0}]\n```'
        actions = _parse_actions(raw)
        assert len(actions) == 1

    def test_handles_invalid_json(self):
        actions = _parse_actions("not json at all")
        assert actions == []

    def test_handles_empty(self):
        actions = _parse_actions("[]")
        assert actions == []

    def test_filters_invalid_types(self):
        raw = '[{"action": "destroy", "element": 0}]'
        actions = _parse_actions(raw)
        assert actions == []

    def test_wraps_single_object(self):
        raw = '{"action": "click", "element": 0}'
        actions = _parse_actions(raw)
        assert len(actions) == 1

    def test_handles_js_action(self):
        raw = '[{"action": "js", "code": "document.title"}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "js"
        assert actions[0].value == "document.title"

    # Backward compat: "type" key still works
    def test_parses_type_key_click(self):
        raw = '[{"type": "click", "selector": "button:has-text(\'Reveal\')"}]'
        actions = _parse_actions(raw)
        assert len(actions) == 1
        assert actions[0].type == "click"
        assert "Reveal" in actions[0].selector

    def test_normalizes_canvas_draw(self):
        raw = '[{"action": "canvas_draw", "element": 4}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "draw_strokes"

    def test_normalizes_drag_fill(self):
        raw = '[{"action": "drag_fill"}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "drag"

    def test_parses_click_with_amount(self):
        raw = '[{"action": "click", "element": 3, "amount": 5}]'
        actions = _parse_actions(raw)
        assert actions[0].element == 3
        assert actions[0].amount == 5

    def test_parses_press_with_key(self):
        raw = '[{"action": "press", "key": "Enter"}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "press"
        assert actions[0].value == "Enter"

    def test_value_extraction_priority(self):
        """text > value > code > key > keys for value field."""
        raw = '[{"action": "fill", "element": 1, "text": "primary", "value": "secondary"}]'
        actions = _parse_actions(raw)
        assert actions[0].value == "primary"

    def test_draw_strokes(self):
        raw = '[{"action": "draw_strokes", "element": 4}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "draw_strokes"
        assert actions[0].element == 4


class TestInstructionHints:
    """Test that _instruction_hints generates appropriate hints."""

    def test_terminal_prompt_gets_hint(self):
        """Dollar-sign prompt instruction gets terminal hint."""
        hints = _instruction_hints("$ awaiting connection...")
        assert "wait" in hints.lower()
        assert "terminal" in hints.lower() or "animation" in hints.lower()

    def test_awaiting_gets_hint(self):
        """'awaiting' instruction gets terminal hint."""
        hints = _instruction_hints("awaiting connection...")
        assert "wait" in hints.lower()

    def test_hover_gets_hint(self):
        """Hover instruction gets hover hint (existing behavior)."""
        hints = _instruction_hints("Hover over the box")
        assert "hover" in hints.lower()

    def test_normal_instruction_no_terminal_hint(self):
        """Normal instruction doesn't get terminal hint."""
        hints = _instruction_hints("Click the button below")
        assert "terminal" not in hints.lower()
        assert "awaiting" not in hints.lower()


class TestSolverInit:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_creates_client(self):
        solver = Solver()
        assert solver._client is not None
        assert solver.total_tokens == 0
        assert solver.total_calls == 0


class TestSolverSolve:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_calls_llm(self):
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"action": "click", "element": 0}]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=1, instruction="Click the button")
        actions = await solver.solve(state)

        assert len(actions) == 1
        assert actions[0].type == "click"
        assert actions[0].element == 0
        assert solver.total_calls == 1
        assert solver.total_tokens == 100

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_handles_failure(self):
        solver = Solver()
        solver._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        state = PageState(step=1, instruction="Click the button")
        actions = await solver.solve(state)
        assert actions == []

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_stuck_with_screenshot(self):
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"action": "js", "code": "reveal()"}]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 200

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=5, instruction="Find the hidden code")
        actions = await solver.solve_stuck(state, screenshot=b"fake_png")

        assert len(actions) == 1
        assert actions[0].type == "js"
        # Verify image was included in the call
        call_kwargs = solver._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
        user_msg = messages[-1]
        assert any(c.get("type") == "image_url" for c in user_msg["content"])

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_passes_temperature(self):
        """Verify temperature parameter is threaded through to API call."""
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=1, instruction="Click")
        await solver.solve(state, temperature=0.3)

        call_kwargs = solver._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_stuck_passes_temperature(self):
        """Verify temperature parameter is threaded through solve_stuck."""
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=1, instruction="Find code")
        await solver.solve_stuck(state, temperature=0.5)

        call_kwargs = solver._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_default_temperature_zero(self):
        """Default temperature should be 0.0."""
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=1, instruction="Click")
        await solver.solve(state)

        call_kwargs = solver._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_sends_page_state_prompt(self):
        """Verify that solve() sends the page state prompt to the LLM."""
        solver = Solver()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        elements = [
            ElementInfo(index=0, tag="button", name="Reveal Code", selector="button:nth-of-type(1)"),
        ]
        state = PageState(step=1, instruction="Click the button", elements=elements)
        await solver.solve(state)

        call_kwargs = solver._client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "STEP: 1/30" in user_msg
        assert "[0] button" in user_msg
        assert "Reveal Code" in user_msg
