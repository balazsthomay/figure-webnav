"""Tests for solver module (LLM interface)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webnav.perception import PageState
from webnav.solver import Solver, _parse_actions


class TestParseActions:
    def test_parses_click_action(self):
        raw = '[{"type": "click", "selector": "button:has-text(\'Reveal\')"}]'
        actions = _parse_actions(raw)
        assert len(actions) == 1
        assert actions[0].type == "click"
        assert "Reveal" in actions[0].selector

    def test_parses_scroll_action(self):
        raw = '[{"type": "scroll", "amount": 500}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "scroll"
        assert actions[0].amount == 500

    def test_parses_multiple_actions(self):
        raw = '[{"type": "click", "selector": "btn"}, {"type": "wait", "amount": 2}]'
        actions = _parse_actions(raw)
        assert len(actions) == 2

    def test_strips_code_fences(self):
        raw = '```json\n[{"type": "click", "selector": "btn"}]\n```'
        actions = _parse_actions(raw)
        assert len(actions) == 1

    def test_handles_invalid_json(self):
        actions = _parse_actions("not json at all")
        assert actions == []

    def test_handles_empty(self):
        actions = _parse_actions("[]")
        assert actions == []

    def test_filters_invalid_types(self):
        raw = '[{"type": "destroy", "selector": "everything"}]'
        actions = _parse_actions(raw)
        assert actions == []

    def test_wraps_single_object(self):
        raw = '{"type": "click", "selector": "btn"}'
        actions = _parse_actions(raw)
        assert len(actions) == 1

    def test_handles_js_action(self):
        raw = '[{"type": "js", "value": "alert(1)"}]'
        actions = _parse_actions(raw)
        assert actions[0].type == "js"
        assert actions[0].value == "alert(1)"


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
        mock_response.choices[0].message.content = '[{"type": "click", "selector": "btn"}]'
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        solver._client.chat.completions.create = AsyncMock(return_value=mock_response)

        state = PageState(step=1, instruction="Click the button")
        actions = await solver.solve(state)

        assert len(actions) == 1
        assert actions[0].type == "click"
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
        mock_response.choices[0].message.content = '[{"type": "js", "value": "reveal()"}]'
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
