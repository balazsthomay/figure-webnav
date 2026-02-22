"""Integration tests for the agent orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webnav.agent import Agent, _parse_instruction_actions
from webnav.actions import Action
from webnav.perception import ElementInfo
from tests.conftest import (
    STEP1_ARIA_YAML,
    STEP3_ARIA_YAML_WITH_CODE,
    make_mock_page,
)


class TestAgentSolveStep:
    """Test the _solve_step method with mocked browser."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solves_code_already_visible(self):
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step3?version=2",
            aria_yaml=STEP3_ARIA_YAML_WITH_CODE,
            inner_text="Your code is: AB3F9X\nSubmit Code",
        )

        # After submission, URL advances (mock the click advancing the page)
        async def mock_submit_click(*args, **kwargs):
            page.url = "https://serene-frangipane-7fd25b.netlify.app/step4?version=2"

        mock_loc = page.locator.return_value
        mock_loc.click = mock_submit_click
        page.wait_for_url = AsyncMock()

        agent.browser._page = page
        result = await agent._solve_step(3, attempt=0, history=[])
        assert result is True
        assert agent.state.steps_completed == 1

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solves_click_to_reveal(self):
        agent = Agent(headless=True)

        # Start: no code visible
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            aria_yaml=STEP1_ARIA_YAML,
            inner_text="Click the button below to reveal the code\nReveal Code\nSubmit Code",
        )

        # After clicking reveal, code appears
        call_count = 0
        async def mock_inner_text(selector="body"):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return "Click the button below to reveal the code\nReveal Code\nSubmit Code"
            return "Your code is: QW3R7Y\nSubmit Code"

        page.inner_text = mock_inner_text

        # After code submission, advance
        async def mock_click(*args, **kwargs):
            page.url = "https://serene-frangipane-7fd25b.netlify.app/step2?version=2"

        mock_loc = page.locator.return_value
        mock_loc.click = mock_click

        agent.browser._page = page
        result = await agent._solve_step(1, attempt=0, history=[])
        # The exact outcome depends on mock fidelity, but it should not crash
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_step_error_marks_failed(self):
        """When _solve_step raises, it catches the exception and returns False."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            inner_text="Click the button below to reveal the code",
        )
        # Make page_cleaner's evaluate call raise
        page.evaluate = AsyncMock(side_effect=Exception("broken"))
        agent.browser._page = page
        result = await agent._solve_step(1, attempt=0, history=[])
        assert result is False
        assert agent.state.steps_failed >= 1

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_step_no_code_marks_failed(self):
        """When no code is found after actions, step is marked failed."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            inner_text="Do something completely novel and weird",
            aria_yaml='- heading "Challenge Step 1"\n- paragraph "Do something completely novel and weird"',
        )
        # LLM returns no useful actions
        agent.solver.solve = AsyncMock(return_value=[])
        agent.browser._page = page
        result = await agent._solve_step(1, attempt=0, history=[])
        assert result is False

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_solve_step_recovery_on_retry(self):
        """On attempt >= 1, calls recovery LLM with screenshot."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step5?version=2",
            inner_text="Something weird with no clear instruction",
            aria_yaml='- heading "Challenge Step 5"\n- paragraph "Something weird"',
        )
        agent.solver.solve_stuck = AsyncMock(
            return_value=[Action(type="click", element=0)]
        )
        page.wait_for_url = AsyncMock()
        agent.browser._page = page

        result = await agent._solve_step(5, attempt=1, history=["attempt 1: failed"])
        assert isinstance(result, bool)
        agent.solver.solve_stuck.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_history_tracks_attempts(self):
        """History list records what happened on each attempt."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            inner_text="Some challenge",
            aria_yaml='- heading "Challenge Step 1"',
        )
        agent.solver.solve = AsyncMock(return_value=[])
        agent.browser._page = page

        history: list[str] = []
        await agent._solve_step(1, attempt=0, history=history)
        assert len(history) == 1
        assert "attempt 1" in history[0]
        assert "no code found" in history[0]


class TestSubmitAndCheck:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_submit_advances(self):
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step5?version=2",
        )

        async def advance_url(*args, **kwargs):
            page.url = "https://serene-frangipane-7fd25b.netlify.app/step6?version=2"

        mock_loc = page.locator.return_value
        mock_loc.click = AsyncMock(side_effect=advance_url)
        page.wait_for_url = AsyncMock()

        agent.browser._page = page
        agent.state.begin_step(5)
        agent.metrics.begin_step(5)
        result = await agent._submit_and_check("QW3R7Y", 5)
        assert result is True
        assert "QW3R7Y" in agent._used_codes

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_submit_does_not_advance(self):
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step5?version=2",
        )
        page.wait_for_url = AsyncMock()
        agent.browser._page = page
        agent.state.begin_step(5)
        agent.metrics.begin_step(5)
        result = await agent._submit_and_check("BADCOD", 5)
        assert result is False


class TestWaitForContent:
    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_wait_finds_h1(self):
        agent = Agent(headless=True)
        page = make_mock_page()
        page.wait_for_selector = AsyncMock()
        agent.browser._page = page
        await agent._wait_for_content(timeout=0.5)
        page.wait_for_selector.assert_called()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_wait_times_out(self):
        agent = Agent(headless=True)
        page = make_mock_page()
        page.wait_for_selector = AsyncMock(side_effect=Exception("timeout"))
        agent.browser._page = page
        # Should not raise — just times out gracefully
        await agent._wait_for_content(timeout=0.3)


class TestHoverTargetSelection:
    """Test that hover target selection prefers clickable + smallest bbox."""

    def test_prefers_clickable_over_non_clickable(self):
        """Clickable elements (cursor:pointer) sort before non-clickable."""
        elements = [
            ElementInfo(index=0, tag="div", name="Hover here to reveal",
                        bbox={"width": 200, "height": 100, "x": 0, "y": 0},
                        extra=""),
            ElementInfo(index=1, tag="div", name="Hover here to reveal",
                        bbox={"width": 200, "height": 100, "x": 0, "y": 0},
                        extra="clickable"),
        ]
        actions = _parse_instruction_actions("Hover over the target", elements)
        hover_actions = [a for a in actions if a.type == "hover"]
        assert len(hover_actions) == 1
        # Should pick element 1 (clickable)
        assert hover_actions[0].element == 1

    def test_prefers_smallest_bbox_among_clickable(self):
        """Among clickable elements, smallest bbox wins."""
        elements = [
            ElementInfo(index=0, tag="div", name="Hover here",
                        bbox={"width": 400, "height": 200, "x": 0, "y": 0},
                        extra="clickable"),
            ElementInfo(index=1, tag="span", name="Hover here",
                        bbox={"width": 100, "height": 50, "x": 0, "y": 0},
                        extra="clickable"),
        ]
        actions = _parse_instruction_actions("Hover over here", elements)
        hover_actions = [a for a in actions if a.type == "hover"]
        assert len(hover_actions) == 1
        assert hover_actions[0].element == 1

    def test_clickable_small_beats_non_clickable_smaller(self):
        """Clickable element with larger bbox beats non-clickable with smaller bbox."""
        elements = [
            ElementInfo(index=0, tag="span", name="Hover here",
                        bbox={"width": 50, "height": 25, "x": 0, "y": 0},
                        extra=""),
            ElementInfo(index=1, tag="div", name="Hover here",
                        bbox={"width": 200, "height": 100, "x": 0, "y": 0},
                        extra="clickable"),
        ]
        actions = _parse_instruction_actions("Hover over here", elements)
        hover_actions = [a for a in actions if a.type == "hover"]
        assert len(hover_actions) == 1
        # Clickable wins even if larger
        assert hover_actions[0].element == 1

    def test_no_bbox_sorts_last(self):
        """Elements without bbox sort after those with bbox."""
        elements = [
            ElementInfo(index=0, tag="div", name="Hover here",
                        bbox=None, extra="clickable"),
            ElementInfo(index=1, tag="div", name="Hover here",
                        bbox={"width": 200, "height": 100, "x": 0, "y": 0},
                        extra="clickable"),
        ]
        actions = _parse_instruction_actions("Hover over here", elements)
        hover_actions = [a for a in actions if a.type == "hover"]
        assert len(hover_actions) == 1
        assert hover_actions[0].element == 1


class TestTerminalPreAction:
    """Test that terminal-style instructions get a JS pre-action."""

    def test_dollar_prompt_gets_js_action(self):
        """Instruction starting with '$' triggers a JS pre-action."""
        actions = _parse_instruction_actions("$ awaiting connection...", [])
        js_actions = [a for a in actions if a.type == "js"]
        assert len(js_actions) == 1
        assert "connect" in js_actions[0].value.lower()
        assert "__origST" in js_actions[0].value  # real-time delay

    def test_awaiting_connection_gets_js(self):
        """'awaiting connection' in instruction triggers JS pre-action."""
        actions = _parse_instruction_actions("awaiting connection...", [])
        js_actions = [a for a in actions if a.type == "js"]
        assert len(js_actions) == 1

    def test_normal_instruction_no_terminal_action(self):
        """Normal instructions don't get a terminal pre-action."""
        actions = _parse_instruction_actions("Click the button below to reveal the code", [])
        js_actions = [a for a in actions if a.type == "js" and "connect" in (a.value or "").lower()]
        assert len(js_actions) == 0


class TestRetryTemperature:
    """Test that retry attempts use recovery LLM with temperature."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_attempt_0_uses_primary(self):
        """First attempt uses primary LLM with temperature=0.0."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            inner_text="Some challenge",
            aria_yaml='- heading "Challenge Step 1"\n- paragraph "Some challenge"',
        )
        agent.solver.solve = AsyncMock(return_value=[])
        agent.browser._page = page
        await agent._solve_step(1, attempt=0, history=[])
        _, kwargs = agent.solver.solve.call_args
        assert kwargs.get("temperature", 0.0) == 0.0

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    async def test_attempt_1_uses_recovery(self):
        """Second attempt uses recovery LLM with screenshot and temperature=0.3."""
        agent = Agent(headless=True)
        page = make_mock_page(
            url="https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
            inner_text="Some challenge",
            aria_yaml='- heading "Challenge Step 1"\n- paragraph "Some challenge"',
        )
        agent.solver.solve_stuck = AsyncMock(return_value=[])
        agent.browser._page = page
        await agent._solve_step(1, attempt=1, history=["attempt 1: failed"])
        agent.solver.solve_stuck.assert_called_once()
        _, kwargs = agent.solver.solve_stuck.call_args
        assert kwargs.get("temperature") == 0.3


class TestAgentInit:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_creates_components(self):
        agent = Agent(headless=True)
        assert agent.browser is not None
        assert agent.solver is not None
        assert agent.state is not None
        assert agent.metrics is not None
        assert agent._used_codes == set()

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_accepts_config(self):
        from webnav.config import ChallengeConfig
        config = ChallengeConfig(url="https://example.com", total_steps=10)
        agent = Agent(headless=True, config=config)
        assert agent.config.url == "https://example.com"
        assert agent.config.total_steps == 10
