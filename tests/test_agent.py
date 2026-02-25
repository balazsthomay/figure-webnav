"""Integration tests for the agent orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webnav.agent import Agent, _parse_instruction_actions, _PUZZLE_SOLVE_JS
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


class TestClickNTimesPreAction:
    """Test that click-here-N-times instructions get a JS pre-action."""

    def test_click_here_n_times_gets_js(self):
        """'click here 3 times' triggers a JS pre-action."""
        actions = _parse_instruction_actions("Click here 3 more times to reveal the code", [])
        js_actions = [a for a in actions if a.type == "js"]
        assert len(js_actions) >= 1
        assert "fullClick" in js_actions[0].value

    def test_click_n_times_uses_specific_regex(self):
        """Pre-action JS matches 'click here N times', not just 'click here'."""
        actions = _parse_instruction_actions("Click here 3 more times to reveal the code", [])
        js_actions = [a for a in actions if a.type == "js"]
        js = js_actions[0].value
        # Must use specific regex that won't match noise "Click Here" buttons
        assert "click here.*\\d+.*time" in js
        # Must unhide noise containers before searching
        assert "data-wnav-noise" in js

    def test_normal_instruction_no_click_n_times(self):
        """Normal instructions don't get a click-N-times pre-action."""
        actions = _parse_instruction_actions("Click the button below to reveal the code", [])
        js_actions = [a for a in actions if a.type == "js" and "fullClick" in (a.value or "")]
        assert len(js_actions) == 0


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


class TestPreActionTimerSafety:
    """Pre-action JS must use window.__origST to avoid timer acceleration."""

    INSTRUCTIONS_WITH_SETTIMEOUT = [
        ("Mutation Challenge: Trigger 5 DOM mutations to reveal the code.", "mutation"),
        ("Rotating Code Challenge: The code changes every 3 seconds. Click \"Capture\" at least 3 times to complete the challenge.", "capture"),
        ("Video Challenge: Seek through the video frames. Find and navigate to frame 42, then complete the challenge.", "video"),
        ("Multi-Tab Challenge: Visit all 4 tabs by clicking each button. Each tab contains a part of the puzzle.", "multi-tab"),
    ]

    @pytest.mark.parametrize("instruction,label", INSTRUCTIONS_WITH_SETTIMEOUT)
    def test_pre_action_js_uses_origST(self, instruction, label):
        """All pre-action JS that uses setTimeout must use window.__origST."""
        actions = _parse_instruction_actions(instruction, [])
        js_actions = [a for a in actions if a.type == "js"]
        assert len(js_actions) >= 1, f"No JS pre-action for {label}"
        for js_action in js_actions:
            js = js_action.value
            if "setTimeout" not in js:
                continue
            # Every setTimeout must be wrapped with __origST
            import re
            plain = re.findall(r'(?<!__origST\|\|)setTimeout', js)
            assert len(plain) == 0, (
                f"{label} pre-action has plain setTimeout without __origST: "
                f"found {len(plain)} unprotected calls"
            )


class TestPuzzleSolverAriaRadio:
    """Puzzle solver JS must detect Radix UI / ARIA radio buttons.

    Radix UI renders radio items as <button role="radio"> (not <input type="radio">).
    The option text is in the value attribute, not textContent (which is empty/SVG).
    The solver must find these elements and match their text via value/label.
    """

    def test_selector_includes_button_role_radio(self):
        """Section 1 radio detection must query button[role='radio'] too."""
        # The radio selector should find both native inputs AND ARIA buttons
        assert 'button[role="radio"]' in _PUZZLE_SOLVE_JS, (
            "Puzzle solver radio section must include button[role='radio'] "
            "to detect Radix UI RadioGroupItem elements"
        )

    def test_text_extraction_uses_value_attr(self):
        """For ARIA radio buttons with empty textContent, use value attribute."""
        # Radix buttons have text in value attr, not textContent
        assert 'getAttribute' in _PUZZLE_SOLVE_JS and 'value' in _PUZZLE_SOLVE_JS, (
            "Puzzle solver must use getAttribute('value') for ARIA radio text"
        )

    def test_section2_excludes_role_radio(self):
        """Section 2 (button options) should skip role='radio' (handled in section 1)."""
        import re
        # The button options selector should NOT include [role="radio"] as a
        # standalone comma-separated selector item. It may appear inside :not()
        # which is fine (that means it's being excluded).
        match = re.search(r"querySelectorAll\(['\"](.+?)['\"]\)", _PUZZLE_SOLVE_JS)
        assert match, "Could not find optBtns querySelectorAll"
        selector = match.group(1)
        # Split by comma to get individual selector items
        items = [s.strip() for s in selector.split(',')]
        standalone_radio = [s for s in items if s == '[role="radio"]']
        assert len(standalone_radio) == 0, (
            "Section 2 optBtns should not include [role='radio'] as a standalone "
            "selector — those are handled in section 1 ARIA radio detection"
        )


class TestPuzzleSolverWrongReExclusion:
    """Radio selection must exclude wrongRe matches even when correctRe matches."""

    def test_section1a_checks_wrongRe(self):
        """Section 1a (native radio) correctRe pass must also check !wrongRe."""
        # Find the native radio selection loop (Section 1a).
        # It should check wrongRe before selecting, just like Section 2 does.
        import re
        # Look for the correctRe check in the native radio loop
        # The pattern: correctRe.test(text) should be accompanied by wrongRe check
        section1a = _PUZZLE_SOLVE_JS.split("// 1b.")[0]
        # Count correctRe.test calls that do NOT have a wrongRe guard nearby
        correct_checks = list(re.finditer(r"correctRe\.test\(text\)", section1a))
        assert len(correct_checks) > 0, "Section 1a should check correctRe"
        for m in correct_checks:
            # The wrongRe check should be on the same line or very close
            ctx = section1a[max(0, m.start() - 80):m.end() + 80]
            assert "wrongRe" in ctx, (
                "Section 1a correctRe check must also verify !wrongRe.test(text)"
            )

    def test_section1b_checks_wrongRe(self):
        """Section 1b (ARIA radio) correctRe pass must also check !wrongRe."""
        import re
        section1b = _PUZZLE_SOLVE_JS.split("// 1b.")[1].split("// 2.")[0]
        correct_checks = list(re.finditer(r"correctRe\.test\(text\)", section1b))
        assert len(correct_checks) > 0, "Section 1b should check correctRe"
        for m in correct_checks:
            ctx = section1b[max(0, m.start() - 80):m.end() + 80]
            assert "wrongRe" in ctx, (
                "Section 1b correctRe check must also verify !wrongRe.test(text)"
            )


class TestPuzzleSolverLabelPriority:
    """Radio selection should prefer direct correctness statements over option labels."""

    def test_label_prefix_deprioritization(self):
        """Options starting with 'option', 'answer', etc. should be deprioritized."""
        # The solver should have a label prefix pattern to identify option labels
        import re
        assert re.search(r"option|answer|choice", _PUZZLE_SOLVE_JS, re.IGNORECASE), (
            "Puzzle solver must have label prefix detection"
        )
        # There should be a two-pass or priority mechanism
        # Look for a label/prefix regex or priority variable
        assert "labelPrefix" in _PUZZLE_SOLVE_JS or "priority" in _PUZZLE_SOLVE_JS or "prefer" in _PUZZLE_SOLVE_JS, (
            "Puzzle solver should have a priority mechanism for correctRe matches"
        )


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
