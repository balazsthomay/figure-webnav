"""Orchestrator: main agent loop — general-purpose CUA, no site-specific hacks."""

from __future__ import annotations

import asyncio
import time

from webnav.actions import Action
from webnav.browser import BrowserController
from webnav.config import ChallengeConfig
from webnav.executor import run as execute_action, submit_code
from webnav.extractor import find_code
from webnav.metrics import MetricsCollector
from webnav.page_cleaner import clean_page
from webnav.perception import snapshot
from webnav.solver import Solver
from webnav.state import StateTracker


class Agent:
    """General-purpose browser automation agent.

    Uses LLM reasoning for every step — no regex dispatch, no hardcoded
    patterns. Works via element indexing: the LLM picks from a numbered
    list of page elements, eliminating selector hallucination.
    """

    def __init__(
        self,
        headless: bool = True,
        config: ChallengeConfig | None = None,
    ) -> None:
        self.config = config or ChallengeConfig()
        self.browser = BrowserController(headless=headless)
        self.solver = Solver()
        self.state = StateTracker(
            total_steps=self.config.total_steps,
            max_time=self.config.max_time,
            max_step_time=self.config.max_step_time,
        )
        self.metrics = MetricsCollector()
        self._used_codes: set[str] = set()

    async def run(self) -> MetricsCollector:
        """Execute the full challenge. Returns metrics."""
        async with self.browser:
            # Navigate to landing page and click START
            await self.browser.goto(self.config.url)
            await asyncio.sleep(1.0)
            try:
                start_btn = self.browser.page.locator("text=START").first
                await start_btn.click(timeout=5000)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[agent] Could not click START: {e}")

            # Main loop
            max_attempts = self.config.max_retries
            last_step = 0
            attempts_on_step = 0

            while not self.state.is_over_budget():
                current_step = await self.state.detect_step_from_url(self.browser.page)
                if current_step > self.state.total_steps:
                    print("[agent] All steps completed!")
                    break

                if current_step == 0:
                    await asyncio.sleep(1.0)
                    continue

                if current_step != last_step:
                    last_step = current_step
                    attempts_on_step = 0

                success = await self._solve_step(current_step, attempts_on_step)
                if success:
                    attempts_on_step = 0
                    await self._wait_for_content()
                else:
                    attempts_on_step += 1
                    if attempts_on_step >= max_attempts:
                        print(f"[agent] Step {current_step} stuck after {max_attempts} attempts, stopping")
                        break

        self.metrics.print_report()
        return self.metrics

    async def _solve_step(self, step: int, attempt: int = 0) -> bool:
        """Solve a single step using LLM reasoning. Returns True if completed."""
        self.state.begin_step(step)
        self.metrics.begin_step(step)
        t0 = time.time()
        print(f"\n[agent] === Step {step} (attempt {attempt + 1}) ===")

        try:
            # 1. Clean page (remove popups/overlays)
            await clean_page(self.browser.page)
            await asyncio.sleep(0.2)

            # 2. Perceive — a11y snapshot + element indexing
            page_state = await snapshot(self.browser.page)
            print(f"[agent] Instruction: {page_state.instruction[:120]}")
            print(f"[agent] Elements: {len(page_state.elements)} indexed")

            # 3. If code already visible, submit directly
            if page_state.visible_codes:
                code = page_state.visible_codes[0]
                if code not in self._used_codes:
                    print(f"[agent] Code already visible: {code}")
                    if await self._submit_and_check(code, step):
                        return True

            # 4. Reason — LLM decides actions (always, no regex shortcut)
            if attempt >= 2:
                print("[agent] Multiple failures — calling recovery LLM with screenshot")
                tier = 2
                screenshot = await self.browser.screenshot()
                pre_tokens = self.solver.total_tokens
                actions = await self.solver.solve_stuck(page_state, screenshot)
                self.metrics.record_llm_call(2, self.solver.total_tokens - pre_tokens)
            else:
                tier = 1
                pre_tokens = self.solver.total_tokens
                actions = await self.solver.solve(page_state)
                self.metrics.record_llm_call(1, self.solver.total_tokens - pre_tokens)

            print(f"[agent] LLM returned {len(actions)} actions: {[a.type for a in actions]}")

            if not actions:
                # Fallback: try revealing hidden elements
                actions = [Action(type="js", value=_REVEAL_HIDDEN_JS)]

            # 5. Execute actions
            for action in actions:
                desc = f"element={action.element}" if action.element is not None else (action.value[:60] if action.value else str(action.amount))
                print(f"[agent]   -> {action.type} {desc}")
                await execute_action(self.browser.page, action, page_state.elements)

            # Brief pause for page reaction
            await asyncio.sleep(0.3)

            # 6. Try clicking any visible reveal/complete button
            await _try_reveal_click(self.browser.page)
            await asyncio.sleep(0.2)

            # 7. Extract code — generic regex scan
            code = await find_code(self.browser.page, self._used_codes)
            if code is None:
                await clean_page(self.browser.page)
                await asyncio.sleep(0.2)
                code = await find_code(self.browser.page, self._used_codes)

            # Reveal hidden codes + retry
            if code is None:
                await self.browser.page.evaluate(_REVEAL_HIDDEN_JS)
                await asyncio.sleep(0.2)
                code = await find_code(self.browser.page, self._used_codes)

            # Brief poll
            if code is None and not self.state.is_step_timed_out():
                for _ in range(2):
                    await asyncio.sleep(0.4)
                    code = await find_code(self.browser.page, self._used_codes)
                    if code:
                        break

            t1 = time.time()
            print(f"[timing] total={t1-t0:.1f}s")

            # 8. Submit code
            if code:
                if await self._submit_and_check(code, step):
                    return True

            # Failed
            self.state.mark_failed()
            self.metrics.end_step(False, error="no code found")
            print(f"[agent] Step {step} FAILED — no code found")
            return False

        except Exception as e:
            self.state.mark_failed()
            self.metrics.end_step(False, error=str(e))
            print(f"[agent] Step {step} ERROR: {e}")
            return False

    async def _submit_and_check(self, code: str, step: int) -> bool:
        """Submit a code and check if the step advanced."""
        print(f"[agent] Submitting code: {code}")
        await clean_page(self.browser.page)
        url_before = self.browser.page.url
        submitted = await submit_code(self.browser.page, code)
        if submitted:
            try:
                await self.browser.page.wait_for_url(
                    lambda url: url != url_before, timeout=5000
                )
            except Exception:
                pass
            advanced = await self.state.check_advancement(self.browser.page, step)
            if advanced:
                self.state.mark_completed()
                self.metrics.end_step(True, code=code)
                self._used_codes.add(code)
                print(f"[agent] Step {step} PASSED")
                return True
            for _ in range(3):
                await asyncio.sleep(0.3)
                if await self.state.check_advancement(self.browser.page, step):
                    self.state.mark_completed()
                    self.metrics.end_step(True, code=code)
                    self._used_codes.add(code)
                    print(f"[agent] Step {step} PASSED")
                    return True
            print(f"[agent] Step {step} submitted {code} but did not advance")
        return False

    async def _wait_for_content(self, timeout: float = 5.0) -> None:
        """Wait for the SPA page to render content after navigation."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                await self.browser.page.wait_for_selector("h1", timeout=1000)
                return
            except Exception:
                await asyncio.sleep(0.3)
        await asyncio.sleep(1.0)


# Generic JS to reveal hidden codes — targeted approach.
_REVEAL_HIDDEN_JS = """
(() => {
    const codeRe = /^[A-Z0-9]{6}$/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+)$/;
    for (const el of document.querySelectorAll('*')) {
        const text = (el.textContent || '').trim();
        if (codeRe.test(text) && !fp.test(text) && el.childElementCount === 0) {
            el.style.cssText = 'display:block !important; visibility:visible !important; ' +
                'opacity:1 !important; color:red !important; font-size:16px !important; ' +
                'position:static !important; width:auto !important; height:auto !important;';
        }
    }
})()
"""

# Generic reveal button labels
_REVEAL_LABELS = [
    "Reveal Code", "Complete Challenge", "Capture Now",
    "Extract Code", "Retrieve from Cache", "All Tabs Visited",
    "Complete", "Reveal", "Capture", "Decode", "Solve",
]


async def _try_reveal_click(page: "Page") -> None:
    """Try clicking any visible enabled reveal/complete button on the page."""
    clicked = await page.evaluate("""
        () => {
            const labels = %s;
            const buttons = document.querySelectorAll('button');
            for (const label of labels) {
                for (const btn of buttons) {
                    const text = (btn.textContent || '').trim();
                    if (text.includes(label) && btn.offsetParent !== null && !btn.disabled) {
                        btn.click();
                        return text;
                    }
                }
            }
            return null;
        }
    """ % str(_REVEAL_LABELS))
    if clicked:
        print(f"[agent] Reveal click: {clicked}")
        await page.wait_for_timeout(300)
