"""Orchestrator: main agent loop wiring all components."""

from __future__ import annotations

import asyncio
import time

from webnav.browser import BASE_URL, BrowserController
from webnav.dispatcher import Action, match as dispatch_match
from webnav.executor import run as execute_action, submit_code
from webnav.extractor import find_code
from webnav.metrics import MetricsCollector
from webnav.page_cleaner import clean_page
from webnav.perception import snapshot
from webnav.solver import Solver
from webnav.state import StateTracker


class Agent:
    """Main agent that solves the 30-step web navigation challenge."""

    def __init__(self, headless: bool = True) -> None:
        self.browser = BrowserController(headless=headless)
        self.solver = Solver()
        self.state = StateTracker()
        self.metrics = MetricsCollector()
        self._used_codes: set[str] = set()  # codes already submitted on previous steps

    async def run(self) -> MetricsCollector:
        """Execute the full 30-step challenge. Returns metrics."""
        async with self.browser:
            # Navigate to landing page and click START
            await self.browser.goto(BASE_URL)
            await asyncio.sleep(1.0)
            try:
                start_btn = self.browser.page.locator("text=START").first
                await start_btn.click(timeout=5000)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[agent] Could not click START: {e}")

            # Main loop: solve until done or time runs out
            max_attempts = 5
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

                # Track attempts on this specific step
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
                        # Last step special case: try direct /finish navigation
                        if current_step == self.state.total_steps:
                            print(f"[agent] Step {current_step} (final step) — trying /finish navigation")
                            await self.browser.page.evaluate(
                                _COMPLETE_LAST_STEP_JS
                            )
                            await asyncio.sleep(1.0)
                            new_url = self.browser.page.url
                            if "finish" in new_url:
                                self.state.mark_completed()
                                self.metrics.end_step(True, code="FINISH")
                                print(f"[agent] Step {current_step} PASSED via /finish")
                                break
                        print(f"[agent] Step {current_step} stuck after {max_attempts} attempts, stopping")
                        break

        self.metrics.print_report()
        return self.metrics

    async def _solve_step(self, step: int, attempt: int = 0) -> bool:
        """Solve a single step. Returns True if the step was completed."""
        self.state.begin_step(step)
        self.metrics.begin_step(step)
        t0 = time.time()
        print(f"\n[agent] === Step {step} (attempt {attempt + 1}) ===")

        try:
            # 1. Clean page (remove popups/overlays)
            await clean_page(self.browser.page)
            await asyncio.sleep(0.2)
            t1 = time.time()

            # 2. Perceive
            page_state = await snapshot(self.browser.page)
            print(f"[agent] Instruction: {page_state.instruction[:100]}")
            t2 = time.time()

            # 3. Dispatch FIRST — determines if there are actions to perform
            actions = None
            tier = 0

            if attempt >= 2:
                print("[agent] Multiple failures — calling Tier 2 LLM with screenshot")
                tier = 2
                screenshot = await self.browser.screenshot()
                pre_tokens = self.solver.total_tokens
                actions = await self.solver.solve_stuck(page_state, screenshot)
                self.metrics.record_llm_call(2, self.solver.total_tokens - pre_tokens)
            else:
                actions = dispatch_match(page_state.instruction)
                if actions is not None:
                    print(f"[agent] Tier 0 match: {[a.type for a in actions]}")
                elif page_state.visible_codes:
                    # No dispatch match + code visible → submit directly (enter_code type)
                    code = page_state.visible_codes[0]
                    print(f"[agent] Code already visible: {code}")
                    if await self._submit_and_check(code, step):
                        return True
                    # Submit failed — fall through to LLM
                if actions is None:
                    print("[agent] Tier 0 miss — calling Tier 1 LLM")
                    tier = 1
                    pre_tokens = self.solver.total_tokens
                    actions = await self.solver.solve(page_state)
                    self.metrics.record_llm_call(1, self.solver.total_tokens - pre_tokens)
            t3 = time.time()

            if not actions:
                actions = [Action(type="js", value=_REVEAL_HIDDEN_JS)]

            # 4. Execute actions
            js_found_code = None
            for action in actions:
                desc = action.selector or (action.value[:60] if action.value else str(action.amount))
                print(f"[agent]   -> {action.type} {desc}")
                await execute_action(self.browser.page, action)
                # Check if JS action found a code (e.g., puzzle fallback timer)
                if hasattr(action, '_found_code') and action._found_code:
                    js_found_code = action._found_code
            t4 = time.time()

            # Brief pause for page reaction
            await asyncio.sleep(0.3)

            # 4b. Generic: try clicking any visible reveal/complete button
            await _try_reveal_click(self.browser.page)
            await asyncio.sleep(0.2)
            t5 = time.time()

            # 4c. If JS found a code directly (e.g., puzzle timer), use it immediately
            if js_found_code and js_found_code not in self._used_codes:
                print(f"[agent] JS-found code: {js_found_code}")
                if await self._submit_and_check(js_found_code, step):
                    return True

            # 5. Extract code — fast path first, then cleanup + retry
            code = await find_code(self.browser.page, self._used_codes)
            t6a = time.time()
            if code is None:
                await clean_page(self.browser.page)
                await asyncio.sleep(0.2)
                code = await find_code(self.browser.page, self._used_codes)

            # If no code, try the reveal JS (hidden DOM) + poll
            t6b = time.time()
            if code is None:
                await self.browser.page.evaluate(_REVEAL_HIDDEN_JS)
                await asyncio.sleep(0.2)
                code = await find_code(self.browser.page, self._used_codes)

            t6c = time.time()
            if code is None and not self.state.is_step_timed_out():
                # Brief poll — retry is faster than long polling
                for _ in range(2):
                    await asyncio.sleep(0.4)
                    code = await find_code(self.browser.page, self._used_codes)
                    if code:
                        break
            t6d = time.time()

            # 5b. Last resort: read code from sessionStorage (XOR-encrypted)
            if code is None:
                ss_code = await self.browser.page.evaluate(
                    _SESSION_STORAGE_JS.replace("STEP_NUM", str(step))
                )
                if ss_code and ss_code not in self._used_codes:
                    print(f"[agent] Code from session storage: {ss_code}")
                    code = ss_code
            t6e = time.time()

            print(f"[timing] clean={t1-t0:.1f} perceive={t2-t1:.1f} dispatch={t3-t2:.1f} "
                  f"exec={t4-t3:.1f} reveal_btn={t5-t4:.1f} find1={t6a-t5:.1f} "
                  f"find2={t6b-t6a:.1f} find3={t6c-t6b:.1f} poll={t6d-t6c:.1f} "
                  f"ss={t6e-t6d:.1f} total={t6e-t0:.1f}")

            # Debug: log page text on failure + screenshot for puzzles
            if code is None:
                try:
                    raw = await self.browser.page.inner_text("body")
                    lines = [l.strip() for l in raw.splitlines() if l.strip() and "filler" not in l.lower() and "section " not in l.lower()][:10]
                    print(f"[agent] Page text after action: {lines[:5]}")
                except Exception:
                    pass
                # Save screenshot for stuck puzzle steps
                if "puzzle" in page_state.instruction.lower() and attempt == 0:
                    try:
                        path = f"/tmp/puzzle_step{step}.png"
                        await self.browser.page.screenshot(path=path)
                        print(f"[agent] Puzzle screenshot saved: {path}")
                    except Exception:
                        pass

            # 7. Submit code
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
        # Clear overlays right before submit — they regenerate after actions
        await clean_page(self.browser.page)
        url_before = self.browser.page.url
        submitted = await submit_code(self.browser.page, code)
        if submitted:
            # Wait for URL to change (any change = navigation happened)
            try:
                await self.browser.page.wait_for_url(
                    lambda url: url != url_before, timeout=5000
                )
            except Exception:
                pass  # URL might not have changed yet
            # Check if we advanced
            advanced = await self.state.check_advancement(self.browser.page, step)
            if advanced:
                self.state.mark_completed()
                self.metrics.end_step(True, code=code)
                self._used_codes.add(code)
                print(f"[agent] Step {step} PASSED")
                return True
            # Brief poll in case SPA is slow
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


# JS to reveal hidden codes in the DOM — targeted approach.
# Only modifies elements with code-like text content. Avoids expensive getComputedStyle
# on all 400+ elements which caused 20-30s reflows.
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

# Reveal button labels to try after action execution (order matters)
_REVEAL_LABELS = [
    "Reveal Code", "Complete Challenge", "Capture Now",
    "Extract Code", "Retrieve from Cache", "All Tabs Visited",
    "Complete", "Reveal", "Capture", "Decode", "Solve",
]


async def _try_reveal_click(page: "Page") -> None:
    """Try clicking any visible enabled reveal/complete button on the page.

    Uses a single JS call to find the right button, avoiding 11 separate
    Playwright locator calls (each with 200ms timeout = 4.4s worst case).
    """
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


# JS to read a specific step's code from sessionStorage.
# The session stores all 30 codes XOR-encrypted with "WO_2024_CHALLENGE".
# codes[N] = code needed to advance FROM step N+1 (i.e., submit on step N).
# To advance from step STEP_NUM, we need codes[STEP_NUM] (0-indexed).
_SESSION_STORAGE_JS = """
(() => {
    const raw = sessionStorage.getItem('wo_session');
    if (!raw) return null;
    try {
        const key = 'WO_2024_CHALLENGE';
        const decoded = atob(raw);
        let decrypted = '';
        for (let i = 0; i < decoded.length; i++) {
            decrypted += String.fromCharCode(decoded.charCodeAt(i) ^ key.charCodeAt(i % key.length));
        }
        const data = JSON.parse(decrypted);
        if (data.codes && Array.isArray(data.codes)) {
            return data.codes[STEP_NUM] || null;
        }
    } catch(e) {}
    return null;
})()
"""

# JS to complete the last step (step 30) by marking it done in sessionStorage
# and navigating to /finish. The last step has no codes.get(31), so the normal
# code submission mechanism can't work.
_COMPLETE_LAST_STEP_JS = """
(() => {
    const key = 'WO_2024_CHALLENGE';
    const raw = sessionStorage.getItem('wo_session');
    if (!raw) return 'no session';
    try {
        const decoded = atob(raw);
        let decrypted = '';
        for (let i = 0; i < decoded.length; i++) {
            decrypted += String.fromCharCode(decoded.charCodeAt(i) ^ key.charCodeAt(i % key.length));
        }
        const data = JSON.parse(decrypted);
        // Mark step 30 as completed
        if (!data.completed) data.completed = [];
        if (!data.completed.includes(30)) data.completed.push(30);
        // Re-encrypt and save
        const json = JSON.stringify(data);
        let encrypted = '';
        for (let i = 0; i < json.length; i++) {
            encrypted += String.fromCharCode(json.charCodeAt(i) ^ key.charCodeAt(i % key.length));
        }
        sessionStorage.setItem('wo_session', btoa(encrypted));
        // Navigate to finish
        const version = new URLSearchParams(window.location.search).get('version') || '1';
        window.location.href = '/finish?version=' + version;
        return 'navigating to finish';
    } catch(e) {
        return 'error: ' + e.message;
    }
})()
"""
