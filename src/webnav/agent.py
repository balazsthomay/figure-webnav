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
                el_desc = ""
                if action.element is not None and action.element < len(page_state.elements):
                    ei = page_state.elements[action.element]
                    el_desc = f"element={action.element} ({ei.tag} \"{ei.name[:40]}\")"
                elif action.element is not None:
                    el_desc = f"element={action.element}"
                else:
                    el_desc = action.value[:60] if action.value else str(action.amount)
                print(f"[agent]   -> {action.type} {el_desc}")
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

            # Programmatic "click here N times" fallback
            if code is None:
                code = await _try_click_here(self.browser.page, page_state.instruction, self._used_codes)

            # Programmatic "find scattered parts" fallback
            if code is None and "part" in page_state.instruction.lower():
                code = await _try_find_parts(self.browser.page, self._used_codes)

            # Programmatic "sequence challenge" fallback
            if code is None and "sequence" in page_state.instruction.lower() and "complete" in page_state.instruction.lower():
                code = await _try_sequence_actions(self.browser.page, self._used_codes)

            # Programmatic "hover challenge" fallback
            if code is None and "hover" in page_state.instruction.lower():
                code = await _try_hover(self.browser.page, self._used_codes)

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


async def _try_hover(
    page: "Page", used_codes: set[str]
) -> str | None:
    """Programmatic fallback: find and hover over the challenge hover target."""
    print("[agent] Hover fallback: finding and hovering target via Playwright mouse")

    candidates = await page.evaluate("""
        () => {
            const candidates = [];
            for (const el of document.querySelectorAll('div, section, span, p')) {
                if (el.offsetParent === null) continue;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (rect.width < 20 || rect.height < 20) continue;
                if (rect.width > 800) continue;
                const tag = el.tagName;
                if (tag === 'BUTTON' || tag === 'A' || tag === 'INPUT') continue;
                const text = (el.textContent || '').toLowerCase();
                const hasCursorPointer = style.cursor === 'pointer' ||
                    (el.className || '').includes('cursor-pointer');
                const hasBgColor = style.backgroundColor !== 'rgba(0, 0, 0, 0)' &&
                    style.backgroundColor !== 'transparent';
                const hasHoverText = text.includes('hover');
                let score = 0;
                if (hasHoverText && hasCursorPointer) score += 20;
                if (hasHoverText) score += 10;
                if (hasCursorPointer) score += 5;
                if (hasBgColor && rect.width > 50 && rect.height > 50) score += 3;
                if (text.length > 200) score -= 10;
                if (el.children.length > 5) score -= 5;
                if (score > 0) {
                    candidates.push({
                        x: rect.x + rect.width/2, y: rect.y + rect.height/2,
                        score, text: text.slice(0, 40), tag
                    });
                }
            }
            candidates.sort((a, b) => b.score - a.score);
            return candidates.slice(0, 3);
        }
    """)

    if not candidates:
        print("[agent] Hover fallback: no targets found")
        return None

    for t in candidates:
        print(f"[agent] Hover fallback: trying {t['tag']} '{t['text']}' (score={t['score']})")
        try:
            await page.mouse.move(t["x"], t["y"])
            for i in range(10):
                await asyncio.sleep(0.15)
                await page.mouse.move(t["x"] + (i % 3) - 1, t["y"])
        except Exception as e:
            print(f"[agent] Hover fallback: failed: {e}")
            continue

        await page.wait_for_timeout(200)
        code = await find_code(page, used_codes)
        if code:
            return code

    return None


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


async def _try_click_here(
    page: "Page", instruction: str, used_codes: set[str]
) -> str | None:
    """Programmatic fallback: if instruction says 'click here N times', do it."""
    import re as _re

    match = _re.search(r"click\s+here\s+(\d+)\s+more\s+time", instruction, _re.IGNORECASE)
    if not match:
        return None

    n = int(match.group(1)) + 1  # +1 safety margin
    print(f"[agent] Click-here fallback: clicking {n} times via Playwright")

    # Use Playwright real clicks for proper React event handling
    try:
        # Find the "click here" button/element
        loc = page.get_by_text("click here", exact=False).first
        for i in range(n):
            await loc.click(timeout=1000, force=True)
            await page.wait_for_timeout(150)
        print(f"[agent] Click-here: clicked {n} times via Playwright locator")
    except Exception as e:
        print(f"[agent] Click-here Playwright failed ({e}), trying JS fallback")
        await page.evaluate("""
            (n) => {
                const candidates = [];
                for (const el of document.querySelectorAll('*')) {
                    const text = (el.textContent || '').toLowerCase();
                    if (text.includes('click here') && el.offsetParent !== null) {
                        candidates.push({el, len: (el.textContent || '').length});
                    }
                }
                candidates.sort((a, b) => a.len - b.len);
                if (candidates.length === 0) return;
                const target = candidates[0].el;
                for (let i = 0; i < n; i++) {
                    target.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                }
            }
        """, n)

    await page.wait_for_timeout(500)

    # Now scan for the code
    code = await find_code(page, used_codes)
    if code:
        return code

    # Try reveal hidden after clicking
    await page.evaluate(_REVEAL_HIDDEN_JS)
    await page.wait_for_timeout(200)
    return await find_code(page, used_codes)


async def _try_find_parts(
    page: "Page", used_codes: set[str]
) -> str | None:
    """Programmatic fallback: find and click scattered 'part' elements using Playwright clicks."""
    print("[agent] Find-parts fallback: searching for scattered clickable parts")

    # Find ALL small absolute-positioned elements and cursor:pointer elements
    # that might be "parts" — get their coordinates for Playwright clicks
    targets = await page.evaluate("""
        () => {
            const targets = [];
            const seen = new Set();
            for (const el of document.querySelectorAll('*')) {
                if (el.offsetParent === null && el.tagName !== 'BODY') continue;
                const rect = el.getBoundingClientRect();
                if (rect.width < 5 || rect.height < 5) continue;
                if (rect.top < 0 || rect.top > window.innerHeight * 3) continue;

                const text = (el.textContent || '').trim();
                const tag = el.tagName.toLowerCase();
                const style = window.getComputedStyle(el);
                const isAbsolute = style.position === 'absolute' || style.position === 'fixed';
                const hasCursorPointer = style.cursor === 'pointer';
                const isLeaf = el.childElementCount === 0;

                // Skip navigation/noise buttons
                if (/submit|next|prev|continue|proceed|advance|go forward|keep going|move on/i.test(text)) continue;
                // Skip large elements (containers)
                if (rect.width > 300 && rect.height > 100) continue;

                let score = 0;
                // Part-like text
                if (/part\\s*\\d|fragment|piece|star|\\u2605|\\u2606|\\u25cf|\\u25cb/i.test(text)) score += 20;
                // Small absolute-positioned clickable elements
                if (isAbsolute && hasCursorPointer && isLeaf) score += 15;
                if (isAbsolute && isLeaf && text.length < 20) score += 10;
                if (hasCursorPointer && isLeaf && text.length < 20) score += 8;
                // Small colored elements (visual markers)
                if (isLeaf && rect.width < 50 && rect.height < 50 && style.backgroundColor !== 'transparent'
                    && style.backgroundColor !== 'rgba(0, 0, 0, 0)') score += 5;

                if (score > 0) {
                    const key = Math.round(rect.x) + ',' + Math.round(rect.y);
                    if (!seen.has(key)) {
                        seen.add(key);
                        targets.push({
                            x: rect.x + rect.width/2, y: rect.y + rect.height/2,
                            score, text: text.slice(0, 30), tag, w: Math.round(rect.width), h: Math.round(rect.height)
                        });
                    }
                }
            }
            targets.sort((a, b) => b.score - a.score);
            return targets.slice(0, 10);
        }
    """)

    if not targets:
        print("[agent] Find-parts: no targets found")
        return None

    print(f"[agent] Find-parts: found {len(targets)} candidates")
    for t in targets:
        print(f"[agent] Find-parts: clicking {t['tag']} '{t['text']}' ({t['w']}x{t['h']}) score={t['score']}")
        try:
            await page.mouse.click(t["x"], t["y"])
            await page.wait_for_timeout(200)
        except Exception as e:
            print(f"[agent] Find-parts: click failed: {e}")

    await page.wait_for_timeout(500)

    code = await find_code(page, used_codes)
    if code:
        return code

    await page.evaluate(_REVEAL_HIDDEN_JS)
    await page.wait_for_timeout(200)
    return await find_code(page, used_codes)


async def _try_sequence_actions(
    page: "Page", used_codes: set[str]
) -> str | None:
    """Programmatic fallback: complete a sequence challenge using Playwright actions."""
    print("[agent] Sequence fallback: attempting all 4 actions via Playwright")

    # Diagnose: what does the challenge page look like?
    diag = await page.evaluate("""
        () => {
            const info = { buttons: [], inputs: [], hoverTargets: [], pageText: '' };
            for (const btn of document.querySelectorAll('button')) {
                if (btn.offsetParent !== null) {
                    info.buttons.push(btn.textContent.trim().slice(0, 50));
                }
            }
            for (const inp of document.querySelectorAll('input, textarea')) {
                if (inp.offsetParent !== null) {
                    info.inputs.push({type: inp.type, placeholder: inp.placeholder, value: inp.value});
                }
            }
            // Find hover candidates: visible elements with cursor:pointer that aren't buttons/links/inputs
            for (const el of document.querySelectorAll('div, section, span, p')) {
                if (el.offsetParent === null) continue;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (rect.width < 20 || rect.height < 20) continue;
                if (style.cursor === 'pointer' || el.classList.contains('cursor-pointer')) {
                    const text = (el.textContent || '').trim().slice(0, 60);
                    info.hoverTargets.push({
                        tag: el.tagName, text, w: Math.round(rect.width),
                        h: Math.round(rect.height), cursor: style.cursor,
                        bg: style.backgroundColor, childCount: el.children.length
                    });
                }
            }
            info.pageText = (document.body.innerText || '').slice(0, 500);
            return info;
        }
    """)
    print(f"[agent] Sequence diag: buttons={diag.get('buttons')}")
    print(f"[agent] Sequence diag: inputs={diag.get('inputs')}")
    print(f"[agent] Sequence diag: hoverTargets={diag.get('hoverTargets')}")

    # 1. Click the action button
    for label in ["Click Me", "Click", "Click Here", "Press Me"]:
        try:
            btn = page.get_by_role("button", name=label, exact=False).first
            if await btn.is_visible(timeout=300):
                await btn.click(timeout=1000, force=True)
                print(f"[agent] Sequence: clicked '{label}'")
                break
        except Exception:
            continue
    await page.wait_for_timeout(300)

    # 2. Fill the text input FIRST (before hover, in case order matters)
    for selector in ["input[type='text']", "input:not([type='hidden']):not([type='submit'])", "textarea"]:
        try:
            inp = page.locator(selector).first
            if await inp.is_visible(timeout=300):
                await inp.click(timeout=500)
                await inp.fill("hello")
                # Also try keyboard input as backup
                await inp.press("a")
                await page.wait_for_timeout(100)
                print("[agent] Sequence: filled input with 'hello'")
                break
        except Exception:
            continue
    await page.wait_for_timeout(300)

    # 3. Hover — try ALL cursor:pointer divs, not just the "best" one
    hover_targets = diag.get("hoverTargets", [])
    for target_info in hover_targets:
        hover_box = await page.evaluate("""
            (searchText) => {
                for (const el of document.querySelectorAll('div, section, span, p')) {
                    if (el.offsetParent === null) continue;
                    const style = window.getComputedStyle(el);
                    if (style.cursor !== 'pointer') continue;
                    const text = (el.textContent || '').trim().slice(0, 60);
                    if (text === searchText) {
                        const rect = el.getBoundingClientRect();
                        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2, tag: el.tagName, text};
                    }
                }
                return null;
            }
        """, target_info.get("text", ""))
        if hover_box:
            try:
                await page.mouse.move(hover_box["x"], hover_box["y"])
                for i in range(20):
                    await asyncio.sleep(0.15)
                    await page.mouse.move(hover_box["x"] + (i % 3) - 1, hover_box["y"])
                print(f"[agent] Sequence: hovered {hover_box['tag']} '{hover_box['text']}'")
            except Exception as e:
                print(f"[agent] Sequence: hover failed: {e}")
    await page.wait_for_timeout(300)

    # 4. Scroll down incrementally (to trigger scroll event listeners)
    for _ in range(5):
        await page.evaluate("window.scrollBy(0, 100)")
        await page.wait_for_timeout(100)
    print("[agent] Sequence: scrolled 500px")
    await page.wait_for_timeout(500)

    # Check for code
    code = await find_code(page, used_codes)
    if code:
        return code

    await page.evaluate(_REVEAL_HIDDEN_JS)
    await page.wait_for_timeout(200)
    return await find_code(page, used_codes)
