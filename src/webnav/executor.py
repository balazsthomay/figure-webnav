"""Action executor: translates Action objects into Playwright calls.

Resolves element indices from the indexed element list into Playwright
locators via stored CSS selectors. No selector hallucination.
"""

from __future__ import annotations

import asyncio
import json
import re

from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeout

from webnav.actions import Action
from webnav.perception import ElementInfo


def _resolve_element(
    page: Page, action: Action, elements: list[ElementInfo]
) -> Locator | None:
    """Resolve an action's element index to a Playwright locator."""
    if action.element is None:
        return None
    if action.element < 0 or action.element >= len(elements):
        print(f"[executor] Element index {action.element} out of range (0-{len(elements)-1})")
        return None
    el = elements[action.element]
    return page.locator(el.selector).first


async def run(
    page: Page, action: Action, elements: list[ElementInfo] | None = None
) -> bool:
    """Execute a single action on the page. Returns True if successful."""
    elems = elements or []
    try:
        match action.type:
            case "click":
                return await _do_click(page, action, elems)
            case "fill":
                return await _do_fill(page, action, elems)
            case "scroll":
                return await _do_scroll(page, action)
            case "wait":
                await asyncio.sleep(action.amount)
                return True
            case "press":
                await page.keyboard.press(action.value)
                return True
            case "js":
                result = await page.evaluate(action.value)
                if result:
                    print(f"[executor] JS returned: {str(result)[:100]}")
                return True
            case "hover":
                return await _do_hover(page, action, elems)
            case "select":
                return await _do_select(page, action, elems)
            case "drag":
                return await _do_drag(page, action, elems)
            case "key_sequence":
                return await _do_key_sequence(page, action)
            case "scroll_element":
                return await _do_scroll_element(page, action)
            case "draw_strokes":
                return await _do_canvas_draw(page, action, elems)
            case _:
                print(f"[executor] Unknown action type: {action.type}")
                return False
    except Exception as e:
        print(f"[executor] Action {action.type} failed: {e}")
        return False


async def submit_code(page: Page, code: str) -> bool:
    """Find the code input field, enter the code, and submit."""
    try:
        textbox = None
        for selector in [
            "input[placeholder*='code' i]",
            "input[placeholder*='character' i]",
            "input[placeholder*='enter' i]",
            "input[type='text']",
            "input:not([type='hidden'])",
            "input",
        ]:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=500):
                    textbox = loc
                    break
            except PlaywrightTimeout:
                continue

        if textbox is None:
            print("[executor] Could not find textbox for code submission")
            return False

        await textbox.fill(code)
        await page.wait_for_timeout(100)

        submit = None
        for selector in [
            "button:has-text('Submit')",
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Next')",
            "button:has-text('Go')",
        ]:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=500):
                    submit = loc
                    break
            except PlaywrightTimeout:
                continue

        if submit:
            await submit.click(force=True)
        else:
            await textbox.press("Enter")

        await page.wait_for_timeout(500)
        return True

    except Exception as e:
        print(f"[executor] Code submission failed: {e}")
        return False


async def _do_click(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Click element by index, falling back to selector-based strategies."""
    n_clicks = max(action.amount, 1)

    # Index-based: resolve element
    if action.element is not None:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                for _ in range(n_clicks):
                    await loc.click(timeout=2000, force=True)
                    if n_clicks > 1:
                        await page.wait_for_timeout(200)
                return True
            except Exception as e:
                print(f"[executor] Index click failed: {e}")

    # Selector-based fallback (for legacy or submit_code paths)
    if action.selector:
        selectors = [s.strip() for s in action.selector.split(",")]
        for selector in selectors:
            try:
                all_loc = page.locator(selector)
                count = await all_loc.count()
                for idx in range(min(count, 8)):
                    loc = all_loc.nth(idx)
                    try:
                        if not await loc.is_visible(timeout=300):
                            continue
                    except Exception:
                        continue
                    for _ in range(n_clicks):
                        fresh = page.locator(selector).nth(idx)
                        try:
                            await fresh.click(timeout=2000, force=True)
                        except (PlaywrightTimeout, Exception):
                            pass
                        if n_clicks > 1:
                            await page.wait_for_timeout(200)
                    return True
            except (PlaywrightTimeout, Exception):
                continue

        # Text-based fallback
        text_match = re.search(r"has-text\(['\"](.+?)['\"]\)", action.selector)
        if text_match:
            text = text_match.group(1)
            try:
                loc = page.get_by_role("button", name=text).first
                await loc.click(timeout=2000, force=True)
                return True
            except Exception:
                pass
            try:
                loc = page.get_by_text(text, exact=False).first
                for _ in range(n_clicks):
                    await loc.click(timeout=2000, force=True)
                    if n_clicks > 1:
                        await page.wait_for_timeout(100)
                return True
            except Exception:
                pass

    print(f"[executor] Click failed — element={action.element} selector={action.selector}")
    return False


async def _do_fill(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Fill an input by index or selector fallback."""
    # Index-based
    if action.element is not None:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                await loc.fill(action.value)
                return True
            except Exception as e:
                print(f"[executor] Index fill failed: {e}")

    # Selector fallback
    for selector in [action.selector, "input[type='text']", "input", "textarea"]:
        if not selector:
            continue
        try:
            loc = page.locator(selector).first
            if await loc.is_visible(timeout=500):
                await loc.fill(action.value)
                return True
        except (PlaywrightTimeout, Exception):
            continue
    return False


async def _do_scroll(page: Page, action: Action) -> bool:
    """Scroll down by a pixel amount. Scrolls incrementally for triggers."""
    total = action.amount or 600
    step = 100
    scrolled = 0
    while scrolled < total:
        chunk = min(step, total - scrolled)
        await page.evaluate(f"window.scrollBy(0, {chunk})")
        scrolled += chunk
        await page.wait_for_timeout(50)
    return True


async def _do_hover(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Hover over an element by index with sustained JS events."""
    duration_ms = int((action.duration or 3) * 1000)

    # Index-based: get element selector for JS hover
    target_selector = None
    if action.element is not None and action.element < len(elements):
        target_selector = elements[action.element].selector

    # JS sustained hover — dispatches mouse events for the full duration
    hover_js = f"""
    (async () => {{
        let target = null;
        // Try specific selector first
        const sel = {json.dumps(target_selector) if target_selector else 'null'};
        if (sel) {{
            target = document.querySelector(sel);
        }}
        // Fallback: find hover target by text
        if (!target) {{
            for (const el of document.querySelectorAll('[class*="cursor-pointer"]')) {{
                const text = (el.textContent || '').toLowerCase();
                if (text.includes('hover') && el.offsetParent !== null) {{
                    target = el; break;
                }}
            }}
        }}
        if (!target) {{
            for (const el of document.querySelectorAll('div, p, span, section')) {{
                const text = (el.textContent || '').toLowerCase();
                if ((text.includes('hover here') || text.includes('hover over this'))
                    && el.children.length <= 3 && el.offsetParent !== null) {{
                    target = el; break;
                }}
            }}
        }}
        if (!target) return 'hover: no target found';
        const rect = target.getBoundingClientRect();
        const cx = rect.x + rect.width / 2;
        const cy = rect.y + rect.height / 2;
        const events = ['pointerenter', 'pointerover', 'mouseenter', 'mouseover', 'mousemove'];
        events.forEach(evt => {{
            target.dispatchEvent(new PointerEvent(evt, {{
                bubbles: true, cancelable: true,
                clientX: cx, clientY: cy, pointerType: 'mouse'
            }}));
        }});
        const duration = {duration_ms};
        let elapsed = 0;
        while (elapsed < duration) {{
            await new Promise(r => setTimeout(r, 100));
            elapsed += 100;
            target.dispatchEvent(new PointerEvent('pointermove', {{
                bubbles: true, cancelable: true,
                clientX: cx + (elapsed % 3), clientY: cy, pointerType: 'mouse'
            }}));
            target.dispatchEvent(new MouseEvent('mousemove', {{
                bubbles: true, cancelable: true,
                clientX: cx + (elapsed % 3), clientY: cy
            }}));
        }}
        return 'hover: held ' + duration + 'ms on ' + target.tagName;
    }})()
    """
    try:
        result = await page.evaluate(hover_js)
        print(f"[executor] Hover: {result}")
        return True
    except Exception as e:
        print(f"[executor] JS hover failed: {e}")

    # Playwright fallback
    if action.element is not None:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                await loc.hover(force=True)
                await asyncio.sleep(action.duration or 3)
                return True
            except Exception:
                pass

    # Text-based fallback
    for selector in [
        "text='Hover here to reveal code'",
        "text='Hover here'",
        "[class*='cursor-pointer']",
    ]:
        try:
            loc = page.locator(selector).first
            if await loc.is_visible(timeout=500):
                await loc.hover(force=True)
                await asyncio.sleep(action.duration or 3)
                return True
        except (PlaywrightTimeout, Exception):
            continue
    return False


async def _do_drag(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Drag-and-drop: JS DragEvent dispatch + Playwright mouse fallback."""
    try:
        # Strategy 1: JS DragEvent dispatch
        filled = await page.evaluate(_DRAG_JS)
        print(f"[executor] Drag JS strategy: {filled}/6 slots filled")
        if filled >= 6:
            return True

        # Strategy 2: Playwright drag_to
        for round_num in range(2):
            pieces_loc = page.locator('[draggable="true"]:not(.opacity-50)')
            pc = await pieces_loc.count()
            slot_loc = page.locator('.border-dashed:not([draggable]):not(.bg-green-100):not(.bg-green-200)')
            sc = await slot_loc.count()
            if pc == 0 or sc == 0:
                break
            print(f"[executor] Drag round {round_num + 1}: {pc} pieces, {sc} unfilled slots")
            for i in range(min(sc, 6)):
                try:
                    src = pieces_loc.nth(i % pc)
                    dst = slot_loc.nth(i)
                    if not await src.is_visible(timeout=200):
                        continue
                    if not await dst.is_visible(timeout=200):
                        continue
                    await src.drag_to(dst, timeout=2000)
                    await page.wait_for_timeout(150)
                except Exception:
                    pass
            filled = await page.evaluate(_DRAG_JS)
            if filled >= 6:
                return True

        # Strategy 3: Mouse simulation
        layout = await page.evaluate("""
            () => {
                const pieces = [], slots = [];
                for (const el of document.querySelectorAll('*')) {
                    const cls = String(el.className || '');
                    const draggable = el.getAttribute('draggable');
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 20 || rect.height < 20) continue;
                    if (rect.y < 0 || rect.y > window.innerHeight * 2) continue;
                    if (draggable === 'true' && !cls.includes('opacity-50')) {
                        pieces.push({x: rect.x + rect.width/2, y: rect.y + rect.height/2});
                    }
                    if (/border-dashed/.test(cls) && !draggable && !cls.includes('bg-green')) {
                        slots.push({x: rect.x + rect.width/2, y: rect.y + rect.height/2});
                    }
                }
                return {pieces, slots};
            }
        """)
        pieces = layout.get("pieces", [])
        slots = layout.get("slots", [])
        for i, slot in enumerate(slots[:6]):
            if not pieces:
                break
            piece = pieces[i % len(pieces)]
            px, py = piece["x"], piece["y"]
            sx, sy = slot["x"], slot["y"]
            await page.mouse.move(px, py)
            await page.wait_for_timeout(80)
            await page.mouse.down()
            await page.wait_for_timeout(80)
            for step in range(1, 16):
                x = px + (sx - px) * step / 15
                y = py + (sy - py) * step / 15
                await page.mouse.move(x, y)
                await page.wait_for_timeout(25)
            await page.mouse.up()
            await page.wait_for_timeout(200)

        filled = await page.evaluate(_DRAG_JS)
        print(f"[executor] Drag: {filled}/6 after mouse sim")
        return filled >= 6

    except Exception as e:
        print(f"[executor] Drag failed: {e}")
        return False


_DRAG_JS = """
(() => {
    const pieces = [...document.querySelectorAll('[draggable="true"]')];
    const allSlots = [...document.querySelectorAll('*')].filter(el => {
        const cls = String(el.className || '');
        return /border-dashed/.test(cls) && el.getAttribute('draggable') !== 'true'
            && el.getBoundingClientRect().width >= 20;
    });
    const slots = allSlots.filter(el => {
        const cls = String(el.className || '');
        return !cls.includes('bg-green');
    });
    for (let i = 0; i < slots.length && i < 6; i++) {
        const piece = pieces[i % pieces.length];
        const slot = slots[i];
        if (!piece || !slot) continue;
        const dt = new DataTransfer();
        try { dt.setData('text/plain', piece.textContent || ''); } catch(e) {}
        piece.dispatchEvent(new DragEvent('dragstart', {dataTransfer: dt, bubbles: true, cancelable: true}));
        slot.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true, cancelable: true}));
        slot.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
        slot.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
        piece.dispatchEvent(new DragEvent('dragend', {dataTransfer: dt, bubbles: true}));
    }
    return allSlots.filter(el => {
        const cls = String(el.className || '');
        return cls.includes('bg-green');
    }).length;
})()
"""


async def _do_scroll_element(page: Page, action: Action) -> bool:
    """Find a scrollable element and scroll it using mouse.wheel()."""
    try:
        box = await page.evaluate("""
            () => {
                const sels = ['.overflow-y-scroll', '[class*="overflow-y-scroll"]',
                              '[class*="overflow-auto"]'];
                for (const sel of sels) {
                    for (const el of document.querySelectorAll(sel)) {
                        if (el.scrollHeight > el.clientHeight + 10) {
                            const r = el.getBoundingClientRect();
                            return {x: r.x + r.width/2, y: r.y + r.height/2};
                        }
                    }
                }
                return null;
            }
        """)
        if not box:
            return False

        await page.mouse.move(box["x"], box["y"])
        await page.wait_for_timeout(50)
        for _ in range(5):
            await page.mouse.wheel(0, 200)
            await page.wait_for_timeout(100)
        return True
    except Exception as e:
        print(f"[executor] Scroll element failed: {e}")
        return False


async def _do_canvas_draw(
    page: Page, action: Action, elements: list[ElementInfo]
) -> bool:
    """Draw strokes on a canvas element using mouse simulation."""
    try:
        # Try index-based canvas first
        canvas = None
        if action.element is not None:
            loc = _resolve_element(page, action, elements)
            if loc and await loc.is_visible(timeout=1000):
                canvas = loc

        # Fallback: find any canvas
        if canvas is None:
            canvas = page.locator("canvas").first
            if not await canvas.is_visible(timeout=1000):
                print("[executor] Canvas not found")
                return False

        box = await canvas.bounding_box()
        if not box:
            return False

        cx, cy, cw, ch = box["x"], box["y"], box["width"], box["height"]

        # Draw 5 strokes across the canvas
        for i in range(5):
            y = cy + ch * (0.2 + 0.15 * i)
            x_start = cx + cw * 0.1
            x_end = cx + cw * 0.9

            await page.mouse.move(x_start, y)
            await page.mouse.down()
            steps = 8
            for s in range(1, steps + 1):
                x = x_start + (x_end - x_start) * s / steps
                await page.mouse.move(x, y + (s % 2) * 5)
                await page.wait_for_timeout(20)
            await page.mouse.up()
            await page.wait_for_timeout(200)

        return True
    except Exception as e:
        print(f"[executor] Canvas draw failed: {e}")
        return False


async def _do_select(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Select an option from a dropdown."""
    try:
        if action.element is not None:
            loc = _resolve_element(page, action, elements)
            if loc:
                await loc.select_option(label=action.value)
                return True
        select = page.locator("select").first
        await select.select_option(label=action.value)
        return True
    except Exception:
        return False


# Map arrow/symbol characters to Playwright key names
_KEY_MAP = {
    "→": "ArrowRight", "←": "ArrowLeft", "↑": "ArrowUp", "↓": "ArrowDown",
    "⬆": "ArrowUp", "⬇": "ArrowDown", "⬅": "ArrowLeft", "➡": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "enter": "Enter", "space": "Space", "tab": "Tab",
    "escape": "Escape", "backspace": "Backspace",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
}


async def _do_key_sequence(page: Page, action: Action) -> bool:
    """Read required key sequence from page (or action value) and press each key."""
    try:
        # If keys are provided in the action value, use those directly
        if action.value:
            keys = _parse_key_tokens(action.value)
            if keys:
                print(f"[executor] Key sequence from action: pressing {keys}")
                for key in keys:
                    await page.keyboard.press(key)
                    await page.wait_for_timeout(200)
                return True

        # Otherwise read from page text
        text = await page.inner_text("body")
        keys = _extract_key_sequence(text)
        if not keys:
            print(f"[executor] Key sequence: could not extract keys from page")
            return False

        print(f"[executor] Key sequence: pressing {keys}")
        for key in keys:
            await page.keyboard.press(key)
            await page.wait_for_timeout(200)
        return True
    except Exception as e:
        print(f"[executor] Key sequence failed: {e}")
        return False


def _extract_key_sequence(text: str) -> list[str]:
    """Extract key names from page text."""
    keys: list[str] = []
    in_sequence = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"(?:required\s+)?sequence\s*:", stripped, re.IGNORECASE):
            in_sequence = True
            after = re.split(r"sequence\s*:\s*", stripped, flags=re.IGNORECASE)[-1]
            if after:
                keys.extend(_parse_key_tokens(after))
            continue
        if in_sequence and stripped:
            parsed = _parse_key_tokens(stripped)
            if parsed:
                keys.extend(parsed)
            elif len(keys) >= 2:
                break
    return keys


def _parse_key_tokens(text: str) -> list[str]:
    """Parse key tokens from a text string."""
    keys: list[str] = []
    # Check for arrow characters individually
    for ch in text:
        if ch in _KEY_MAP:
            keys.append(_KEY_MAP[ch])

    if keys:
        return keys

    # Split on common delimiters
    tokens = re.split(r"[\s,→←↑↓]+", text)
    for token in tokens:
        token = token.strip().strip("'\"")
        if not token:
            continue
        low = token.lower()
        if low in _KEY_MAP:
            keys.append(_KEY_MAP[low])
        elif len(token) == 1 and token.isalpha():
            keys.append(token.lower())
        elif token in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                       "Enter", "Space", "Tab", "Escape", "Backspace"):
            keys.append(token)
    return keys
