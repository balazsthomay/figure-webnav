"""Action executor: translates Action objects into Playwright calls."""

from __future__ import annotations

import asyncio
import re

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from webnav.dispatcher import Action


async def run(page: Page, action: Action) -> bool:
    """Execute a single action on the page. Returns True if successful."""
    try:
        match action.type:
            case "click":
                return await _do_click(page, action)
            case "fill":
                return await _do_fill(page, action)
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
                    print(f"[executor] JS returned: {result}")
                    # Store fallback code for agent to extract
                    if isinstance(result, str) and result.startswith("fallback_code:"):
                        action._found_code = result.split(":", 1)[1]
                return True
            case "hover":
                return await _do_hover(page, action)
            case "select":
                return await _do_select(page, action)
            case "drag_fill":
                return await _do_drag_fill(page, action)
            case "key_sequence":
                return await _do_key_sequence(page, action)
            case "scroll_element":
                return await _do_scroll_element(page, action)
            case "canvas_draw":
                return await _do_canvas_draw(page, action)
            case _:
                print(f"[executor] Unknown action type: {action.type}")
                return False
    except Exception as e:
        print(f"[executor] Action {action.type} failed: {e}")
        return False


async def submit_code(page: Page, code: str) -> bool:
    """Find the code input field, enter the code, and submit.

    Uses Playwright fill() which properly interacts with React controlled inputs.
    """
    try:
        # Find the code submission textbox (prioritize code-specific inputs)
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

        # Find and click submit button
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


async def _do_click(page: Page, action: Action) -> bool:
    """Click element(s). Supports comma-separated selectors (try each).

    Iterates ALL matching elements (not just .first) to avoid hidden overlay
    decoys that appear first in DOM order.
    """
    selectors = [s.strip() for s in action.selector.split(",")]
    n_clicks = max(action.amount, 1)

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
                # Found a visible element — click it
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

    # Fallback: try by text content
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

    print(f"[executor] Click failed — no visible element for: {action.selector}")
    return False


async def _do_fill(page: Page, action: Action) -> bool:
    """Fill a textbox with a value."""
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


async def _do_hover(page: Page, action: Action) -> bool:
    """Hover over an element and hold."""
    selectors = [s.strip() for s in action.selector.split(",")]
    # Also try text-based selectors for "Hover here"
    selectors.extend([
        "text='Hover here to reveal code'",
        "text='Hover here'",
        "[class*='cursor-pointer']",
    ])
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if await loc.is_visible(timeout=500):
                await loc.hover(force=True)
                return True
        except (PlaywrightTimeout, Exception):
            continue
    return False


async def _do_drag_fill(page: Page, action: Action) -> bool:
    """Fill drag-and-drop slots using multiple strategies.

    Strategy 1: JS DragEvent dispatch (directly invokes React's onDrop)
    Strategy 2: Playwright drag_to (CDP drag events)
    Strategy 3: Mouse simulation fallback
    """
    try:
        # Strategy 1: JS DragEvent dispatch — most reliable for React
        filled = await page.evaluate(_DRAG_JS)
        print(f"[executor] Drag JS strategy: {filled}/6 slots filled")
        if filled >= 6:
            return True

        # Strategy 2: Playwright drag_to with verification
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
            # Re-check via JS
            filled = await page.evaluate(_DRAG_JS)
            if filled >= 6:
                print(f"[executor] Drag: all slots filled after round {round_num + 1}")
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
        print(f"[executor] Drag mouse sim: {len(pieces)} pieces, {len(slots)} unfilled slots")

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
        print(f"[executor] Drag fill failed: {e}")
        return False


# JS that dispatches DragEvents on pieces/slots to trigger React's handlers,
# then returns the count of filled slots.
_DRAG_JS = """
(() => {
    const pieces = [...document.querySelectorAll('[draggable="true"]')];
    const allSlots = [...document.querySelectorAll('*')].filter(el => {
        const cls = String(el.className || '');
        return /border-dashed/.test(cls) && el.getAttribute('draggable') !== 'true'
            && el.getBoundingClientRect().width >= 20;
    });
    // Only target unfilled slots
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

        piece.dispatchEvent(new DragEvent('dragstart', {
            dataTransfer: dt, bubbles: true, cancelable: true
        }));
        slot.dispatchEvent(new DragEvent('dragenter', {
            dataTransfer: dt, bubbles: true, cancelable: true
        }));
        slot.dispatchEvent(new DragEvent('dragover', {
            dataTransfer: dt, bubbles: true, cancelable: true
        }));
        slot.dispatchEvent(new DragEvent('drop', {
            dataTransfer: dt, bubbles: true, cancelable: true
        }));
        piece.dispatchEvent(new DragEvent('dragend', {
            dataTransfer: dt, bubbles: true
        }));
    }

    // Count filled slots (bg-green class)
    return allSlots.filter(el => {
        const cls = String(el.className || '');
        return cls.includes('bg-green');
    }).length;
})()
"""


async def _do_scroll_element(page: Page, action: Action) -> bool:
    """Find a scrollable element on the page and scroll it using mouse.wheel().

    This triggers real browser scroll events that React can detect,
    unlike programmatic scrollTop changes which may not fire synthetic events.
    """
    try:
        # Find the scrollable element
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
            print("[executor] No scrollable element found")
            return False

        # Move mouse to the element and scroll with wheel
        await page.mouse.move(box["x"], box["y"])
        await page.wait_for_timeout(50)
        for _ in range(5):
            await page.mouse.wheel(0, 200)
            await page.wait_for_timeout(100)
        return True
    except Exception as e:
        print(f"[executor] Scroll element failed: {e}")
        return False


async def _do_canvas_draw(page: Page, action: Action) -> bool:
    """Draw strokes on a canvas element using mouse simulation."""
    try:
        canvas = page.locator("canvas").first
        if not await canvas.is_visible(timeout=1000):
            print("[executor] Canvas not found")
            return False

        box = await canvas.bounding_box()
        if not box:
            return False

        cx, cy, cw, ch = box["x"], box["y"], box["width"], box["height"]
        print(f"[executor] Canvas at ({cx:.0f},{cy:.0f}) size {cw:.0f}x{ch:.0f}")

        # Draw 5 strokes across the canvas (more than enough)
        for i in range(5):
            y = cy + ch * (0.2 + 0.15 * i)  # Spread strokes vertically
            x_start = cx + cw * 0.1
            x_end = cx + cw * 0.9

            await page.mouse.move(x_start, y)
            await page.mouse.down()
            # Draw in steps for stroke detection
            steps = 8
            for s in range(1, steps + 1):
                x = x_start + (x_end - x_start) * s / steps
                await page.mouse.move(x, y + (s % 2) * 5)  # Slight zigzag
                await page.wait_for_timeout(20)
            await page.mouse.up()
            await page.wait_for_timeout(200)

        return True
    except Exception as e:
        print(f"[executor] Canvas draw failed: {e}")
        return False


async def _do_select(page: Page, action: Action) -> bool:
    """Select an option from a dropdown."""
    try:
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
    """Read required key sequence from page and press each key."""
    try:
        text = await page.inner_text("body")
        # Extract key sequence from page text
        keys = _extract_key_sequence(text)
        if not keys:
            print(f"[executor] Key sequence: could not extract keys from page")
            # Debug: print relevant lines
            for line in text.splitlines():
                line = line.strip()
                if any(kw in line.lower() for kw in ["sequence", "arrow", "key", "press"]):
                    print(f"  {line[:80]}")
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
    # Look for lines after "Required sequence:" or "Sequence:"
    in_sequence = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"(?:required\s+)?sequence\s*:", stripped, re.IGNORECASE):
            in_sequence = True
            # Check if keys are on the same line after the colon
            after = re.split(r"sequence\s*:\s*", stripped, flags=re.IGNORECASE)[-1]
            if after:
                keys.extend(_parse_key_tokens(after))
            continue
        if in_sequence and stripped:
            # Each line might be a key name
            parsed = _parse_key_tokens(stripped)
            if parsed:
                keys.extend(parsed)
            elif len(keys) >= 2:
                break  # Probably moved past the sequence section
    return keys


def _parse_key_tokens(text: str) -> list[str]:
    """Parse key tokens from a text string."""
    keys: list[str] = []
    # Split on common delimiters
    tokens = re.split(r"[\s,→←↑↓]+", text)
    # Also check for arrow characters individually
    for ch in text:
        if ch in _KEY_MAP:
            keys.append(_KEY_MAP[ch])

    if keys:
        return keys

    # Try individual tokens
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
