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
    """Resolve an action's element index to a Playwright locator.

    Handles both regular CSS selectors and shadow DOM elements
    (selector starts with 'shadow:').
    """
    if action.element is None:
        return None
    if action.element < 0 or action.element >= len(elements):
        print(f"[executor] Element index {action.element} out of range (0-{len(elements)-1})")
        return None
    el = elements[action.element]
    if el.selector.startswith("shadow:"):
        # Shadow DOM element — use the host's data-wnav-host attribute
        # to find the shadow root, then query inside it
        idx = el.selector.split(":")[1]
        # Use Playwright's >> pierce syntax for shadow DOM
        return page.locator(f'[data-wnav-host="{idx}"] >> [data-wnav="{idx}"]').first
    return page.locator(el.selector).first


async def _resolve_shadow_element_bbox(
    page: Page, selector: str,
) -> dict[str, float] | None:
    """Get bounding box for a shadow DOM element via JS evaluation.

    Fallback when Playwright's pierce selector doesn't work.
    """
    idx = selector.split(":")[1]
    return await page.evaluate(f"""
        (() => {{
            const host = document.querySelector('[data-wnav-host="{idx}"]');
            if (!host) return null;
            const sr = host.shadowRoot || host.__shadow;
            if (!sr) return null;
            const el = sr.querySelector('[data-wnav="{idx}"]');
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {{x: r.x, y: r.y, width: r.width, height: r.height}};
        }})()
    """)


async def _unhide_noise_parent(page: Page, selector: str) -> None:
    """If the target element or any ancestor is noise-hidden, unhide it.

    The indexer captures elements inside clean_page-hidden containers
    (it temporarily unhides during indexing). For execution, we need
    to permanently unhide the specific container so Playwright can
    interact with the element.

    Also checks the element itself — clean_page Phase 3 marks some
    floating divs directly with data-wnav-noise (not just parents).
    """
    await page.evaluate(f"""
        (() => {{
            const el = document.querySelector('{selector}');
            if (!el) return;
            // Check the element itself first
            if (el.hasAttribute('data-wnav-noise')) {{
                el.style.removeProperty('display');
            }}
            // Walk up ancestors
            let parent = el.parentElement;
            while (parent) {{
                if (parent.hasAttribute('data-wnav-noise')) {{
                    parent.style.removeProperty('display');
                    break;
                }}
                parent = parent.parentElement;
            }}
        }})()
    """)


async def run(
    page: Page, action: Action, elements: list[ElementInfo] | None = None
) -> bool:
    """Execute a single action on the page. Returns True if successful."""
    elems = elements or []

    # Unhide noise container if the target element is inside one.
    # The indexer captures elements inside hidden containers but they
    # need to be visible for Playwright interaction.
    if action.element is not None and 0 <= action.element < len(elems):
        sel = elems[action.element].selector
        if sel:
            await _unhide_noise_parent(page, sel)

    try:
        match action.type:
            case "click":
                return await _do_click(page, action, elems)
            case "fill":
                return await _do_fill(page, action, elems)
            case "scroll":
                return await _do_scroll(page, action, elems)
            case "wait":
                # Page timers run 10x faster (add_init_script acceleration),
                # so scale real wait accordingly: amount/10 + buffer for
                # React re-render + DOM update overhead.
                await asyncio.sleep(min(action.amount / 10 + 0.5, 2.0))
                return True
            case "press":
                await page.keyboard.press(action.value)
                return True
            case "js":
                import asyncio as _aio
                js_code = action.value
                # Auto-wrap in IIFE if code has bare return statements
                if "return " in js_code and not js_code.strip().startswith("("):
                    js_code = f"(() => {{ {js_code} }})()"
                try:
                    result = await _aio.wait_for(
                        page.evaluate(js_code), timeout=8.0
                    )
                    if result:
                        print(f"[executor] JS returned: {str(result)[:100]}")
                except _aio.TimeoutError:
                    print("[executor] JS timed out after 8s")
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
        await page.wait_for_timeout(50)

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

        await page.wait_for_timeout(250)
        return True

    except Exception as e:
        print(f"[executor] Code submission failed: {e}")
        return False


async def _do_click(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Click element by index, falling back to selector-based strategies."""
    n_clicks = max(action.amount, 1)

    # Layer button: find and click the level/reveal button via Playwright
    if action.value == "layer_button":
        try:
            for selector in [
                "button:has-text('Enter')",
                "button:has-text('Next')",
            ]:
                try:
                    loc = page.locator(selector).first
                    if await loc.is_visible(timeout=300):
                        await loc.click(timeout=2000, force=True)
                        print(f"[executor] Layer click: {selector}")
                        await asyncio.sleep(0.25)
                        return True
                except Exception:
                    continue
            # Fallback: click any visible div with "level" or "shadow" in text
            for text_match in ["Shadow Level", "Level", "Layer"]:
                try:
                    loc = page.locator(f"div:has-text('{text_match}')").first
                    if await loc.is_visible(timeout=300):
                        await loc.click(timeout=2000, force=True)
                        print(f"[executor] Layer click: div '{text_match}'")
                        await asyncio.sleep(0.25)
                        return True
                except Exception:
                    continue
            print("[executor] Layer click: no target found")
            return False
        except Exception as e:
            print(f"[executor] Layer click failed: {e}")
            return False

    # Index-based: resolve element
    if action.element is not None:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                # Scroll into view first (force=True skips this)
                try:
                    await loc.scroll_into_view_if_needed(timeout=1000)
                except Exception:
                    pass
                for _ in range(n_clicks):
                    await loc.click(timeout=2000, force=True)
                    if n_clicks > 1:
                        await page.wait_for_timeout(100)
                # For non-standard elements (div, span, label, etc.),
                # also do a JS .click() — Playwright mouse events sometimes
                # don't trigger React handlers on these elements.
                if action.element < len(elements):
                    el_tag = elements[action.element].tag
                    el_sel = elements[action.element].selector
                    if el_tag in ("div", "span", "label", "section", "p") and el_sel:
                        try:
                            for _ in range(n_clicks):
                                await page.evaluate(
                                    f"document.querySelector('{el_sel}')?.click()"
                                )
                        except Exception:
                            pass
                return True
            except Exception as e:
                print(f"[executor] Index click failed: {e}")
                # Shadow DOM fallback: click via coordinates from JS bbox
                if action.element < len(elements) and elements[action.element].selector.startswith("shadow:"):
                    bbox = await _resolve_shadow_element_bbox(page, elements[action.element].selector)
                    if bbox:
                        cx = bbox["x"] + bbox["width"] / 2
                        cy = bbox["y"] + bbox["height"] / 2
                        for _ in range(n_clicks):
                            await page.mouse.click(cx, cy)
                            if n_clicks > 1:
                                await page.wait_for_timeout(200)
                        print(f"[executor] Shadow DOM click via coords: ({cx:.0f}, {cy:.0f})")
                        return True

        # Text-based fallback using the element's name
        if action.element < len(elements):
            el_info = elements[action.element]
            name = (el_info.name or "").strip()[:60]
            if name:
                for strategy in [
                    lambda: page.get_by_role("button", name=name, exact=False).first,
                    lambda: page.get_by_text(name, exact=False).first,
                ]:
                    try:
                        fallback_loc = strategy()
                        for _ in range(n_clicks):
                            await fallback_loc.click(timeout=2000, force=True)
                            if n_clicks > 1:
                                await page.wait_for_timeout(200)
                        print(f"[executor] Text fallback click succeeded: \"{name[:30]}\"")
                        return True
                    except Exception:
                        continue

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
    text = action.value or "hello"  # Never fill empty text
    # Index-based
    if action.element is not None:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                await loc.fill(text, timeout=3000)
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
                await loc.fill(text, timeout=3000)
                return True
        except (PlaywrightTimeout, Exception):
            continue
    return False


async def _do_scroll(
    page: Page, action: Action, elements: list[ElementInfo] | None = None,
) -> bool:
    """Scroll down by a pixel amount using mouse.wheel (triggers scroll events).

    If element is specified, positions the mouse over that element first so
    the wheel events scroll inside the container rather than the page.
    """
    total = action.amount or 600

    # Element-targeted scroll: position mouse over the element
    if action.element is not None and elements:
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                import asyncio as _aio
                box = await _aio.wait_for(loc.bounding_box(), timeout=3.0)
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    await page.mouse.move(cx, cy)
                    await page.wait_for_timeout(25)
                    step = 100
                    scrolled = 0
                    while scrolled < total:
                        chunk = min(step, total - scrolled)
                        await page.mouse.wheel(0, chunk)
                        scrolled += chunk
                        await page.wait_for_timeout(40)
                    return True
            except Exception as e:
                print(f"[executor] Element scroll failed: {e}")

    # Default: scroll the main page
    await page.mouse.move(640, 360)
    await page.wait_for_timeout(25)
    step = 100
    scrolled = 0
    while scrolled < total:
        chunk = min(step, total - scrolled)
        await page.mouse.wheel(0, chunk)
        scrolled += chunk
        await page.wait_for_timeout(40)
    return True


async def _do_hover(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Hover over an element by index with real Playwright mouse + JS event dispatch."""
    duration_secs = min(action.duration or 3, 10)  # Cap at 10s

    # Strategy 1: Playwright real hover (most reliable for React event listeners)
    hover_loc = None

    # Try index-based element first
    if action.element is not None and action.element < len(elements):
        loc = _resolve_element(page, action, elements)
        if loc:
            try:
                if await loc.is_visible(timeout=500):
                    hover_loc = loc
            except Exception:
                pass

    # Fallback: find hover target by common selectors
    if hover_loc is None:
        for selector in [
            "[class*='cursor-pointer']",
            "text='Hover here'",
            "text='Hover Here'",
            "text='Hover over'",
        ]:
            try:
                loc = page.locator(selector).first
                if await loc.is_visible(timeout=300):
                    hover_loc = loc
                    break
            except (PlaywrightTimeout, Exception):
                continue

    # Fallback: find any div/section with "hover" in text
    if hover_loc is None:
        try:
            box = await page.evaluate("""
                () => {
                    for (const el of document.querySelectorAll('div, section, span, p')) {
                        const text = (el.textContent || '').toLowerCase();
                        if ((text.includes('hover here') || text.includes('hover over'))
                            && el.children.length <= 3 && el.offsetParent !== null) {
                            const r = el.getBoundingClientRect();
                            if (r.width > 20 && r.height > 20) {
                                return {x: r.x + r.width/2, y: r.y + r.height/2, tag: el.tagName};
                            }
                        }
                    }
                    // Try any cursor-pointer element
                    for (const el of document.querySelectorAll('*')) {
                        const style = window.getComputedStyle(el);
                        if (style.cursor === 'pointer' && el.offsetParent !== null
                            && el.tagName !== 'BUTTON' && el.tagName !== 'A' && el.tagName !== 'INPUT') {
                            const r = el.getBoundingClientRect();
                            if (r.width > 30 && r.height > 30 && r.width < 500) {
                                return {x: r.x + r.width/2, y: r.y + r.height/2, tag: el.tagName};
                            }
                        }
                    }
                    return null;
                }
            """)
            if box:
                # Use raw mouse positioning for sustained hover
                await page.mouse.move(box["x"], box["y"])
                print(f"[executor] Hover: mouse at ({box['x']:.0f}, {box['y']:.0f}) on {box['tag']}")
                # Hold position with micro-movements for duration
                for i in range(int(duration_secs * 5)):
                    await asyncio.sleep(0.2)
                    await page.mouse.move(box["x"] + (i % 3) - 1, box["y"])
                print(f"[executor] Hover: held {duration_secs}s via mouse positioning")
                return True
        except Exception as e:
            print(f"[executor] Hover coordinate fallback failed: {e}")

    if hover_loc:
        try:
            await hover_loc.hover(force=True)
            # Sustained hover with micro-movements
            import asyncio as _aio
            box = await _aio.wait_for(hover_loc.bounding_box(), timeout=3.0)
            if box:
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                for i in range(int(duration_secs * 5)):
                    await asyncio.sleep(0.2)
                    await page.mouse.move(cx + (i % 3) - 1, cy)
            else:
                await asyncio.sleep(duration_secs)
            print(f"[executor] Hover: Playwright hover held {duration_secs}s")
            return True
        except Exception as e:
            print(f"[executor] Playwright hover failed: {e}")

    # JS event dispatch fallback
    duration_ms = int(duration_secs * 1000)
    target_selector = None
    if action.element is not None and action.element < len(elements):
        target_selector = elements[action.element].selector

    hover_js = f"""
    (async () => {{
        let target = null;
        const sel = {json.dumps(target_selector) if target_selector else 'null'};
        if (sel) target = document.querySelector(sel);
        if (!target) {{
            for (const el of document.querySelectorAll('div, section, span, p')) {{
                const text = (el.textContent || '').toLowerCase();
                if ((text.includes('hover here') || text.includes('hover over'))
                    && el.children.length <= 3 && el.offsetParent !== null) {{
                    target = el; break;
                }}
            }}
        }}
        if (!target) return 'hover: no target found';
        const rect = target.getBoundingClientRect();
        const cx = rect.x + rect.width / 2;
        const cy = rect.y + rect.height / 2;
        ['pointerenter','pointerover','mouseenter','mouseover','mousemove'].forEach(evt => {{
            target.dispatchEvent(new PointerEvent(evt, {{
                bubbles:true, cancelable:true, clientX:cx, clientY:cy, pointerType:'mouse'
            }}));
        }});
        let elapsed = 0;
        while (elapsed < {duration_ms}) {{
            await new Promise(r => setTimeout(r, 100));
            elapsed += 100;
            target.dispatchEvent(new PointerEvent('pointermove', {{
                bubbles:true, cancelable:true, clientX:cx+(elapsed%3), clientY:cy, pointerType:'mouse'
            }}));
        }}
        return 'hover: JS held ' + {duration_ms} + 'ms on ' + target.tagName;
    }})()
    """
    try:
        result = await page.evaluate(hover_js)
        print(f"[executor] Hover: {result}")
        return True
    except Exception as e:
        print(f"[executor] JS hover failed: {e}")
    return False


async def _do_drag(page: Page, action: Action, elements: list[ElementInfo]) -> bool:
    """Drag-and-drop: JS DragEvent dispatch + Playwright mouse fallback."""
    try:
        # Strategy 1: JS DragEvent dispatch
        filled = await page.evaluate(_DRAG_JS)
        print(f"[executor] Drag JS strategy: {filled}/6 slots filled")
        if filled >= 6:
            return True

        # Strategy 2: Mouse simulation (skip Playwright drag_to — too slow)
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
            await page.wait_for_timeout(50)
            await page.mouse.down()
            await page.wait_for_timeout(50)
            for step in range(1, 11):
                x = px + (sx - px) * step / 10
                y = py + (sy - py) * step / 10
                await page.mouse.move(x, y)
                await page.wait_for_timeout(15)
            await page.mouse.up()
            await page.wait_for_timeout(100)

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
    """Draw strokes on a canvas element via JS event dispatch.

    Uses dispatchEvent with MouseEvents directly on the canvas element.
    This works reliably in both headed and headless mode because it
    bypasses Playwright's mouse hit-testing (which can miss the canvas
    in headless Chromium when overlays affect z-stacking).
    """
    try:
        result = await page.evaluate(_CANVAS_DRAW_JS)
        print(f"[executor] Canvas draw: {result}")
        return "ok" in str(result).lower()
    except Exception as e:
        print(f"[executor] Canvas draw failed: {e}")
        return False


_CANVAS_DRAW_JS = """
(async () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return 'no canvas found';
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return 'canvas too small: ' + rect.width + 'x' + rect.height;

    const fire = (type, x, y) => {
        canvas.dispatchEvent(new MouseEvent(type, {
            bubbles: true, cancelable: true, clientX: x, clientY: y
        }));
    };

    // Draw 4 strokes — each mousedown + series of mousemove + mouseup
    const strokes = [
        {y: 0.25, x0: 0.1, x1: 0.9},
        {y: 0.50, x0: 0.9, x1: 0.1},
        {y: 0.75, x0: 0.1, x1: 0.9},
        {y: 0.40, x0: 0.2, x1: 0.8},
    ];
    for (const s of strokes) {
        const sy = rect.top + rect.height * s.y;
        const sx = rect.left + rect.width * s.x0;
        const ex = rect.left + rect.width * s.x1;

        fire('mousedown', sx, sy);
        // Allow React state update (isDrawing = true) to flush
        await new Promise(r => (window.__origST||setTimeout)(r, 50));

        for (let i = 1; i <= 8; i++) {
            const x = sx + (ex - sx) * i / 8;
            fire('mousemove', x, sy);
            await new Promise(r => (window.__origST||setTimeout)(r, 20));
        }

        fire('mouseup', ex, sy);
        // Allow React to increment stroke count
        await new Promise(r => (window.__origST||setTimeout)(r, 100));
    }
    return 'ok: 4 strokes drawn on ' + rect.width + 'x' + rect.height + ' canvas';
})()
"""


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
    """Read required key sequence from page text and press each key.

    Handles progressive sequences where the page reveals one key at a time:
    read → press → re-read → press new keys → repeat.
    """
    try:
        pressed = 0
        all_keys: list[str] = []

        # Iterative: extract, press new keys, re-read (handles progressive reveals)
        for _round in range(8):
            text = await page.inner_text("body")
            keys = _extract_key_sequence(text)

            if not keys and not all_keys and action.value:
                # Fallback: LLM-provided keys if page extraction found nothing
                keys = _parse_key_tokens(action.value)

            if len(keys) <= pressed:
                # No new keys appeared — done or stuck
                if _round > 0:
                    break
                # First round with no keys — try LLM keys
                if action.value and not keys:
                    keys = _parse_key_tokens(action.value)
                if not keys:
                    print("[executor] Key sequence: could not extract keys from page")
                    return False

            # Press only newly revealed keys
            new_keys = keys[pressed:]
            print(f"[executor] Key sequence round {_round + 1}: pressing {new_keys}")
            for key in new_keys:
                await page.keyboard.press(key)
                await page.wait_for_timeout(150)
            pressed = len(keys)
            all_keys = keys
            await page.wait_for_timeout(200)  # Wait for page to update

        print(f"[executor] Key sequence: pressed {pressed} keys total")
        return True
    except Exception as e:
        print(f"[executor] Key sequence failed: {e}")
        return False


def _extract_key_sequence(text: str) -> list[str]:
    """Extract key names from page text."""
    keys: list[str] = []
    in_sequence = False
    blank_lines_since_key = 0
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"(?:required\s+)?sequence\s*:", stripped, re.IGNORECASE):
            in_sequence = True
            after = re.split(r"sequence\s*:\s*", stripped, flags=re.IGNORECASE)[-1]
            if after:
                keys.extend(_parse_key_tokens(after))
            blank_lines_since_key = 0
            continue
        if in_sequence:
            if not stripped:
                blank_lines_since_key += 1
                if blank_lines_since_key > 2 and keys:
                    break
                continue
            parsed = _parse_key_tokens(stripped)
            if parsed:
                keys.extend(parsed)
                blank_lines_since_key = 0
            else:
                blank_lines_since_key += 1
                # Only stop after 3 non-key lines with at least some keys found
                if blank_lines_since_key > 3 and keys:
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
