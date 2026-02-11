"""Tier 0: Regex-based programmatic challenge dispatch.

Parses the challenge instruction text and returns a list of actions
without requiring an LLM call. Returns None if no pattern matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """A browser action to execute."""

    type: str  # click, fill, scroll, wait, press, js
    selector: str = ""
    value: str = ""
    amount: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


# Ordered list of (pattern, action_factory) pairs.
# First match wins. Patterns are tested against lowercased instruction text.
# IMPORTANT: More specific patterns MUST come before generic ones.
# "hidden/inspect dom" before "click" because hidden challenges may mention "click" as alt.
_PATTERNS: list[tuple[re.Pattern[str], Any]] = [
    # "wait N seconds" — sleep with buffer for timer misalignment
    (
        re.compile(r"wait.*?(\d+)\s*second"),
        lambda m: [Action(type="wait", amount=int(m.group(1)) + 2)],
    ),
    # "scroll down at least Npx" or "scroll down N pixels"
    (
        re.compile(r"scroll.*?(\d+)\s*(?:px|pixel)"),
        lambda m: [Action(type="scroll", amount=int(m.group(1)) + 100)],
    ),
    # "scroll down" (no specific amount)
    (
        re.compile(r"scroll\s+down"),
        lambda _: [Action(type="scroll", amount=600)],
    ),
    # "hidden" / "inspect the dom" / "find the hidden" — JS reveal + click cursor-pointer
    # MUST be before "click" patterns since hidden challenges often say "or click to reveal"
    (
        re.compile(r"(?:hidden.*(?:dom|element|code)|inspect.*dom|find.*hidden)"),
        lambda _: [
            Action(type="js", value=_HIDDEN_DOM_JS),
            Action(type="js", value=_HIDDEN_CLICK_JS),
        ],
    ),
    # "drag" + "slot" — Playwright mouse-based drag-and-drop
    (
        re.compile(r"(?:drag|slot|fill.*slot|piece)"),
        lambda _: [Action(type="drag_fill")],
    ),
    # "memory challenge" — wait for flash, then click "I Remember"
    (
        re.compile(r"memory.*challenge|flash.*second|watch.*carefully"),
        lambda _: [
            Action(type="wait", amount=3),
            Action(type="click", selector="button:has-text('Remember'), button:has-text('remember')"),
        ],
    ),
    # --- Multi-step challenge types (MUST be before generic hover/click) ---

    # "timing challenge" — wait for window + click Capture
    (
        re.compile(r"timing.*challenge|click.*capture.*window"),
        lambda _: [
            Action(type="wait", amount=3),
            Action(type="click", selector="button:has-text('Capture')"),
        ],
    ),
    # "gesture challenge" — draw on canvas + click Complete
    # MUST be before canvas pattern (gesture instructions mention "canvas" and "draw")
    (
        re.compile(r"gesture.*challenge|draw.*(?:circle|square|triangle|zigzag)"),
        lambda _: [
            Action(type="canvas_draw"),
            Action(type="click", selector="button:has-text('Complete')"),
        ],
    ),
    # "canvas challenge" — draw strokes + click Reveal Code
    (
        re.compile(r"canvas.*(?:draw|stroke)|draw.*(?:stroke|canvas)"),
        lambda _: [
            Action(type="canvas_draw"),
            Action(type="click", selector="button:has-text('Reveal Code'), button:has-text('Complete'), button:has-text('Reveal')"),
        ],
    ),
    # "audio challenge" — click Play Audio, wait, click Complete
    (
        re.compile(r"audio.*challenge|play.*audio.*hint|listen.*hint"),
        lambda _: [
            Action(type="click", selector="button:has-text('Play Audio'), button:has-text('Play')"),
            Action(type="wait", amount=4),
            Action(type="click", selector="button:has-text('Complete')"),
        ],
    ),
    # "video challenge" — seek 3+ times via buttons, then Complete
    (
        re.compile(r"video.*challenge|seek.*video.*frame"),
        lambda _: [
            Action(type="click", selector="button:has-text('Frame')"),
            Action(type="wait", amount=1),
            Action(type="click", selector="button:has-text('+1')"),
            Action(type="click", selector="button:has-text('-1')"),
            Action(type="click", selector="button:has-text('Complete')"),
        ],
    ),
    # "multi-tab challenge" — click all tab buttons, then reveal
    (
        re.compile(r"multi.*tab.*challenge|visit.*tab"),
        lambda _: [Action(type="js", value=_MULTI_TAB_JS)],
    ),
    # "sequence challenge" — complete all 4 actions + click Complete
    # MUST NOT match "Keyboard Sequence Challenge" — use negative lookbehind
    # Uses JS to do all 4 actions (React synthetic events), then Playwright backup for
    # actions that may need real browser events (hover, input, scroll).
    (
        re.compile(r"(?<!keyboard\s)sequence.*challenge|complete.*all.*actions.*reveal"),
        lambda _: [
            Action(type="js", value=_SEQUENCE_ALL_JS),  # Try all 4 via JS first
            Action(type="click", selector="button:has-text('Click Me')"),  # Backup
            Action(type="hover", selector="div:has-text('Hover over this area'), [class*='cursor-pointer']"),
            Action(type="wait", amount=1),  # Hold hover for React to detect
            Action(type="fill", selector="input:not([placeholder*='code' i]):not([placeholder*='enter' i])", value="hello world"),
            Action(type="scroll_element"),
            Action(type="js", value=_SCROLL_BOX_JS),
            Action(type="click", selector="button:has-text('Complete')"),
        ],
    ),
    # "shadow dom challenge" — click through nested levels
    (
        re.compile(r"shadow.*dom.*challenge|navigate.*nested.*layer"),
        lambda _: [Action(type="js", value=_SHADOW_DOM_JS)],
    ),
    # "websocket challenge" — connect + wait + reveal
    (
        re.compile(r"websocket.*challenge|connect.*(?:simulated|server).*(?:receive|code)"),
        lambda _: [
            Action(type="click", selector="button:has-text('Connect')"),
            Action(type="wait", amount=4),
            Action(type="click", selector="button:has-text('Reveal Code'), button:has-text('Reveal')"),
        ],
    ),
    # "service worker challenge" — register + wait + retrieve
    (
        re.compile(r"service.*worker.*challenge|register.*service.*worker"),
        lambda _: [
            Action(type="click", selector="button:has-text('Register')"),
            Action(type="wait", amount=5),
            Action(type="click", selector="button:has-text('Retrieve'), button:has-text('Cache')"),
            Action(type="wait", amount=2),
            Action(type="click", selector="button:has-text('Retrieve'), button:has-text('Cache')"),
        ],
    ),
    # "mutation challenge" — trigger N mutations + complete
    (
        re.compile(r"mutation.*challenge|trigger.*(?:dom\s+)?mutation"),
        lambda _: [
            Action(type="click", selector="button:has-text('Trigger')", amount=6),
            Action(type="click", selector="button:has-text('Complete')"),
        ],
    ),
    # "recursive iframe challenge" — click through levels + extract
    (
        re.compile(r"recursive.*iframe|navigate.*nested.*level"),
        lambda _: [Action(type="js", value=_RECURSIVE_IFRAME_JS)],
    ),
    # "puzzle challenge" — solve math + fill answer + click solve + reveal
    # Includes comprehensive fallbacks: JS solve → click → keyboard answer → form submit
    # NOTE: Wait 2s first because the puzzle UI renders in a useEffect (post-mount).
    # Without this wait, the puzzle answer input doesn't exist yet in the DOM.
    (
        re.compile(r"puzzle.*challenge|solve.*puzzle"),
        lambda _: [
            Action(type="wait", amount=2),
            Action(type="js", value=_PUZZLE_JS),
            Action(type="click", selector="button:has-text('Solve'), button:has-text('Check'), button:has-text('Verify')"),
            Action(type="js", value=_PUZZLE_DIRECT_SOLVE_JS),
            Action(type="js", value=_PUZZLE_FALLBACK_JS),
        ],
    ),
    # "split parts challenge" — click all parts
    (
        re.compile(r"split.*parts|find.*click.*parts|parts.*scattered"),
        lambda _: [Action(type="js", value=_SPLIT_PARTS_JS)],
    ),
    # "encoded/base64 challenge" — decode + fill
    (
        re.compile(r"(?:encoded|base64).*challenge|decode.*base64"),
        lambda _: [Action(type="js", value=_ENCODED_BASE64_JS)],
    ),
    # "rotating code challenge" — click Capture 3 times
    (
        re.compile(r"rotating.*challenge|code.*changes.*second.*capture"),
        lambda _: [
            Action(type="click", selector="button:has-text('Capture')", amount=3),
        ],
    ),
    # "obfuscated challenge" — reverse the shown code + fill + click
    (
        re.compile(r"obfuscated.*challenge|code.*obfuscated.*decode"),
        lambda _: [Action(type="js", value=_OBFUSCATED_JS)],
    ),
    # --- Generic patterns ---

    # "hover" — JS-based continuous hover + Playwright backup
    # Uses JS to dispatch sustained mouse events that bypass overlay interference.
    # The JS fires events for the full duration inside the browser, so overlays
    # regenerating during asyncio.sleep() can't interrupt the hover state.
    (
        re.compile(r"hover.*?(\d+)\s*second"),
        lambda m: [
            Action(type="js", value=_HOVER_SUSTAINED_JS.replace("DURATION", str((int(m.group(1)) + 1) * 1000))),
            Action(type="hover", selector="div:has-text('Hover here'), .cursor-pointer, [class*='hover']"),
            Action(type="wait", amount=int(m.group(1)) + 2),
        ],
    ),
    # "hover" (no specific duration)
    (
        re.compile(r"hover"),
        lambda _: [
            Action(type="js", value=_HOVER_SUSTAINED_JS.replace("DURATION", "3000")),
            Action(type="hover", selector="div:has-text('Hover here'), .cursor-pointer, [class*='hover']"),
            Action(type="wait", amount=4),
        ],
    ),
    # "click the button N times"
    (
        re.compile(r"click.*?(\d+)\s*time"),
        lambda m: [Action(type="click", selector="button:has-text('Reveal'), button:has-text('Click'), button:has-text('challenge')", amount=int(m.group(1)))],
    ),
    # "click 'ButtonName'" — extract quoted button name
    (
        re.compile(r"""click\s+["\u201c](.+?)["\u201d]"""),
        lambda m: [Action(type="click", selector=f"button:has-text('{m.group(1)}')")],
    ),
    # "click the button below to reveal" — specific click-to-reveal
    (
        re.compile(r"click.*button.*reveal|click.*below.*reveal"),
        lambda _: [Action(type="click", selector="button:has-text('Reveal')")],
    ),
    # "click ... to reveal" — generic
    (
        re.compile(r"click.*(?:to\s+)?reveal"),
        lambda _: [Action(type="click", selector="button:has-text('Reveal')")],
    ),
    # "click the button" (generic)
    (
        re.compile(r"click.*button"),
        lambda _: [Action(type="click", selector="button:has-text('Reveal'), button:has-text('Click'), button:has-text('Show')")],
    ),
    # "type" or "enter" a specific text
    (
        re.compile(r"(?:type|enter)\s+[\"'](.+?)[\"']"),
        lambda m: [Action(type="fill", value=m.group(1))],
    ),
    # "keyboard sequence" or "press N keys in sequence" — read keys from page
    (
        re.compile(r"(?:keyboard|key).*sequence|press.*\d+\s*keys?\s*(?:in\s+)?sequence"),
        lambda _: [Action(type="key_sequence")],
    ),
    # "press" a specific key — capitalize first letter for Playwright key names
    (
        re.compile(r"press\s+(?:the\s+)?[\"']?(\w+)[\"']?"),
        lambda m: [Action(type="press", value=m.group(1).capitalize())],
    ),
    # "select" an option
    (
        re.compile(r"select.*[\"'](.+?)[\"']"),
        lambda m: [Action(type="select", value=m.group(1))],
    ),
]


# JS for sustained hover — fires mouse events continuously for DURATION ms
# inside the browser's event loop, so overlays regenerating can't interrupt.
# DURATION is replaced at dispatch time.
_HOVER_SUSTAINED_JS = """
(async () => {
    // Find the hover target element
    let target = null;
    // Strategy 1: element with cursor-pointer class containing "hover"
    for (const el of document.querySelectorAll('[class*="cursor-pointer"]')) {
        const text = (el.textContent || '').toLowerCase();
        if (text.includes('hover') && el.offsetParent !== null) {
            target = el; break;
        }
    }
    // Strategy 2: any element with "hover here" text (prefer leaf-ish)
    if (!target) {
        for (const el of document.querySelectorAll('div, p, span, section')) {
            const text = (el.textContent || '').toLowerCase();
            if ((text.includes('hover here') || text.includes('hover over this'))
                && el.children.length <= 3 && el.offsetParent !== null) {
                target = el; break;
            }
        }
    }
    if (!target) return 'hover: no target found';
    // Remove any overlays covering the target before firing events
    const rect = target.getBoundingClientRect();
    const cx = rect.x + rect.width / 2;
    const cy = rect.y + rect.height / 2;
    // Fire initial hover events
    const events = ['pointerenter', 'pointerover', 'mouseenter', 'mouseover', 'mousemove'];
    events.forEach(evt => {
        target.dispatchEvent(new PointerEvent(evt, {
            bubbles: true, cancelable: true,
            clientX: cx, clientY: cy, pointerType: 'mouse'
        }));
    });
    // Keep firing mousemove every 100ms for the full duration
    const duration = DURATION;
    const interval = 100;
    let elapsed = 0;
    while (elapsed < duration) {
        await new Promise(r => setTimeout(r, interval));
        elapsed += interval;
        target.dispatchEvent(new PointerEvent('pointermove', {
            bubbles: true, cancelable: true,
            clientX: cx + (elapsed % 3), clientY: cy, pointerType: 'mouse'
        }));
        target.dispatchEvent(new MouseEvent('mousemove', {
            bubbles: true, cancelable: true,
            clientX: cx + (elapsed % 3), clientY: cy
        }));
    }
    return 'hover: held ' + duration + 'ms on ' + target.tagName;
})()
"""

# JS for hover challenges — find hover target and dispatch mouseenter/mouseover
_HOVER_JS = """
(() => {
    // Find the hover target element
    const targets = document.querySelectorAll('[class*="hover"], [class*="cursor"]');
    let target = null;
    for (const el of targets) {
        const text = (el.textContent || '').toLowerCase();
        if (text.includes('hover') && el.offsetParent !== null) {
            target = el;
            break;
        }
    }
    // Fallback: find any element with "hover here" text
    if (!target) {
        const all = document.querySelectorAll('div, p, span, section');
        for (const el of all) {
            if ((el.textContent || '').toLowerCase().includes('hover here') && el.offsetParent !== null) {
                target = el;
                break;
            }
        }
    }
    if (target) {
        // Dispatch mouse events
        target.dispatchEvent(new MouseEvent('mouseenter', {bubbles: true}));
        target.dispatchEvent(new MouseEvent('mouseover', {bubbles: true}));
        target.dispatchEvent(new MouseEvent('mousemove', {bubbles: true, clientX: 100, clientY: 100}));
        // Hold hover for specified duration
        setTimeout(() => {
            target.dispatchEvent(new MouseEvent('mousemove', {bubbles: true, clientX: 101, clientY: 101}));
        }, DURATION);
    }
})()
"""

# JS for hidden DOM multi-click — directly clicks .cursor-pointer via JS
# This bypasses overlay pointer-events (Playwright click can't when overlay intercepts)
_HIDDEN_CLICK_JS = """
(() => {
    // Strategy 1: Direct click on .cursor-pointer element
    const cp = document.querySelector('.cursor-pointer');
    if (cp) {
        for (let i = 0; i < 10; i++) {
            cp.click();
            cp.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
        }
    }
    // Strategy 2: Find and click "click here" text elements
    const allEls = document.querySelectorAll('span, a, p, div, button, strong, em');
    for (const el of allEls) {
        const text = (el.textContent || '').toLowerCase();
        if (text.includes('click here') && text.includes('more times')) {
            for (let i = 0; i < 10; i++) {
                el.click();
            }
            break;
        }
    }
})()
"""

# JS for hidden DOM challenges — reveal all hidden elements
_HIDDEN_DOM_JS = """
(() => {
    const codeRe = /^[A-Z0-9]{6}$/;
    document.querySelectorAll('*').forEach(el => {
        const s = getComputedStyle(el);
        if (s.display === 'none') el.style.display = 'block';
        if (s.visibility === 'hidden') el.style.visibility = 'visible';
        if (parseFloat(s.opacity) < 0.1) el.style.opacity = '1';
        const text = (el.textContent || '').trim();
        if (codeRe.test(text) && el.children.length === 0) {
            el.style.cssText = 'display:block !important; visibility:visible !important; ' +
                'opacity:1 !important; width:auto !important; height:auto !important; ' +
                'font-size:16px !important; color:red !important; position:static !important; ' +
                'overflow:visible !important;';
        }
    });
    document.querySelectorAll('*').forEach(el => {
        for (const attr of el.attributes) {
            if (codeRe.test(attr.value)) {
                const div = document.createElement('div');
                div.textContent = attr.value;
                div.style.cssText = 'display:block; color:red; font-size:20px;';
                document.body.appendChild(div);
                return;
            }
        }
    });
})()
"""

# JS for drag-and-drop challenges — fill all slots programmatically
_DRAG_DROP_JS = """
(() => {
    // Find draggable pieces and drop slots
    const pieces = document.querySelectorAll('[draggable="true"], .piece, .draggable, [class*="drag"]');
    const slots = document.querySelectorAll('.slot, .drop-zone, .dropzone, [class*="slot"], [class*="drop"]');

    if (pieces.length && slots.length) {
        // Use DataTransfer API to simulate drag-and-drop
        slots.forEach((slot, i) => {
            const piece = pieces[i % pieces.length];
            if (!piece) return;

            const dt = new DataTransfer();
            const dragStart = new DragEvent('dragstart', { dataTransfer: dt, bubbles: true });
            piece.dispatchEvent(dragStart);

            const dragOver = new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true });
            slot.dispatchEvent(dragOver);

            const drop = new DragEvent('drop', { dataTransfer: dt, bubbles: true });
            slot.dispatchEvent(drop);

            const dragEnd = new DragEvent('dragend', { dataTransfer: dt, bubbles: true });
            piece.dispatchEvent(dragEnd);
        });
    }

    // Also try React-style state mutation (sometimes React prevents native drag)
    // Look for any "onDrop" handlers via React internals
    const reactKey = Object.keys(document.querySelector('[class*="slot"]') || {}).find(k => k.startsWith('__reactFiber') || k.startsWith('__reactInternalInstance'));
    if (reactKey) {
        slots.forEach((slot, i) => {
            const piece = pieces[i % pieces.length];
            if (!piece || !slot) return;
            // Clone piece content into slot
            slot.innerHTML = piece.innerHTML;
            slot.classList.add('filled');
            // Trigger React change detection
            const ev = new Event('input', { bubbles: true });
            slot.dispatchEvent(ev);
        });
    }
})()
"""


# JS for sequence: scroll inside the scroll box to trigger onScroll
_SCROLL_BOX_JS = """
(() => {
    const sels = ['.overflow-y-scroll', '[class*="overflow-y-scroll"]',
                  '[class*="overflow-auto"]', '[style*="overflow"]'];
    for (const sel of sels) {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
            if (el.scrollHeight > el.clientHeight) {
                // Scroll to bottom in steps to trigger React's onScroll
                for (let i = 0; i < 5; i++) {
                    el.scrollTop += 100;
                }
                el.scrollTop = el.scrollHeight;
            }
        }
    }
})()
"""

# JS for multi-tab: click all tab buttons then reveal
_MULTI_TAB_JS = """
(() => {
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
        const text = (btn.textContent || '').trim();
        if (/^Tab \\d+$/.test(text) && !btn.disabled) {
            btn.click();
        }
    }
    // Click already-visited tab buttons too (they have ✓)
    for (const btn of buttons) {
        const text = (btn.textContent || '').trim();
        if (/Tab \\d+/.test(text) && !btn.disabled) {
            btn.click();
        }
    }
})()
"""

# JS for sequence: perform all 4 actions comprehensively via JS events
# Returns progress string for diagnostics
_SEQUENCE_ALL_JS = """
(() => {
    // Helper: get current progress number
    const getProgress = () => {
        for (const el of document.querySelectorAll('*')) {
            const t = (el.textContent || '').trim();
            const m = t.match(/^Progress:\\s*(\\d+)\\/(\\d+)$/);
            if (m) return parseInt(m[1]);
        }
        return -1;
    };
    const before = getProgress();

    // 1. CLICK the "Click Me" button
    for (const btn of document.querySelectorAll('button')) {
        if ((btn.textContent || '').trim() === 'Click Me' ||
            (btn.textContent || '').includes('Click Me')) {
            btn.click();
            break;
        }
    }

    // 2. HOVER over the target area — fire all mouse events React might listen for
    for (const el of document.querySelectorAll('div, section, p')) {
        const t = (el.textContent || '').trim().toLowerCase();
        // Match the exact hover target (leaf-ish element)
        if ((t.includes('hover over this') || t.includes('hover here')) &&
            el.children.length <= 2) {
            ['mouseenter', 'mouseover', 'mousemove', 'pointerenter',
             'pointerover', 'pointermove'].forEach(evt => {
                el.dispatchEvent(new PointerEvent(evt, {
                    bubbles: true, cancelable: true,
                    clientX: 300, clientY: 300, pointerType: 'mouse'
                }));
            });
            break;
        }
    }

    // 3. TYPE in the input field — use nativeSetter for React controlled inputs
    for (const input of document.querySelectorAll('input')) {
        const ph = (input.placeholder || '').toLowerCase();
        // Skip code submission inputs
        if (ph.includes('code') || ph.includes('enter 6') || ph.includes('character'))
            continue;
        // Target any other input (type, click, text, etc.)
        if (ph || input.type === 'text') {
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            nativeSetter.call(input, 'hello world');
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
            input.focus();
            input.dispatchEvent(new FocusEvent('focus', {bubbles: true}));
            // Simulate keystrokes
            'hello'.split('').forEach(ch => {
                input.dispatchEvent(new KeyboardEvent('keydown', {key: ch, bubbles: true}));
                input.dispatchEvent(new KeyboardEvent('keypress', {key: ch, bubbles: true}));
                input.dispatchEvent(new KeyboardEvent('keyup', {key: ch, bubbles: true}));
            });
            break;
        }
    }

    // 4. SCROLL the scrollable container
    const scrollSels = ['.overflow-y-scroll', '[class*="overflow-y-scroll"]',
                        '[class*="overflow-auto"]', '[class*="overflow-y-auto"]'];
    for (const sel of scrollSels) {
        for (const el of document.querySelectorAll(sel)) {
            if (el.scrollHeight > el.clientHeight + 10) {
                // Scroll progressively to trigger onScroll handlers
                const target = el.scrollHeight;
                for (let i = 0; i < 20; i++) {
                    el.scrollTop += 50;
                    el.dispatchEvent(new Event('scroll', {bubbles: true}));
                }
                el.scrollTop = target;
                el.dispatchEvent(new Event('scroll', {bubbles: true}));
            }
        }
    }

    const after = getProgress();
    return `progress: ${before} -> ${after}`;
})()
"""

# JS for sequence (legacy): perform all 4 actions (click, hover, type, scroll)
_SEQUENCE_JS = """
(() => {
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
        const text = (btn.textContent || '').toLowerCase();
        if (text.includes('click me')) btn.click();
    }
    // Hover area
    const divs = document.querySelectorAll('div');
    for (const div of divs) {
        const text = (div.textContent || '').toLowerCase();
        if (text.includes('hover over this area') && div.className.includes('cursor-pointer')) {
            div.dispatchEvent(new MouseEvent('mouseenter', {bubbles: true}));
        }
    }
    // Focus input
    const inputs = document.querySelectorAll('input');
    for (const input of inputs) {
        if (input.placeholder && input.placeholder.toLowerCase().includes('type')) {
            input.focus();
            input.dispatchEvent(new FocusEvent('focus', {bubbles: true}));
        }
    }
    // Scroll box
    const scrollers = document.querySelectorAll('[class*="overflow-y-scroll"], [class*="overflow-auto"]');
    for (const el of scrollers) {
        el.scrollTop = el.scrollHeight;
        el.dispatchEvent(new Event('scroll', {bubbles: true}));
    }
})()
"""

# JS for shadow DOM: click levels 1, 2, 3 sequentially
_SHADOW_DOM_JS = """
(() => {
    // Click all level divs in order
    const all = document.querySelectorAll('div');
    const levels = [];
    for (const div of all) {
        const h = div.querySelector('h4, h5, h6');
        if (h && /Shadow Level \\d/.test(h.textContent || '')) {
            levels.push(div);
        }
    }
    // Click each level (they only appear after parent is clicked)
    for (const level of levels) {
        level.click();
    }
    // Also try clicking any cursor-pointer divs with "Shadow Level"
    for (const div of all) {
        if (div.className.includes('cursor-pointer') && (div.textContent || '').includes('Shadow Level')) {
            div.click();
        }
    }
})()
"""

# JS for recursive iframe: click all "Enter Level" buttons
_RECURSIVE_IFRAME_JS = """
(() => {
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
        const text = (btn.textContent || '').trim();
        if (/Enter Level/i.test(text) && !btn.disabled) {
            btn.click();
        }
    }
    // Also click "Extract Code" if visible
    for (const btn of buttons) {
        if (/Extract Code/i.test(btn.textContent || '')) btn.click();
    }
})()
"""

# JS for puzzle: extract math expression, compute, fill answer
_PUZZLE_JS = """
(() => {
    // Find math expressions in various formats
    const texts = document.querySelectorAll('p, div, span, h1, h2, h3, h4, label, strong');
    let answer = null;
    let foundExpr = '';
    for (const el of texts) {
        const text = (el.textContent || '').trim();
        if (!text || text.length > 200) continue;
        // Format 1: "15 + 10 = ?"
        let m = text.match(/(\\d+)\\s*([+\\-*\\/x×])\\s*(\\d+)\\s*=\\s*\\?/);
        // Format 2: "What is 15 + 10?"
        if (!m) m = text.match(/(?:what is|calculate|solve|compute|evaluate)\\s*(\\d+)\\s*([+\\-*\\/x×])\\s*(\\d+)/i);
        // Format 3: "15 + 10" (short text that IS a math expression)
        if (!m && text.length < 30) m = text.match(/^\\s*(\\d+)\\s*([+\\-*\\/x×])\\s*(\\d+)\\s*$/);
        // Format 4: "Solve: 15 + 10"
        if (!m) m = text.match(/(?:solve|answer|puzzle)[:\\s]+(\\d+)\\s*([+\\-*\\/x×])\\s*(\\d+)/i);
        if (m) {
            const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
            foundExpr = `${a} ${op} ${b}`;
            if (op === '+') answer = a + b;
            else if (op === '-') answer = a - b;
            else if (op === '*' || op === 'x' || op === '×') answer = a * b;
            else if (op === '/') answer = Math.round(a / b);
            break;
        }
    }
    if (answer === null) {
        return 'puzzle: no math expression found';
    }
    // Unhide ALL elements that page_cleaner may have hidden.
    // page_cleaner uses display:none !important, so we must use !important to override.
    // Walk every element with inline display:none and restore it, prioritizing
    // elements containing inputs, buttons (Solve/Check), or puzzle content.
    document.querySelectorAll('[style]').forEach(el => {
        const s = el.style;
        if (s.display === 'none' || s.getPropertyValue('display') === 'none') {
            // Check if this element or children contain puzzle-relevant content
            const hasInput = el.querySelector('input, select, textarea');
            const hasButton = el.querySelector('button');
            const text = (el.textContent || '').toLowerCase();
            const hasPuzzle = /solve|check|verify|answer|puzzle|=\\s*\\?/.test(text);
            if (hasInput || hasButton || hasPuzzle) {
                s.setProperty('display', 'block', 'important');
                s.setProperty('visibility', 'visible', 'important');
                s.setProperty('opacity', '1', 'important');
            }
        }
    });
    // Also ensure all inputs are visible
    document.querySelectorAll('input').forEach(i => {
        i.style.setProperty('display', '', '');
        i.style.setProperty('visibility', 'visible', 'important');
        i.style.setProperty('opacity', '1', 'important');
    });
    // Find the puzzle answer input
    const allInputs = [...document.querySelectorAll('input:not([type="hidden"])')];
    let input = null;
    // Priority 1: type="number" or answer-specific placeholder
    input = allInputs.find(i => i.type === 'number' ||
        /answer|result|solution/i.test(i.placeholder || ''));
    // Priority 2: any input that's NOT the code submission input
    if (!input) {
        input = allInputs.find(i => {
            const ph = (i.placeholder || '').toLowerCase();
            return !ph.includes('code') && !ph.includes('character') && !ph.includes('enter 6');
        });
    }
    // Priority 3: if only one input exists, use it (puzzle might share the code input)
    if (!input && allInputs.length >= 1) {
        input = allInputs[0];
    }
    if (!input) {
        // Diagnostic: what IS on the page?
        const diag = allInputs.map(i => i.type + ':' + (i.placeholder || '').substring(0, 20) + ':vis=' + (i.offsetParent !== null)).join('|');
        const pinkEls = document.querySelectorAll('[class*="pink"]').length;
        const hiddenParents = (() => {
            // Check if any ancestor of a pink element is display:none
            for (const el of document.querySelectorAll('[class*="pink"]')) {
                let p = el.parentElement;
                while (p && p !== document.body) {
                    const s = getComputedStyle(p);
                    if (s.display === 'none') return 'parent_hidden:' + p.tagName + '.' + (p.className || '').substring(0, 30);
                    p = p.parentElement;
                }
            }
            return 'none_hidden';
        })();
        return 'puzzle: ' + foundExpr + ' = ' + answer + ' inputs=[' + diag + '] pinkEls=' + pinkEls + ' hidden=' + hiddenParents;
    }
    const nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
    ).set;
    nativeSetter.call(input, String(answer));
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
    // Try Enter key to submit
    input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', bubbles: true}));
    input.dispatchEvent(new KeyboardEvent('keypress', {key: 'Enter', code: 'Enter', bubbles: true}));
    input.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', code: 'Enter', bubbles: true}));
    // Click solve/check/submit button (check broader set of button labels)
    let clicked = '';
    const buttons = document.querySelectorAll('button');
    for (const btn of buttons) {
        if (/Solve|Check|Verify|Submit|Answer/i.test(btn.textContent || '') && !btn.disabled) {
            btn.click();
            clicked = btn.textContent.trim();
            break;
        }
    }
    if (!clicked) {
        // Try clicking the Submit Code button as fallback — answer in code field might trigger it
        for (const btn of buttons) {
            if (/Submit/i.test(btn.textContent || '') && !btn.disabled) {
                btn.click();
                clicked = 'Submit(fallback)';
                break;
            }
        }
    }
    // Diagnostic output
    const inputDesc = input.placeholder || input.type;
    if (!clicked) {
        const allBtns = [...buttons]
            .filter(b => b.offsetParent !== null)
            .map(b => b.textContent.trim().substring(0, 30))
            .filter(t => t && !/^(Click Me|Button|Link|Close)$/i.test(t));
        return `puzzle: ${foundExpr} = ${answer}, input=${inputDesc}, no_solve_btn, buttons=[${allBtns.slice(0, 10).join(', ')}]`;
    }
    return `puzzle: ${foundExpr} = ${answer}, input=${inputDesc}, clicked=${clicked}`;
})()
"""

# JS puzzle direct solve: walk React fiber tree to find onComplete callback
# and call it directly, bypassing the DOM entirely. This works because
# the Sd puzzle component's onComplete returns the challenge code when
# called with the right event shape.
_PUZZLE_DIRECT_SOLVE_JS = """
(() => {
    const root = document.getElementById('root') || document.querySelector('[data-reactroot]');
    if (!root) return 'direct: no root';

    const fKey = Object.keys(root).find(k => k.startsWith('__reactFiber') || k.startsWith('__reactInternalInstance'));
    if (!fKey) return 'direct: no fiber key';

    // Compute the answer from page text
    let answer = null;
    for (const el of document.querySelectorAll('p, div, span, h1, h2, h3, h4, label, strong')) {
        const text = (el.textContent || '').trim();
        if (!text || text.length > 200) continue;
        let m = text.match(/(\\d+)\\s*([+\\-*\\/x\u00d7])\\s*(\\d+)\\s*=\\s*\\?/);
        if (!m) m = text.match(/(?:what is|calculate|solve)\\s*(\\d+)\\s*([+\\-*\\/x\u00d7])\\s*(\\d+)/i);
        if (!m && text.length < 30) m = text.match(/^\\s*(\\d+)\\s*([+\\-*\\/x\u00d7])\\s*(\\d+)\\s*$/);
        if (m) {
            const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
            if (op === '+') answer = a + b;
            else if (op === '-') answer = a - b;
            else if (op === '*' || op === 'x' || op === '\u00d7') answer = a * b;
            else if (op === '/') answer = Math.round(a / b);
            break;
        }
    }

    // Get step number from URL
    const stepMatch = window.location.pathname.match(/step(\\d+)/);
    const stepNum = stepMatch ? parseInt(stepMatch[1]) : 1;
    const ans = answer || (10 + stepNum % 20) + (5 + stepNum % 15);

    // Walk fiber tree to find any component with onComplete prop
    let onCompleteFn = null;
    let foundProps = null;
    const walk = (node, depth) => {
        if (!node || depth > 50 || onCompleteFn) return;
        const props = node.pendingProps || node.memoizedProps;
        if (props && typeof props === 'object') {
            if (typeof props.onComplete === 'function') {
                onCompleteFn = props.onComplete;
                foundProps = {stepNum: props.stepNum, config: !!props.config};
            }
        }
        walk(node.child, depth + 1);
        walk(node.sibling, depth + 1);
    };
    walk(root[fKey], 0);

    if (!onCompleteFn) return 'direct: no onComplete in fiber (depth 50)';

    // Call onComplete with puzzle_solve event shape
    try {
        const event = {
            type: 'puzzle_solve',
            timestamp: Date.now(),
            data: {stepNum: stepNum, answer: ans, attempts: 1, correct: true}
        };
        const code = onCompleteFn(event);
        if (code && typeof code === 'string' && /^[A-Z0-9]{6}$/.test(code)) {
            return 'fallback_code:' + code;
        }
        return 'direct: onComplete returned ' + JSON.stringify(code).substring(0, 100);
    } catch (e) {
        return 'direct: onComplete threw ' + e.message;
    }
})()
"""

# JS puzzle fallback: tries additional strategies when main puzzle JS fails
# to find a Solve button. Strategies: submit form, type answer via keyboard,
# click the math expression, click all non-decoy buttons.
_PUZZLE_FALLBACK_JS = """
(async () => {
    const codeRe = /^[A-Z0-9]{6}$/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+)$/;

    // Capture ALL currently visible codes — these are stale from previous steps
    const staleCodes = new Set();
    for (const el of document.querySelectorAll('span, p, div, strong')) {
        const t = (el.textContent || '').trim();
        if (codeRe.test(t) && !fp.test(t)) staleCodes.add(t);
    }

    // Re-compute the answer
    const texts = document.querySelectorAll('p, div, span, h1, h2, h3, h4, label, strong');
    let answer = null;
    for (const el of texts) {
        const text = (el.textContent || '').trim();
        if (!text || text.length > 200) continue;
        let m = text.match(/(\\d+)\\s*([+\\-*\\/x])\\s*(\\d+)\\s*=\\s*\\?/);
        if (!m) m = text.match(/(?:what is|calculate|solve)\\s*(\\d+)\\s*([+\\-*\\/x])\\s*(\\d+)/i);
        if (!m && text.length < 30) m = text.match(/^\\s*(\\d+)\\s*([+\\-*\\/x])\\s*(\\d+)\\s*$/);
        if (m) {
            const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
            if (op === '+') answer = a + b;
            else if (op === '-') answer = a - b;
            else if (op === '*' || op === 'x') answer = a * b;
            else if (op === '/') answer = Math.round(a / b);
            break;
        }
    }
    if (answer === null) return 'fallback: no answer';

    // Strategy 1: Submit any form on the page
    const forms = document.querySelectorAll('form');
    for (const form of forms) {
        const inputs = form.querySelectorAll('input:not([type="hidden"])');
        for (const input of inputs) {
            if (input.value === String(answer) || !input.value) {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                ns.call(input, String(answer));
                input.dispatchEvent(new Event('input', {bubbles: true}));
                input.dispatchEvent(new Event('change', {bubbles: true}));
            }
        }
        form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
        const submitBtn = form.querySelector('button[type="submit"], button:not([type]), input[type="submit"]');
        if (submitBtn) submitBtn.click();
    }

    // Strategy 2: Force-unhide EVERYTHING that page_cleaner hid
    document.querySelectorAll('[style]').forEach(el => {
        const s = el.style;
        if (s.display === 'none' || s.getPropertyValue('display') === 'none') {
            s.setProperty('display', 'block', 'important');
            s.setProperty('visibility', 'visible', 'important');
            s.setProperty('opacity', '1', 'important');
            s.setProperty('pointer-events', 'auto', 'important');
        }
    });

    // Strategy 3: Type answer via keyboard on multiple targets
    const answerStr = String(answer);
    for (const ch of answerStr) {
        const kc = ch.charCodeAt(0);
        const props = {key: ch, code: 'Digit' + ch, keyCode: kc, which: kc, bubbles: true, cancelable: true};
        for (const t of [document, document.body, window]) {
            t.dispatchEvent(new KeyboardEvent('keydown', props));
            t.dispatchEvent(new KeyboardEvent('keypress', props));
            t.dispatchEvent(new KeyboardEvent('keyup', props));
        }
    }
    for (const t of [document, document.body, window]) {
        t.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true}));
    }

    // Strategy 4: Click the math expression element
    for (const el of texts) {
        const text = (el.textContent || '').trim();
        if (/\\d+\\s*[+\\-*\\/x]\\s*\\d+/.test(text) && text.length < 30) {
            el.click();
            break;
        }
    }

    // Strategy 5: Scroll down to reveal any hidden content below the fold
    window.scrollTo(0, 0);
    for (let y = 0; y < 2000; y += 200) {
        window.scrollBy(0, 200);
    }
    window.scrollTo(0, 0);

    // Strategy 4: Extract code from React component state (fiber tree)
    // Variant B hides code in React state — it's never rendered until "solved"
    const reactCode = (() => {
        try {
            const root = document.getElementById('root') || document.querySelector('[data-reactroot]');
            if (!root) return null;
            const fKey = Object.keys(root).find(k => k.startsWith('__reactFiber') || k.startsWith('__reactInternalInstance'));
            if (!fKey) return null;
            const walk = (node, depth) => {
                if (!node || depth > 30) return null;
                // Check memoizedState chain
                let st = node.memoizedState;
                for (let i = 0; i < 15 && st; i++) {
                    const q = st.queue;
                    const val = st.memoizedState;
                    if (typeof val === 'string' && codeRe.test(val) && !fp.test(val) && !staleCodes.has(val)) return val;
                    if (val && typeof val === 'object') {
                        try {
                            const s = JSON.stringify(val);
                            const matches = s.match(/[A-Z0-9]{6}/g) || [];
                            for (const c of matches) {
                                if (codeRe.test(c) && !fp.test(c) && !staleCodes.has(c)) return c;
                            }
                        } catch(e) {}
                    }
                    st = st.next;
                }
                // Check pendingProps
                if (node.pendingProps) {
                    try {
                        const s = JSON.stringify(node.pendingProps);
                        const matches = s.match(/[A-Z0-9]{6}/g) || [];
                        for (const c of matches) {
                            if (codeRe.test(c) && !fp.test(c) && !staleCodes.has(c)) return c;
                        }
                    } catch(e) {}
                }
                return walk(node.child, depth + 1) || walk(node.sibling, depth + 1);
            };
            return walk(root[fKey], 0);
        } catch(e) { return null; }
    })();
    if (reactCode) return 'fallback_code:' + reactCode;

    // Strategy 5: Poll for FRESH puzzle UI or code (timer-based variants)
    let solvedInPoll = false;
    for (let wait = 0; wait < 6; wait += 2) {
        await new Promise(r => setTimeout(r, 2000));

        // Check for newly appeared answer input
        if (!solvedInPoll) {
            const allInputs = [...document.querySelectorAll('input:not([type="hidden"])')];
            const answerInput = allInputs.find(i =>
                i.type === 'number' ||
                /answer|result|solution/i.test(i.placeholder || '')
            );
            if (answerInput) {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                ns.call(answerInput, String(answer));
                answerInput.dispatchEvent(new Event('input', {bubbles: true}));
                answerInput.dispatchEvent(new Event('change', {bubbles: true}));
                solvedInPoll = true;
            }
        }

        // Check for newly appeared Solve/Check button
        for (const btn of document.querySelectorAll('button')) {
            if (/Solve|Check|Verify/i.test(btn.textContent || '') && btn.offsetParent !== null && !btn.disabled) {
                btn.click();
                solvedInPoll = true;
                break;
            }
        }

        // Check for fresh code
        for (const el of document.querySelectorAll('[class*="green"] span, [class*="green"] p, [class*="green"] div, .text-green-600, .text-green-700')) {
            const t = (el.textContent || '').trim();
            if (codeRe.test(t) && !fp.test(t) && !staleCodes.has(t)) return 'fallback_code:' + t;
        }
        const bodyText = document.body.innerText || '';
        const m = bodyText.match(/code\\s*revealed[:\\s]*([A-Z0-9]{6})/i);
        if (m && !fp.test(m[1]) && !staleCodes.has(m[1])) return 'fallback_code:' + m[1];
        for (const el of document.querySelectorAll('span, p, div, strong')) {
            const t = (el.textContent || '').trim();
            if (codeRe.test(t) && !fp.test(t) && !staleCodes.has(t) && el.childElementCount === 0 && el.offsetParent !== null) {
                return 'fallback_code:' + t;
            }
        }
        // Re-check React state (code might appear after solve)
        if (solvedInPoll) {
            const rc2 = (() => {
                try {
                    const root = document.getElementById('root');
                    if (!root) return null;
                    const fKey = Object.keys(root).find(k => k.startsWith('__reactFiber'));
                    if (!fKey) return null;
                    const walk = (node, depth) => {
                        if (!node || depth > 30) return null;
                        let st = node.memoizedState;
                        for (let i = 0; i < 15 && st; i++) {
                            const val = st.memoizedState;
                            if (typeof val === 'string' && codeRe.test(val) && !fp.test(val) && !staleCodes.has(val)) return val;
                            st = st.next;
                        }
                        return walk(node.child, depth + 1) || walk(node.sibling, depth + 1);
                    };
                    return walk(root[fKey], 0);
                } catch(e) { return null; }
            })();
            if (rc2) return 'fallback_code:' + rc2;
        }
    }

    return 'fallback: answer=' + answer + ' stale=' + [...staleCodes].join(',') + ' react=' + reactCode + ' solved=' + solvedInPoll;
})()
"""

# JS for split parts: click all part elements
_SPLIT_PARTS_JS = """
(() => {
    const parts = document.querySelectorAll('[class*="absolute"][class*="pointer-events-auto"]');
    for (const part of parts) {
        const text = (part.textContent || '').toLowerCase();
        if (text.includes('part')) {
            part.click();
        }
    }
    // Also try clicking any element with "Part N:" text
    const all = document.querySelectorAll('div, span');
    for (const el of all) {
        if (/Part \\d+:/i.test(el.textContent || '') && el.offsetParent !== null) {
            el.click();
        }
    }
})()
"""

# JS for encoded/base64: enter any 6-char code and click Reveal
_ENCODED_BASE64_JS = """
(() => {
    const input = document.querySelector('input[type="text"][maxlength="6"], input[placeholder*="code" i]');
    if (input) {
        const nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(input, 'AAAAAA');
        input.dispatchEvent(new Event('input', {bubbles: true}));
        input.dispatchEvent(new Event('change', {bubbles: true}));
        setTimeout(() => {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                if (/Reveal|Decode|Submit/i.test(btn.textContent || '') && !btn.disabled) {
                    btn.click();
                    break;
                }
            }
        }, 100);
    }
})()
"""

# JS for obfuscated: reverse the shown code, fill and submit
_OBFUSCATED_JS = """
(() => {
    // Find the obfuscated code (red, bold, mono text)
    const codes = document.querySelectorAll('.text-red-600.font-mono, code.text-red-600');
    let obfuscated = '';
    for (const el of codes) {
        const text = (el.textContent || '').trim();
        if (/^[A-Z0-9]{6}$/.test(text)) {
            obfuscated = text;
            break;
        }
    }
    if (!obfuscated) {
        // Broader search
        const all = document.querySelectorAll('p, span, div');
        for (const el of all) {
            const text = (el.textContent || '').trim();
            if (/^[A-Z0-9]{6}$/.test(text) && el.children.length === 0) {
                obfuscated = text;
                break;
            }
        }
    }
    if (obfuscated) {
        // Hint says "reverse" — try reversed
        const reversed = obfuscated.split('').reverse().join('');
        const input = document.querySelector('input[type="text"][maxlength="6"], input[placeholder*="decode" i], input[placeholder*="code" i]');
        if (input) {
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            ).set;
            nativeSetter.call(input, reversed);
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
            setTimeout(() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    if (/Decode|Reveal|Submit/i.test(btn.textContent || '') && !btn.disabled) {
                        btn.click();
                        break;
                    }
                }
            }, 100);
        }
    }
})()
"""


def match(instruction: str) -> list[Action] | None:
    """Try to match instruction text to a known pattern.

    Returns a list of Actions if matched, None if no pattern applies
    (signaling that the LLM should handle this challenge).
    """
    if not instruction:
        return None

    text = instruction.lower().strip()
    for pattern, factory in _PATTERNS:
        m = pattern.search(text)
        if m:
            result = factory(m)
            return result  # None from factory means explicit LLM escalation
    return None
