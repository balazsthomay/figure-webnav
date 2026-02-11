"""Code extraction from page content."""

from __future__ import annotations

import re

from playwright.async_api import Page

from webnav.perception import CODE_PATTERN, _FALSE_POSITIVE


async def find_code(page: Page, used_codes: set[str] | None = None) -> str | None:
    """Scan the page for a valid 6-char challenge code.

    Checks inner text, aria snapshot, and hidden DOM elements.
    Returns the first valid code found, or None.
    Skips codes in `used_codes` (previously submitted on earlier steps).
    """
    skip = used_codes or set()

    # Priority 0: check for "confirmed" codes in green success boxes
    # and "Code revealed:" patterns (puzzle challenge success messages).
    # This runs BEFORE page_cleaner can hide high z-index success elements.
    green_code = await page.evaluate("""
        () => {
            const codeRe = /^[A-Z0-9]{6}$/;
            const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+)$/;
            // Strategy 1: Look for code inside green success containers
            const greens = document.querySelectorAll(
                '.text-green-600, .text-green-700, [class*="bg-green"] span, [class*="bg-green"] p, [class*="bg-green"] div'
            );
            for (const el of greens) {
                const text = (el.textContent || '').trim();
                if (codeRe.test(text) && !fp.test(text)) return text;
            }
            // Strategy 2: Look for "Code revealed:" pattern in visible text
            const allText = document.body.innerText || '';
            const revealMatch = allText.match(/(?:code\\s*revealed|your\\s*code)[:\\s]*([A-Z0-9]{6})/);
            if (revealMatch && !fp.test(revealMatch[1])) return revealMatch[1];
            // Strategy 3: Find ANY 6-char code in green/success-styled containers
            for (const el of document.querySelectorAll('[class*="green"], [class*="success"], [style*="green"]')) {
                const m = (el.textContent || '').match(/([A-Z0-9]{6})/);
                if (m && !fp.test(m[1])) return m[1];
            }
            return null;
        }
    """)
    if green_code and green_code not in skip:
        return green_code

    # Primary: scan visible text (fast — single DOM read)
    try:
        text = await page.inner_text("body")
    except Exception:
        text = ""

    candidates = [c for c in _extract_candidates(text) if c not in skip]
    if candidates:
        return candidates[0]

    # Fallback: single JS scan for hidden codes (attributes, hidden elements, comments)
    # Combines all checks into one evaluate() call to avoid multiple DOM traversals.
    hidden_code = await page.evaluate(_HIDDEN_CODE_SCAN_JS)
    if hidden_code and hidden_code not in skip:
        return hidden_code

    return None


# Single-pass JS scan for hidden codes in attributes, hidden elements, comments.
# Avoids expensive operations like getComputedStyle for pseudo-elements.
_HIDDEN_CODE_SCAN_JS = """
(() => {
    const exactRe = /^[A-Z0-9]{6}$/;
    const partialRe = /\\b([A-Z0-9]{6})\\b/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+)$/;

    // Pass 1a: Check all element attributes — exact match
    for (const el of document.querySelectorAll('*')) {
        for (const attr of el.attributes) {
            if (exactRe.test(attr.value) && !fp.test(attr.value)) return attr.value;
        }
    }

    // Pass 1b: Check data-* and aria-* attributes — partial match
    for (const el of document.querySelectorAll('*')) {
        for (const attr of el.attributes) {
            if (attr.name === 'class' || attr.name === 'style' || attr.name === 'src') continue;
            const m = attr.value.match(partialRe);
            if (m && !fp.test(m[1])) return m[1];
        }
    }

    // Pass 2: Check hidden leaf text (exact match)
    for (const el of document.querySelectorAll('*')) {
        const text = (el.textContent || '').trim();
        if (exactRe.test(text) && !fp.test(text) && el.childElementCount === 0) {
            return text;
        }
    }

    // Pass 3: Search within text of actually-hidden elements (partial match)
    for (const el of document.querySelectorAll('*')) {
        const style = window.getComputedStyle(el);
        const hidden = style.display === 'none' || style.visibility === 'hidden' ||
            style.opacity === '0' || style.color === 'rgba(0, 0, 0, 0)' ||
            style.color === 'transparent' ||
            (el.offsetWidth === 0 && el.offsetHeight === 0);
        if (!hidden) continue;
        const text = el.textContent || '';
        const m = text.match(partialRe);
        if (m && !fp.test(m[1])) return m[1];
    }

    // Pass 4: Check pseudo-element content
    for (const el of document.querySelectorAll('*')) {
        for (const pseudo of ['::before', '::after']) {
            try {
                const content = window.getComputedStyle(el, pseudo).content;
                if (content && content !== 'none' && content !== 'normal') {
                    const clean = content.replace(/['"]/g, '');
                    const m = clean.match(partialRe);
                    if (m && !fp.test(m[1])) return m[1];
                }
            } catch(e) {}
        }
    }

    // Pass 5: HTML comments
    const walker = document.createTreeWalker(
        document.body, NodeFilter.SHOW_COMMENT, null, false
    );
    let comment;
    while (comment = walker.nextNode()) {
        const m = (comment.textContent || '').match(partialRe);
        if (m && !fp.test(m[1])) return m[1];
    }

    return null;
})()
"""


def _extract_candidates(text: str) -> list[str]:
    """Extract valid code candidates from text."""
    candidates: list[str] = []
    for match in CODE_PATTERN.finditer(text):
        code = match.group(1)
        if not _FALSE_POSITIVE.match(code):
            candidates.append(code)
    return candidates


def _flatten_tree_text(node: dict) -> str:
    """Recursively extract all text from a11y tree dict."""
    parts = []
    name = node.get("name", "")
    if name:
        parts.append(name)
    for child in node.get("children", []):
        parts.append(_flatten_tree_text(child))
    return " ".join(parts)
