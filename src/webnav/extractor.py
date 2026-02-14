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
            const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;
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

    # Fallback: scan Shadow DOM roots for codes (web components hide content)
    shadow_code = await page.evaluate(_SHADOW_DOM_SCAN_JS)
    if shadow_code and shadow_code not in skip:
        return shadow_code

    # Fallback: search all nested iframes for codes via Playwright frame API
    for frame in page.frames[1:]:  # Skip main frame (already searched)
        try:
            frame_text = await frame.evaluate("() => document.body?.innerText || ''")
            frame_candidates = [c for c in _extract_candidates(frame_text) if c not in skip]
            if frame_candidates:
                return frame_candidates[0]
        except Exception:
            continue

    # Fallback: recursive JS search through iframe contentDocuments
    # (catches same-origin iframes that Playwright may not enumerate)
    iframe_code = await page.evaluate(_IFRAME_RECURSIVE_SCAN_JS)
    if iframe_code and iframe_code not in skip:
        return iframe_code

    return None


# Single-pass JS scan for hidden codes in attributes, hidden elements, comments.
# Avoids expensive operations like getComputedStyle for pseudo-elements.
_HIDDEN_CODE_SCAN_JS = """
(() => {
    const exactRe = /^[A-Z0-9]{6}$/;
    const partialRe = /\\b([A-Z0-9]{6})\\b/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;

    // Pass 1a: Check all element attributes — exact match (skip noise)
    for (const el of document.querySelectorAll('*')) {
        // No noise skip — extractor should check ALL elements for codes
        for (const attr of el.attributes) {
            if (exactRe.test(attr.value) && !fp.test(attr.value)) return attr.value;
        }
    }

    // Pass 1b: Check data-* and aria-* attributes — partial match (skip noise)
    for (const el of document.querySelectorAll('*')) {
        // No noise skip — extractor should check ALL elements for codes
        for (const attr of el.attributes) {
            if (attr.name === 'class' || attr.name === 'style' || attr.name === 'src') continue;
            const m = attr.value.match(partialRe);
            if (m && !fp.test(m[1])) return m[1];
        }
    }

    // Pass 2: Check hidden leaf text (exact match) — skip noise elements
    for (const el of document.querySelectorAll('*')) {
        // No noise skip — extractor should check ALL elements for codes
        const text = (el.textContent || '').trim();
        if (exactRe.test(text) && !fp.test(text) && el.childElementCount === 0) {
            return text;
        }
    }

    // Pass 3: Search within text of actually-hidden elements (partial match) — skip noise
    for (const el of document.querySelectorAll('*')) {
        // No noise skip — extractor should check ALL elements for codes
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


# Recursively search Shadow DOM roots for 6-char codes.
# document.querySelectorAll('*') doesn't cross shadow boundaries;
# we must explicitly traverse each element's shadowRoot.
_SHADOW_DOM_SCAN_JS = """
(() => {
    const codeRe = /\\b([A-Z0-9]{6})\\b/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;
    function searchShadow(root, depth) {
        if (!root || depth > 10) return null;
        for (const el of root.querySelectorAll('*')) {
            // Support both open (el.shadowRoot) and closed (__shadow) roots
            const sr = el.shadowRoot || el.__shadow;
            if (sr) {
                const text = (sr.textContent || sr.innerHTML || '');
                const m = text.match(codeRe);
                if (m && !fp.test(m[1])) return m[1];
                // Check attributes inside shadow root
                for (const child of sr.querySelectorAll('*')) {
                    for (const attr of child.attributes) {
                        if (/^[A-Z0-9]{6}$/.test(attr.value) && !fp.test(attr.value))
                            return attr.value;
                    }
                }
                const deeper = searchShadow(sr, depth + 1);
                if (deeper) return deeper;
            }
        }
        return null;
    }
    return searchShadow(document, 0);
})()
"""


# Recursively search same-origin iframes for 6-char codes.
# Uses contentDocument to cross iframe boundaries from JS.
_IFRAME_RECURSIVE_SCAN_JS = """
(() => {
    const codeRe = /\\b([A-Z0-9]{6})\\b/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;
    function search(doc, depth) {
        if (!doc || depth > 10) return null;
        const text = (doc.body?.innerText || '');
        const m = text.match(codeRe);
        if (m && !fp.test(m[1])) return m[1];
        // Check attributes in this document
        for (const el of doc.querySelectorAll('*')) {
            for (const attr of el.attributes) {
                if (/^[A-Z0-9]{6}$/.test(attr.value) && !fp.test(attr.value)) return attr.value;
            }
        }
        // Recurse into iframes
        for (const iframe of doc.querySelectorAll('iframe')) {
            try {
                const result = search(iframe.contentDocument, depth + 1);
                if (result) return result;
            } catch(e) {}
        }
        return null;
    }
    return search(document, 0);
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
