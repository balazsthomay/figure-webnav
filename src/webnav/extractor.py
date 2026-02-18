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
    # Also re-hides noise elements that React re-renders may have restored
    # (clean_page sets display:none !important, but React style reconciliation
    # can override inline styles after any click triggers a re-render).
    green_code = await page.evaluate("""
        () => {
            // Re-hide noise elements first — React re-renders can undo display:none
            document.querySelectorAll('[data-wnav-noise]').forEach(el =>
                el.style.setProperty('display', 'none', 'important'));
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

    # Fallback: decode Base64-encoded strings that may contain a 6-char code
    b64_code = await page.evaluate(_BASE64_DECODE_SCAN_JS)
    if b64_code and b64_code not in skip:
        return b64_code

    # Fallback: scan inline <script> tags for codes in JS data/variables
    script_code = await page.evaluate(_SCRIPT_TAG_SCAN_JS)
    if script_code and script_code not in skip:
        return script_code

    # Fallback: split content reassembly + multi-encoding + template/noscript/SVG
    advanced_code = await page.evaluate(_ADVANCED_SCAN_JS)
    if advanced_code and advanced_code not in skip:
        return advanced_code

    return None


# Scan text content and data-* attributes for Base64-encoded 6-char codes.
# Tries atob() on Base64-like strings, checks if decoded result matches ^[A-Z0-9]{6}$.
_BASE64_DECODE_SCAN_JS = """
(() => {
    const codeRe = /^[A-Z0-9]{6}$/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;
    const b64Re = /(?:^|[\\s:='"(])([A-Za-z0-9+/]{8,}={0,2})(?:[\\s:='")\\.;,]|$)/g;
    function tryDecode(str) {
        try {
            const decoded = atob(str);
            const m = decoded.match(/([A-Z0-9]{6})/);
            if (m && !fp.test(m[1])) return m[1];
        } catch(e) {}
        return null;
    }
    // Check all text content and data-* attributes
    for (const el of document.querySelectorAll('*')) {
        // Check data-* attributes
        for (const attr of el.attributes) {
            if (!attr.name.startsWith('data-') && attr.name !== 'title'
                && attr.name !== 'alt' && attr.name !== 'value') continue;
            let m;
            b64Re.lastIndex = 0;
            while ((m = b64Re.exec(attr.value)) !== null) {
                const code = tryDecode(m[1]);
                if (code) return code;
            }
            // Also try the whole attribute value as Base64
            const code = tryDecode(attr.value.trim());
            if (code) return code;
        }
        // Check leaf text content
        if (el.childElementCount === 0) {
            const text = (el.textContent || '').trim();
            if (text.length >= 8 && text.length <= 100) {
                const code = tryDecode(text);
                if (code) return code;
            }
            // Check for embedded Base64 in text
            let m;
            b64Re.lastIndex = 0;
            while ((m = b64Re.exec(text)) !== null) {
                const code = tryDecode(m[1]);
                if (code) return code;
            }
        }
    }
    // Check sessionStorage and localStorage for Base64-encoded codes
    for (const storage of [sessionStorage, localStorage]) {
        for (let i = 0; i < storage.length; i++) {
            const val = storage.getItem(storage.key(i)) || '';
            const code = tryDecode(val.trim());
            if (code) return code;
        }
    }
    return null;
})()
"""


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

    // Pass 6: Raw localStorage/sessionStorage (non-Base64)
    for (const storage of [sessionStorage, localStorage]) {
        try {
            for (let i = 0; i < storage.length; i++) {
                const val = storage.getItem(storage.key(i)) || '';
                const m = val.match(partialRe);
                if (m && !fp.test(m[1])) return m[1];
            }
        } catch(e) {}
    }

    // Pass 7: CSS custom properties on challenge containers
    for (const el of document.querySelectorAll('[class*="challenge"], [class*="step"], main, section')) {
        try {
            const style = window.getComputedStyle(el);
            for (const prop of ['--code', '--secret', '--hidden', '--value', '--answer']) {
                const val = style.getPropertyValue(prop).trim();
                if (val) {
                    const m = val.replace(/['"]/g, '').match(partialRe);
                    if (m && !fp.test(m[1])) return m[1];
                }
            }
        } catch(e) {}
    }

    // Pass 8: Window/document properties
    for (const prop of ['code', 'secretCode', 'hiddenCode', 'challengeCode', 'answer']) {
        try {
            const val = String(window[prop] || '');
            const m = val.match(partialRe);
            if (m && !fp.test(m[1])) return m[1];
        } catch(e) {}
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


# Scan inline <script> tags for 6-char codes in JS data, variable assignments,
# JSON objects, or string literals. Catches codes hidden in JS code.
_SCRIPT_TAG_SCAN_JS = """
(() => {
    const codeRe = /\\b([A-Z0-9]{6})\\b/g;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION|WINDOW|DOCUME|FUNCTI|RETURN|OBJECT|SCRIPT|TYPEOF|NUMBER|LENGTH|FILTER|CONCAT|IMPORT|EXPORT|MODULE|REQUIR)$/;
    for (const script of document.querySelectorAll('script:not([src])')) {
        const text = script.textContent || '';
        if (text.length < 6 || text.length > 50000) continue;
        // Look for codes in string assignments: var code = "ABC123"
        // or JSON: {"code": "ABC123"} or array: ["ABC123"]
        const stringLiterals = text.match(/["']([A-Z0-9]{6})["']/g);
        if (stringLiterals) {
            for (const lit of stringLiterals) {
                const code = lit.slice(1, -1);
                if (!fp.test(code)) return code;
            }
        }
        // Broader scan: any 6-char match in script content
        let m;
        codeRe.lastIndex = 0;
        while ((m = codeRe.exec(text)) !== null) {
            if (!fp.test(m[1])) return m[1];
        }
    }
    return null;
})()
"""


# Advanced scan: split content reassembly, multi-encoding, template/noscript/SVG,
# CSS custom properties from stylesheets, script concatenation patterns.
# Runs as a single evaluate() call to minimize round-trips.
_ADVANCED_SCAN_JS = """
(() => {
    const codeRe = /^[A-Z0-9]{6}$/;
    const partialRe = /([A-Z0-9]{6})/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION|WINDOW|DOCUME|FUNCTI|RETURN|OBJECT|SCRIPT|TYPEOF|NUMBER|LENGTH|FILTER|CONCAT|IMPORT|EXPORT|MODULE|REQUIR)$/;

    function ok(c) { return codeRe.test(c) && !fp.test(c); }
    function okP(c) { return c && !fp.test(c); }

    // --- 1. Split content: codes spread across adjacent inline siblings ---
    const inlineTags = new Set([
        'SPAN','A','B','I','EM','STRONG','CODE','SMALL','SUB','SUP',
        'ABBR','MARK','S','U','VAR','KBD','SAMP','Q','CITE','DFN'
    ]);
    for (const parent of document.querySelectorAll('*')) {
        if (parent.children.length < 2 || parent.children.length > 20) continue;
        let combined = '';
        let hasInline = false;
        for (const child of parent.children) {
            if (inlineTags.has(child.tagName)) {
                combined += (child.textContent || '').trim();
                hasInline = true;
            }
        }
        if (hasInline && ok(combined)) return combined;
    }

    // --- 2. Split data attributes: data-part1 + data-part2 etc. ---
    for (const el of document.querySelectorAll('*')) {
        const parts = [];
        for (let i = 1; i <= 10; i++) {
            const v = el.getAttribute('data-part' + i)
                   || el.getAttribute('data-part-' + i)
                   || el.getAttribute('data-code-' + i);
            if (v) parts.push(v); else break;
        }
        if (parts.length > 1 && ok(parts.join(''))) return parts.join('');
        for (const [a, b] of [['first','second'],['start','end'],['left','right'],['prefix','suffix']]) {
            const va = el.dataset?.[a] || '';
            const vb = el.dataset?.[b] || '';
            if (va && vb && ok(va + vb)) return va + vb;
        }
    }

    // --- 3. Pseudo-element combined: ::before + text + ::after ---
    for (const el of document.querySelectorAll('*')) {
        try {
            const before = (getComputedStyle(el, '::before').content || '').replace(/^["']|["']$/g, '');
            const text = (el.childElementCount === 0 ? (el.textContent || '').trim() : '');
            const after = (getComputedStyle(el, '::after').content || '').replace(/^["']|["']$/g, '');
            const bC = (before === 'none' || before === 'normal') ? '' : before;
            const aC = (after === 'none' || after === 'normal') ? '' : after;
            if ((bC || aC) && text) {
                const combined = bC + text + aC;
                if (ok(combined)) return combined;
            }
        } catch(e) {}
    }

    // --- 4. Template, noscript, SVG text elements ---
    for (const tmpl of document.querySelectorAll('template')) {
        const c = tmpl.content || tmpl;
        const t = (c.textContent || '');
        const m = t.match(partialRe);
        if (m && okP(m[1])) return m[1];
    }
    for (const ns of document.querySelectorAll('noscript')) {
        const t = ns.textContent || ns.innerHTML || '';
        const m = t.match(partialRe);
        if (m && okP(m[1])) return m[1];
    }
    for (const svg of document.querySelectorAll('svg')) {
        for (const textEl of svg.querySelectorAll('text, tspan')) {
            const t = (textEl.textContent || '').trim();
            if (ok(t)) return t;
        }
    }

    // --- 5. Multi-encoding: ROT13, hex, charCode, reverse, URL-decode ---
    function tryDecode(str) {
        if (!str || str.length < 6 || str.length > 200) return null;
        // ROT13
        const rot13 = str.replace(/[A-Za-z]/g, c => {
            const base = c <= 'Z' ? 65 : 97;
            return String.fromCharCode(((c.charCodeAt(0) - base + 13) % 26) + base);
        });
        let m = rot13.match(partialRe);
        if (m && okP(m[1])) return m[1];
        // Hex pairs: "48656C6C6F" or "48 65 6C 6C 6F"
        if (/^(?:[0-9A-Fa-f]{2}\\s*){3,}$/.test(str.trim())) {
            try {
                const dec = str.trim().replace(/\\s/g, '')
                    .match(/.{2}/g).map(h => String.fromCharCode(parseInt(h, 16))).join('');
                m = dec.match(partialRe);
                if (m && okP(m[1])) return m[1];
            } catch(e) {}
        }
        // Comma-separated char codes: "65,66,67,49,50,51"
        if (/^[\\d,\\s]+$/.test(str.trim()) && str.includes(',')) {
            try {
                const dec = String.fromCharCode(...str.split(',').map(n => parseInt(n.trim())));
                m = dec.match(partialRe);
                if (m && okP(m[1])) return m[1];
            } catch(e) {}
        }
        // Reversed string
        const rev = str.split('').reverse().join('');
        m = rev.match(partialRe);
        if (m && okP(m[1])) return m[1];
        // Double Base64
        try {
            const first = atob(str);
            try {
                const second = atob(first);
                m = second.match(partialRe);
                if (m && okP(m[1])) return m[1];
            } catch(e) {}
        } catch(e) {}
        return null;
    }
    // Apply multi-decode to leaf text and data-* attributes
    for (const el of document.querySelectorAll('*')) {
        if (el.childElementCount === 0) {
            const t = (el.textContent || '').trim();
            if (t.length >= 6 && t.length <= 200) {
                const r = tryDecode(t);
                if (r) return r;
            }
        }
        for (const attr of el.attributes) {
            if (!attr.name.startsWith('data-')) continue;
            const r = tryDecode(attr.value.trim());
            if (r) return r;
        }
    }

    // --- 6. CSS custom properties from all stylesheets ---
    try {
        for (const sheet of document.styleSheets) {
            try {
                for (const rule of sheet.cssRules || []) {
                    if (rule.type !== 1) continue;
                    for (const prop of rule.style) {
                        if (!prop.startsWith('--')) continue;
                        const val = rule.style.getPropertyValue(prop).trim().replace(/['"]/g, '');
                        const m = val.match(partialRe);
                        if (m && okP(m[1])) return m[1];
                        // Try Base64 on CSS var value
                        try {
                            const dec = atob(val);
                            const m2 = dec.match(partialRe);
                            if (m2 && okP(m2[1])) return m2[1];
                        } catch(e) {}
                    }
                }
            } catch(e) {}
        }
    } catch(e) {}

    // --- 7. Script tag concatenation patterns ---
    for (const script of document.querySelectorAll('script:not([src])')) {
        const text = script.textContent || '';
        if (text.length < 6 || text.length > 50000) continue;
        // "AB" + "C1" + "23"
        const concatRe = /["']([A-Z0-9]{1,5})["']\\s*\\+\\s*["']([A-Z0-9]{1,5})["'](?:\\s*\\+\\s*["']([A-Z0-9]{1,5})["'])?(?:\\s*\\+\\s*["']([A-Z0-9]{1,5})["'])?/g;
        let m2;
        while ((m2 = concatRe.exec(text)) !== null) {
            const c = (m2[1]||'') + (m2[2]||'') + (m2[3]||'') + (m2[4]||'');
            if (ok(c)) return c;
        }
        // String.fromCharCode(65,66,67,49,50,51)
        const fccRe = /String\\.fromCharCode\\(([^)]+)\\)/g;
        while ((m2 = fccRe.exec(text)) !== null) {
            try {
                const dec = String.fromCharCode(...m2[1].split(',').map(n => parseInt(n.trim())));
                if (ok(dec)) return dec;
            } catch(e) {}
        }
        // ["A","B","C"].join("")
        const ajRe = /\\[([^\\]]*)\\]\\.join\\s*\\(\\s*["']['"]\\s*\\)/g;
        while ((m2 = ajRe.exec(text)) !== null) {
            try {
                const items = m2[1].match(/["']([^"']+)["']/g);
                if (items) {
                    const c = items.map(s => s.slice(1, -1)).join('');
                    if (ok(c)) return c;
                }
            } catch(e) {}
        }
        // "321CBA".split("").reverse().join("")
        const revRe = /["']([A-Z0-9]{6})["']\\.split\\s*\\([^)]*\\)\\.reverse\\s*\\(\\s*\\)\\.join/g;
        while ((m2 = revRe.exec(text)) !== null) {
            const rev = m2[1].split('').reverse().join('');
            if (ok(rev)) return rev;
        }
    }

    // --- 8. Window globals (broader than hardcoded names) ---
    try {
        const builtins = new Set(Object.getOwnPropertyNames(window.__proto__));
        for (const key of Object.getOwnPropertyNames(window)) {
            if (builtins.has(key)) continue;
            try {
                const val = window[key];
                if (typeof val === 'string' && ok(val)) return val;
                if (val && typeof val === 'object' && !Array.isArray(val)) {
                    for (const k of Object.keys(val)) {
                        if (typeof val[k] === 'string' && ok(val[k])) return val[k];
                    }
                }
            } catch(e) {}
        }
    } catch(e) {}

    // --- 9. Meta tags, hidden inputs ---
    for (const meta of document.querySelectorAll('meta')) {
        for (const attr of ['content', 'name', 'value']) {
            const val = meta.getAttribute(attr) || '';
            const m = val.match(partialRe);
            if (m && okP(m[1])) return m[1];
        }
    }
    for (const inp of document.querySelectorAll('input[type="hidden"]')) {
        const val = inp.value || '';
        const m = val.match(partialRe);
        if (m && okP(m[1])) return m[1];
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
