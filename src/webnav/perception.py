"""A11y tree extraction, element indexing, and compression into a compact page state."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Page


# Noise patterns to strip from a11y text
_NOISE_PATTERNS = re.compile(
    r"(lorem ipsum|content block loaded|click me!|button!|link!|complete the challenges|enter the code to proceed|filler content|keep scrolling|browser navigation challenge|scroll down to find)",
    re.IGNORECASE,
)

# Labels look like "Click to Reveal:" or "Hidden DOM Challenge:" — short, end with ":"
_LABEL_RE = re.compile(r"^[\w\s]+:\s*$")

# Matches challenge codes: exactly 6 uppercase alphanumeric chars
CODE_PATTERN = re.compile(r"\b([A-Z0-9]{6})\b")

# False-positive code patterns (common element IDs, CSS-like tokens)
_FALSE_POSITIVE = re.compile(r"^(SUBMIT|SCROLL|CLICK[S]?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\d+|HELLOA|CANVAS|MOVING|COMPLE|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$")


@dataclass
class ElementInfo:
    """An interactive element on the page, indexed for LLM reference."""

    index: int
    tag: str
    role: str = ""
    name: str = ""
    type: str = ""  # input type attribute
    placeholder: str = ""
    selector: str = ""  # unique CSS selector for Playwright
    visible: bool = True
    bbox: dict[str, float] | None = None
    extra: str = ""  # e.g. "scrollable", "clickable"


@dataclass
class PageState:
    """Compressed representation of a challenge page."""

    step: int = 0
    instruction: str = ""
    progress: str = ""
    interactive: list[dict[str, str]] = field(default_factory=list)
    visible_codes: list[str] = field(default_factory=list)
    raw_text: str = ""
    aria_yaml: str = ""
    elements: list[ElementInfo] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Render as compact text for LLM consumption with indexed elements."""
        parts = [
            f"STEP: {self.step}/30",
            f'INSTRUCTION: "{self.instruction}"',
        ]
        if self.progress:
            parts.append(f'PROGRESS: "{self.progress}"')
        if self.visible_codes:
            parts.append(f"VISIBLE_CODES: {self.visible_codes}")

        # Render indexed elements
        if self.elements:
            elem_lines = []
            for el in self.elements:
                desc = f"[{el.index}] {el.tag}"
                if el.role:
                    desc += f" role={el.role}"
                if el.name:
                    desc += f' "{el.name}"'
                if el.type:
                    desc += f" type={el.type}"
                if el.placeholder:
                    desc += f' placeholder="{el.placeholder}"'
                if el.extra:
                    desc += f" [{el.extra}]"
                elem_lines.append(desc)
            parts.append("ELEMENTS:\n" + "\n".join(elem_lines))

        # Include page text context (filtered)
        if self.raw_text:
            lines = []
            for line in self.raw_text.splitlines():
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                if _NOISE_PATTERNS.search(line):
                    continue
                lines.append(line)
            text_block = "\n".join(lines[:30])
            if text_block:
                parts.append(f"PAGE_TEXT:\n{text_block}")

        if self.aria_yaml:
            truncated = "\n".join(self.aria_yaml.splitlines()[:50])
            parts.append(f"ARIA_TREE:\n{truncated}")
        return "\n".join(parts)


# ---------- Element indexing ----------

# JS that enumerates all interactive elements and returns their metadata.
# Returns a JSON-serializable array of element info dicts.
_INDEX_ELEMENTS_JS = """
(() => {
    // Clear stale data-wnav attributes from previous indexing runs
    document.querySelectorAll('[data-wnav]').forEach(el => el.removeAttribute('data-wnav'));

    // Temporarily unhide clean_page-hidden containers so elements inside
    // them are in the layout with valid bounding boxes for indexing.
    const __noiseEls = document.querySelectorAll('[data-wnav-noise]');
    __noiseEls.forEach(el => el.style.removeProperty('display'));

    const selector = 'a, button, input, select, textarea, canvas, ' +
        '[role="button"], [draggable="true"], [onclick], [tabindex], ' +
        '[onmouseenter], [onmouseover], [style*="cursor"]';
    const selectorEls = new Set(document.querySelectorAll(selector));

    // Find elements with React event handlers (onClick, onMouseEnter, onPointerEnter, onScroll, etc.)
    const reactTargets = 'div, span, section, p, li, article, main, aside, header, footer, figure, label';
    for (const el of document.querySelectorAll(reactTargets)) {
        const pk = Object.keys(el).find(k => k.startsWith('__reactProps'));
        if (pk && el[pk]) {
            const props = el[pk];
            if (typeof props.onClick === 'function' ||
                typeof props.onMouseEnter === 'function' ||
                typeof props.onMouseOver === 'function' ||
                typeof props.onPointerEnter === 'function' ||
                typeof props.onPointerOver === 'function' ||
                typeof props.onMouseDown === 'function' ||
                typeof props.onScroll === 'function' ||
                typeof props.onInput === 'function' ||
                typeof props.onChange === 'function') {
                selectorEls.add(el);
            }
        }
    }

    // Find elements with challenge action text (hover here, scroll inside, click me, type here)
    // These may lack React handlers or cursor:pointer but are still interactive.
    const actionTextRe = /\b(hover\s*(here|over|this)|scroll\s*inside|click\s*me|type\s*(here|in\s*this))\b/i;
    for (const el of document.querySelectorAll('div, span, section, p')) {
        if (selectorEls.has(el)) continue;
        const text = (el.textContent || '').trim();
        if (!actionTextRe.test(text)) continue;
        if (el.childElementCount > 5) continue; // Skip large containers
        const rect = el.getBoundingClientRect();
        if (rect.width < 20 || rect.height < 20) continue;
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
        selectorEls.add(el);
    }

    // Find scroll containers (overflow:auto/scroll with scrollable content)
    for (const el of document.querySelectorAll('div, section, main, article, aside, ul, ol, nav')) {
        if (selectorEls.has(el)) continue;
        if (el.scrollHeight > el.clientHeight + 10) {
            const cs = getComputedStyle(el);
            const ov = cs.overflow + ' ' + cs.overflowY;
            if (ov.includes('auto') || ov.includes('scroll')) {
                selectorEls.add(el);
            }
        }
    }

    // Find cursor:pointer elements (non-standard interactive)
    for (const el of document.querySelectorAll('div, span, section, p, li, label')) {
        if (selectorEls.has(el)) continue;
        const cs = getComputedStyle(el);
        if (cs.cursor === 'pointer' && cs.display !== 'none' && cs.visibility !== 'hidden') {
            const rect = el.getBoundingClientRect();
            if (rect.width > 20 && rect.height > 20 && rect.width < 600) {
                selectorEls.add(el);
            }
        }
    }

    // Shadow DOM traversal: find interactive elements inside shadow roots
    // (both open via .shadowRoot and closed via .__shadow captured by browser.py)
    function walkShadow(root, depth) {
        if (!root || depth > 5) return;
        for (const el of root.querySelectorAll('*')) {
            const sr = el.shadowRoot || el.__shadow;
            if (!sr) continue;
            // Tag the host element so executor can find the shadow root
            el.setAttribute('data-wnav-shadow-host', 'true');
            const innerSelector = 'a, button, input, select, textarea, ' +
                '[role="button"], [draggable="true"], [onclick], [tabindex]';
            for (const inner of sr.querySelectorAll(innerSelector)) {
                const cs = getComputedStyle(inner);
                if (cs.display === 'none' || cs.visibility === 'hidden') continue;
                const rect = inner.getBoundingClientRect();
                if (rect.width < 2 || rect.height < 2) continue;
                selectorEls.add(inner);
                // Mark inner element with shadow host reference for executor
                inner.__shadowHost = el;
            }
            // Also add any cursor:pointer elements inside shadow
            for (const inner of sr.querySelectorAll('div, span, section, p')) {
                if (selectorEls.has(inner)) continue;
                const cs = getComputedStyle(inner);
                if (cs.cursor === 'pointer' && cs.display !== 'none') {
                    const rect = inner.getBoundingClientRect();
                    if (rect.width > 20 && rect.height > 20) {
                        selectorEls.add(inner);
                        inner.__shadowHost = el;
                    }
                }
            }
            walkShadow(sr, depth + 1);
        }
    }
    walkShadow(document, 0);

    const els = selectorEls;
    const results = [];
    let idx = 0;
    for (const el of els) {
        const rect = el.getBoundingClientRect();
        // Skip invisible elements (zero size or off-screen)
        if (rect.width < 2 || rect.height < 2) continue;
        if (rect.bottom < 0 || rect.top > window.innerHeight * 3) continue;
        // Skip truly hidden elements (visibility/display)
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
        // Skip popup/noise elements with generic navigation text
        const elText = (el.textContent || '').trim();
        const noiseRe = /^(advance|continue|next|proceed|go forward|keep going|move on|next page|next step|next section|continue reading|continue journey|proceed forward|go|here!?|try this!?|button!?|link!?|moving!?)$/i;
        if (noiseRe.test(elText)) continue;

        // Tag element with data attribute for reliable selection
        el.setAttribute('data-wnav', String(idx));
        let uniqueSelector;
        // Shadow DOM elements need a JS-based selector since CSS selectors
        // can't cross shadow boundaries from document.querySelector
        if (el.__shadowHost) {
            const hostIdx = el.__shadowHost.getAttribute('data-wnav-shadow-host') ? 'true' : '';
            el.__shadowHost.setAttribute('data-wnav-host', String(idx));
            uniqueSelector = 'shadow:' + idx;
        } else {
            uniqueSelector = '[data-wnav="' + idx + '"]';
        }

        // Detect element characteristics
        const isScrollable = el.scrollHeight > el.clientHeight + 10;
        const isCursorPointer = cs.cursor === 'pointer' && el.tagName !== 'BUTTON' && el.tagName !== 'A';
        let extra = '';
        if (isScrollable) extra = 'scrollable';
        if (isCursorPointer) extra += (extra ? ' ' : '') + 'clickable';

        // Name: prefer textContent. For radio/checkbox buttons with no
        // visible text, fall back to value/aria-label (needed for quiz puzzles).
        // Do NOT use value for regular buttons — their value often contains
        // form data, not display text, and creates false names for noise buttons.
        let elName = (el.textContent || '').trim();
        if (!elName) {
            const role = el.getAttribute('role');
            if (role === 'radio' || role === 'checkbox' || role === 'option') {
                elName = el.getAttribute('value') || el.getAttribute('aria-label') || '';
            } else {
                elName = el.getAttribute('aria-label') || el.getAttribute('title') || '';
            }
        }

        results.push({
            index: idx,
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            name: elName.substring(0, 80),
            type: el.getAttribute('type') || '',
            placeholder: el.getAttribute('placeholder') || '',
            selector: uniqueSelector,
            visible: el.offsetParent !== null || el.tagName === 'INPUT',
            bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
            extra: extra
        });
        idx++;
    }

    // Re-hide noise containers after indexing is complete
    __noiseEls.forEach(el => el.style.setProperty('display', 'none', 'important'));

    return results;
})()
"""


async def index_elements(page: Page) -> list[ElementInfo]:
    """Enumerate all interactive elements on the page and return indexed list."""
    try:
        raw = await page.evaluate(_INDEX_ELEMENTS_JS)
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    elements: list[ElementInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        elements.append(ElementInfo(
            index=item.get("index", 0),
            tag=item.get("tag", ""),
            role=item.get("role", ""),
            name=item.get("name", ""),
            type=item.get("type", ""),
            placeholder=item.get("placeholder", ""),
            selector=item.get("selector", ""),
            visible=item.get("visible", True),
            bbox=item.get("bbox"),
            extra=item.get("extra", ""),
        ))
    return elements


# ---------- Node-based helpers (used by both old dict tree and new YAML) ----------

def _walk_tree(node: dict[str, Any], depth: int = 0) -> list[dict[str, Any]]:
    """Flatten a11y tree dict into a list of nodes with depth."""
    results = [{"depth": depth, **{k: v for k, v in node.items() if k != "children"}}]
    for child in node.get("children", []):
        results.extend(_walk_tree(child, depth + 1))
    return results


def _extract_instruction(nodes: list[dict[str, Any]]) -> str:
    """Find the challenge instruction text near headings.

    Skips short labels ending with ":" (e.g. "Click to Reveal:")
    since those are just category names, not the actual instruction.
    """
    instruction_hints = [
        "click", "scroll", "wait", "enter", "find", "reveal",
        "hidden", "inspect", "drag", "type", "hover", "solve",
        "complete", "press", "select", "choose",
    ]
    for node in nodes:
        name = node.get("name", "").strip().strip('"')
        if not name or len(name) < 15:
            continue
        if _NOISE_PATTERNS.search(name):
            continue
        if _LABEL_RE.match(name):
            continue
        name_lower = name.lower()
        if any(hint in name_lower for hint in instruction_hints):
            return name
    for node in nodes:
        name = node.get("name", "").strip().strip('"')
        role = node.get("role", "")
        if role in ("heading", "paragraph", "text") and len(name) > 20:
            if not _NOISE_PATTERNS.search(name) and not _LABEL_RE.match(name):
                return name
    return ""


def _extract_codes(nodes: list[dict[str, Any]]) -> list[str]:
    """Find potential 6-char alphanumeric codes."""
    codes: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        name = node.get("name", "")
        for match in CODE_PATTERN.finditer(name):
            code = match.group(1)
            if code not in seen and not _FALSE_POSITIVE.match(code):
                codes.append(code)
                seen.add(code)
    return codes


def _extract_interactive(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract interactive elements (buttons, textboxes, links)."""
    interactive_roles = {"button", "textbox", "link", "combobox", "checkbox"}
    results: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for node in nodes:
        role = node.get("role", "")
        name = node.get("name", "")
        if role in interactive_roles and name:
            if _NOISE_PATTERNS.search(name):
                continue
            key = f"{role}:{name}"
            if key not in seen_names:
                seen_names.add(key)
                results.append({"role": role, "name": name})
    return results


def _extract_progress(nodes: list[dict[str, Any]]) -> str:
    """Find progress indicators like 'Scrolled: 0px / 500px'."""
    progress_hints = ["scrolled", "progress", "timer", "remaining", "count"]
    for node in nodes:
        name = node.get("name", "")
        if name and any(h in name.lower() for h in progress_hints):
            return name.strip()
    return ""


def _extract_step_number(nodes: list[dict[str, Any]], url: str) -> int:
    """Parse step number from URL or page content."""
    url_match = re.search(r"/step(\d+)", url)
    if url_match:
        return int(url_match.group(1))
    for node in nodes:
        name = node.get("name", "")
        m = re.search(r"step\s*(\d+)", name, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


# ---------- YAML-based parsing for Playwright >=1.49 ----------

# Pattern: - role "text" or - role "text" [attr=val]
_YAML_LINE_RE = re.compile(r"^\s*-\s+(\w+)(?:\s+\"(.+?)\")?(?:\s+\[.+\])?$")


def _parse_yaml_to_nodes(yaml_str: str) -> list[dict[str, Any]]:
    """Parse Playwright YAML aria snapshot into node dicts.

    Example input:
        - heading "Challenge Step 1" [level=1]
        - paragraph: Click the button below to reveal the code
        - button "Reveal Code"
        - textbox "Enter 6-character code"
    """
    nodes: list[dict[str, Any]] = []
    for line in yaml_str.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Try structured line: - role "name"
        m = _YAML_LINE_RE.match(line)
        if m:
            role = m.group(1)
            name = m.group(2) or ""
            nodes.append({"role": role, "name": name})
            continue

        # Try lines like: - paragraph: Some text here
        colon_match = re.match(r"^\s*-\s+(\w+):\s*(.+)$", line)
        if colon_match:
            role = colon_match.group(1)
            name = colon_match.group(2).strip().strip('"')
            nodes.append({"role": role, "name": name})
            continue

        # Plain text lines (might contain codes or instructions)
        # Strip leading "- " and whitespace
        text = re.sub(r"^\s*-\s*", "", stripped)
        if text:
            nodes.append({"role": "text", "name": text})

    return nodes


def _extract_instruction_from_text(text: str) -> str:
    """Extract challenge instruction from raw page text.

    Prefers longer lines that contain action verbs.
    Inner text naturally concatenates label + description.
    """
    instruction_hints = [
        "click", "scroll", "wait", "enter", "find", "reveal",
        "hidden", "inspect", "drag", "type", "hover", "solve",
        "complete", "press", "select", "choose",
    ]
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 15:
            continue
        if _NOISE_PATTERNS.search(line):
            continue
        if _LABEL_RE.match(line):
            continue
        line_lower = line.lower()
        if any(hint in line_lower for hint in instruction_hints):
            return line
    return ""


async def snapshot(page: Page) -> PageState:
    """Take a compressed accessibility snapshot with indexed elements.

    Uses the modern Playwright aria_snapshot() API (v1.49+)
    and enumerates interactive elements for index-based actions.
    """
    url = page.url

    # Get YAML aria snapshot
    try:
        aria_yaml = await page.locator("body").aria_snapshot(timeout=5000)
    except Exception:
        aria_yaml = ""

    # Parse YAML into node dicts for our extraction functions
    nodes = _parse_yaml_to_nodes(aria_yaml) if aria_yaml else []

    step = _extract_step_number(nodes, url)
    progress = _extract_progress(nodes)
    interactive = _extract_interactive(nodes)
    codes = _extract_codes(nodes)

    # Get raw inner_text — naturally concatenates label + description
    try:
        raw_text = await page.inner_text("body")
    except Exception:
        raw_text = ""

    # Prefer inner_text for instruction (concatenated labels), fall back to YAML
    instruction = _extract_instruction_from_text(raw_text)
    if not instruction:
        instruction = _extract_instruction(nodes)

    for match in CODE_PATTERN.finditer(raw_text):
        code = match.group(1)
        if code not in codes and not _FALSE_POSITIVE.match(code):
            codes.append(code)

    # Index interactive elements
    elements = await index_elements(page)

    return PageState(
        step=step,
        instruction=instruction,
        progress=progress,
        interactive=interactive,
        visible_codes=codes,
        raw_text=raw_text,
        aria_yaml=aria_yaml,
        elements=elements,
    )
