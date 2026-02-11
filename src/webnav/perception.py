"""A11y tree extraction and compression into a compact page state."""

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
_FALSE_POSITIVE = re.compile(r"^(SUBMIT|SCROLL|CLICK[S]?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\d+)$")


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

    def to_prompt(self) -> str:
        """Render as compact text for LLM consumption."""
        parts = [
            f"STEP: {self.step}/30",
            f'INSTRUCTION: "{self.instruction}"',
        ]
        if self.progress:
            parts.append(f'PROGRESS: "{self.progress}"')
        if self.interactive:
            items = [f'{e["role"]} "{e["name"]}"' for e in self.interactive[:8]]
            parts.append(f"INTERACTIVE: [{', '.join(items)}]")
        if self.visible_codes:
            parts.append(f"VISIBLE_CODES: {self.visible_codes}")
        if self.aria_yaml:
            # Include truncated YAML for LLM context
            truncated = "\n".join(self.aria_yaml.splitlines()[:50])
            parts.append(f"ARIA_TREE:\n{truncated}")
        return "\n".join(parts)


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
    """Take a compressed accessibility snapshot of the current page.

    Uses the modern Playwright aria_snapshot() API (v1.49+).
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

    return PageState(
        step=step,
        instruction=instruction,
        progress=progress,
        interactive=interactive,
        visible_codes=codes,
        raw_text=raw_text,
        aria_yaml=aria_yaml,
    )
