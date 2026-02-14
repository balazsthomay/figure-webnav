"""Shared test fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake YAML aria snapshots (representative of real challenge pages)
# ---------------------------------------------------------------------------

STEP1_ARIA_YAML = """\
- heading "Challenge Step 1" [level=1]
- paragraph "Click the button below to reveal the code"
- button "Reveal Code"
- button "Click Me!"
- button "Button!"
- group:
  - textbox "Enter 6-character code"
  - button "Submit Code"
- text "Content Block Loaded"
- text "Lorem ipsum dolor sit amet"
"""

STEP2_ARIA_YAML = """\
- heading "Challenge Step 2" [level=1]
- paragraph "Scroll down at least 500px to reveal the code"
- text "Scrolled: 0px / 500px"
- text "Content Block Loaded"
- group:
  - textbox "Enter 6-character code"
  - button "Submit Code"
"""

STEP3_ARIA_YAML_WITH_CODE = """\
- heading "Challenge Step 3" [level=1]
- paragraph "Wait 3 seconds for the code to appear"
- text "Your code is: AB3F9X"
- group:
  - textbox "Enter 6-character code"
  - button "Submit Code"
"""

STEP5_ARIA_YAML = """\
- heading "Challenge Step 5" [level=1]
- paragraph "The code is hidden in the DOM. Inspect the page to find it."
- group:
  - textbox "Enter 6-character code"
  - button "Submit Code"
"""

# Legacy dict trees (still used by some unit tests of internal helpers)
STEP1_A11Y_TREE: dict[str, Any] = {
    "role": "WebArea",
    "name": "",
    "children": [
        {"role": "heading", "name": "Challenge Step 1", "level": 1},
        {"role": "paragraph", "name": "Click the button below to reveal the code"},
        {"role": "button", "name": "Reveal Code"},
        {"role": "button", "name": "Click Me!"},
        {"role": "button", "name": "Button!"},
        {
            "role": "group",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Enter 6-character code"},
                {"role": "button", "name": "Submit Code"},
            ],
        },
        {"role": "text", "name": "Content Block Loaded"},
        {"role": "text", "name": "Lorem ipsum dolor sit amet"},
    ],
}

STEP2_A11Y_TREE: dict[str, Any] = {
    "role": "WebArea",
    "name": "",
    "children": [
        {"role": "heading", "name": "Challenge Step 2", "level": 1},
        {"role": "paragraph", "name": "Scroll down at least 500px to reveal the code"},
        {"role": "text", "name": "Scrolled: 0px / 500px"},
        {"role": "text", "name": "Content Block Loaded"},
        {
            "role": "group",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Enter 6-character code"},
                {"role": "button", "name": "Submit Code"},
            ],
        },
    ],
}

STEP3_A11Y_TREE_WITH_CODE: dict[str, Any] = {
    "role": "WebArea",
    "name": "",
    "children": [
        {"role": "heading", "name": "Challenge Step 3", "level": 1},
        {"role": "paragraph", "name": "Wait 3 seconds for the code to appear"},
        {"role": "text", "name": "Your code is: AB3F9X"},
        {
            "role": "group",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Enter 6-character code"},
                {"role": "button", "name": "Submit Code"},
            ],
        },
    ],
}

STEP5_A11Y_TREE: dict[str, Any] = {
    "role": "WebArea",
    "name": "",
    "children": [
        {"role": "heading", "name": "Challenge Step 5", "level": 1},
        {"role": "paragraph", "name": "The code is hidden in the DOM. Inspect the page to find it."},
        {
            "role": "group",
            "name": "",
            "children": [
                {"role": "textbox", "name": "Enter 6-character code"},
                {"role": "button", "name": "Submit Code"},
            ],
        },
    ],
}


def make_mock_page(
    url: str = "https://serene-frangipane-7fd25b.netlify.app/step1?version=2",
    aria_yaml: str | None = None,
    inner_text: str = "",
    # Legacy support
    a11y_tree: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock Playwright Page with common methods."""
    page = MagicMock()
    page.url = url

    # Common async methods
    page.evaluate = AsyncMock(return_value=None)
    page.inner_text = AsyncMock(return_value=inner_text)
    page.screenshot = AsyncMock(return_value=b"fake_png_data")
    page.wait_for_timeout = AsyncMock()
    page.goto = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.press = AsyncMock()
    page.mouse = MagicMock()
    page.mouse.move = AsyncMock()
    page.mouse.down = AsyncMock()
    page.mouse.up = AsyncMock()
    page.mouse.wheel = AsyncMock()

    # Locator support — must handle both general locators and body aria_snapshot
    mock_locator = MagicMock()
    mock_locator.first = mock_locator
    mock_locator.is_visible = AsyncMock(return_value=True)
    mock_locator.click = AsyncMock()
    mock_locator.fill = AsyncMock()
    mock_locator.hover = AsyncMock()
    mock_locator.press = AsyncMock()
    mock_locator.select_option = AsyncMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_locator.nth = MagicMock(return_value=mock_locator)

    # aria_snapshot on the body locator
    yaml_to_use = aria_yaml or STEP1_ARIA_YAML
    mock_locator.aria_snapshot = AsyncMock(return_value=yaml_to_use)

    page.locator = MagicMock(return_value=mock_locator)
    page.get_by_text = MagicMock(return_value=mock_locator)

    return page
