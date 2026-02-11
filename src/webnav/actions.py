"""Action dataclass for browser automation commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """A browser action to execute.

    For index-based actions (click, fill, hover, drag), `element` refers
    to the index into the page's indexed element list.
    For non-element actions (scroll, wait, press, js), `element` is ignored.
    """

    type: str  # click, fill, scroll, wait, press, js, hover, drag, canvas_draw, key_sequence
    element: int | None = None  # index into ElementInfo list
    selector: str = ""  # CSS selector fallback (used by submit_code, legacy)
    value: str = ""  # text to type/press/evaluate
    amount: int = 0  # pixels to scroll, seconds to wait, click count
    to_element: int | None = None  # target element for drag actions
    duration: float = 0.0  # hover duration in seconds

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}
