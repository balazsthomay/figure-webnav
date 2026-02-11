"""LLM-based challenge solving via OpenRouter — general-purpose CUA."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from webnav.actions import Action
from webnav.perception import PageState


# OpenRouter models
FAST_MODEL = "google/gemini-3-flash-preview"  # Primary: fast + cheap
SMART_MODEL = "anthropic/claude-haiku-4-5"  # Recovery: vision + reasoning

SYSTEM_PROMPT = """You are a browser automation agent. You see a numbered list of page elements and a description of the current page. Decide what actions to take to complete the challenge and reveal a 6-character code.

Available actions (use element indices from the ELEMENTS list):
- click(element) — click an element by index
- click(element, amount) — click an element N times
- fill(element, text) — type text into an input by index
- scroll(amount) — scroll down by pixels (no element needed)
- hover(element, duration) — hover over element for N seconds
- press(key) — press keyboard key (e.g. "Enter", "ArrowUp")
- wait(seconds) — wait N seconds
- drag(from_element, to_element) — drag and drop between elements
- draw_strokes(element) — draw strokes on a canvas element
- key_sequence(keys) — press multiple keys (e.g. "ArrowUp ArrowDown ArrowLeft")
- js(code) — run JavaScript (use sparingly, only when no element action works)

Respond with a JSON array of actions. Each action is an object:
- {"action": "click", "element": 3}
- {"action": "click", "element": 3, "amount": 5}
- {"action": "fill", "element": 7, "text": "hello world"}
- {"action": "scroll", "amount": 600}
- {"action": "hover", "element": 5, "duration": 3}
- {"action": "press", "key": "Enter"}
- {"action": "wait", "seconds": 5}
- {"action": "drag", "from_element": 2, "to_element": 8}
- {"action": "draw_strokes", "element": 4}
- {"action": "key_sequence", "keys": "ArrowUp ArrowDown"}
- {"action": "js", "code": "document.querySelector('.hidden').style.display='block'"}

Think step by step about what the page is asking you to do. Common patterns:
- "Click the button" → find the button in ELEMENTS, click it
- "Scroll down N px" → scroll(N+100)
- "Wait N seconds" → wait(N+2)
- "Hover over element for N seconds" → hover the element with duration N+1
- "Hidden in DOM" → use js() to reveal hidden elements, then look for codes
- "Drag pieces into slots" → drag each draggable to its target
- "Draw on canvas" → draw_strokes on the canvas element
- "Press keys in sequence" → read the required keys from PAGE_TEXT, use key_sequence
- "Solve the puzzle" → read the math expression from PAGE_TEXT, compute answer, fill the answer input
- If a code is already visible in VISIBLE_CODES, no action needed (agent will submit it)

Return ONLY the JSON array, no explanation."""

STUCK_PROMPT = """You are analyzing a browser challenge where previous attempts failed. Look at the page state and screenshot carefully.

The agent tried standard approaches but couldn't reveal the 6-character code. Consider:
- Hidden elements that need JS to reveal (display:none, visibility:hidden, opacity:0)
- Elements in shadow DOM, iframes, or unusual locations
- Timer-based reveals that need waiting
- Multi-step interactions (click then hover then type)
- Canvas/drawing challenges
- Drag-and-drop challenges
- Elements hidden in HTML comments or data attributes
- React component state containing the code

Use the available actions from the element list. Be creative — try approaches the agent hasn't attempted.

Return ONLY a JSON array of actions, no explanation."""


@dataclass
class Solver:
    """LLM-powered challenge solver via OpenRouter."""

    _client: AsyncOpenAI = field(init=False, repr=False)
    total_tokens: int = 0
    total_calls: int = 0

    def __post_init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    async def solve(self, state: PageState) -> list[Action]:
        """Primary LLM solve using compressed page state with indexed elements."""
        prompt = state.to_prompt()
        return await self._call_llm(FAST_MODEL, SYSTEM_PROMPT, prompt)

    async def solve_stuck(
        self, state: PageState, screenshot: bytes | None = None
    ) -> list[Action]:
        """Recovery LLM with optional screenshot for stuck puzzles."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": STUCK_PROMPT},
        ]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": state.to_prompt()},
        ]
        if screenshot:
            b64 = base64.b64encode(screenshot).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages.append({"role": "user", "content": content})

        return await self._call_llm_raw(SMART_MODEL, messages)

    async def _call_llm(
        self, model: str, system: str, user_msg: str
    ) -> list[Action]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        return await self._call_llm_raw(model, messages)

    async def _call_llm_raw(
        self, model: str, messages: list[dict[str, Any]]
    ) -> list[Action]:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens

            raw = response.choices[0].message.content or "[]"
            return _parse_actions(raw)
        except Exception as e:
            print(f"[solver] LLM call failed ({model}): {e}")
            return []


def _parse_actions(raw: str) -> list[Action]:
    """Parse LLM JSON response into Action objects.

    Handles the new index-based format:
        [{"action": "click", "element": 3}, {"action": "scroll", "amount": 600}]
    """
    # Strip markdown code fences if present
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        data = [data]

    valid_actions = {
        "click", "fill", "scroll", "wait", "press", "js", "hover",
        "select", "drag", "key_sequence", "draw_strokes",
        # Legacy names for backward compat with LLM output
        "drag_fill", "canvas_draw",
    }

    actions: list[Action] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        # Support both "action" and "type" keys
        action_type = item.get("action") or item.get("type", "")

        # Normalize legacy names
        if action_type == "canvas_draw":
            action_type = "draw_strokes"
        if action_type == "drag_fill":
            action_type = "drag"

        if action_type not in valid_actions:
            continue

        actions.append(Action(
            type=action_type,
            element=item.get("element"),
            selector=item.get("selector", ""),
            value=item.get("text") or item.get("value") or item.get("code") or item.get("key") or item.get("keys") or "",
            amount=int(item.get("amount") or item.get("seconds") or 0),
            to_element=item.get("to_element"),
            duration=float(item.get("duration") or 0),
        ))
    return actions
