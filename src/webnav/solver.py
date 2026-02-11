"""Tier 1+2: LLM-based challenge solving via OpenRouter."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from webnav.dispatcher import Action
from webnav.perception import PageState


# OpenRouter models
FAST_MODEL = "google/gemini-3-flash-preview"  # Tier 1: fast + cheap
SMART_MODEL = "anthropic/claude-haiku-4-5"  # Tier 2: vision + reasoning

SYSTEM_PROMPT = """You are a browser automation agent solving challenges on a website.
Each challenge requires you to perform an action (click, scroll, wait, etc.) to reveal a 6-character alphanumeric code.

Given the page state, return a JSON array of actions to perform.
Each action is an object with:
- "type": one of "click", "fill", "scroll", "wait", "press", "js", "hover"
- "selector": CSS selector or text selector (for click/fill/hover)
- "value": value to type/press/evaluate (for fill/press/js)
- "amount": numeric amount (for scroll px or wait seconds)

Examples:
- [{"type": "click", "selector": "button:has-text('Reveal Code')"}]
- [{"type": "scroll", "amount": 600}]
- [{"type": "wait", "amount": 5}]
- [{"type": "js", "value": "document.querySelector('.hidden').style.display='block'"}]

Return ONLY the JSON array, no explanation."""

STUCK_PROMPT = """You are analyzing a browser challenge where the agent is stuck.
Look at the page state and screenshot carefully.

What action should be taken to reveal the 6-character code?
Consider:
- Hidden elements that need to be revealed
- Buttons that look like decoys but are real
- Timer-based reveals
- Scroll-triggered reveals
- Interaction sequences

Return a JSON array of actions. Be creative — the obvious approach already failed.
Return ONLY the JSON array, no explanation."""


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
        """Tier 1: Fast LLM solve using compressed page state."""
        prompt = state.to_prompt()
        return await self._call_llm(FAST_MODEL, SYSTEM_PROMPT, prompt)

    async def solve_stuck(
        self, state: PageState, screenshot: bytes | None = None
    ) -> list[Action]:
        """Tier 2: Smart LLM with optional screenshot for stuck recovery."""
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
                max_tokens=256,
                temperature=0.0,
            )
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens

            raw = response.choices[0].message.content or "[]"
            return _parse_actions(raw)
        except Exception as e:
            # Fallback: return empty actions on LLM failure
            print(f"[solver] LLM call failed ({model}): {e}")
            return []


def _parse_actions(raw: str) -> list[Action]:
    """Parse LLM JSON response into Action objects."""
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

    actions: list[Action] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action_type = item.get("type", "")
        if action_type not in {"click", "fill", "scroll", "wait", "press", "js", "hover", "select", "drag_fill", "key_sequence", "canvas_draw"}:
            continue
        actions.append(Action(
            type=action_type,
            selector=item.get("selector", ""),
            value=item.get("value", ""),
            amount=int(item.get("amount", 0)),
        ))
    return actions
