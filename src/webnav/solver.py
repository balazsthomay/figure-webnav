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
SMART_MODEL = "google/gemini-3-flash-preview"  # Recovery: same model + screenshot

SYSTEM_PROMPT = """\
You are a browser automation agent solving web challenges. You see a numbered list \
of interactive page elements, the page instruction, and visible text. Your job: \
figure out what the challenge wants, take the right actions, and reveal a 6-character code.

## Available actions (use element indices from the ELEMENTS list)

- click(element) — click element by index
- click(element, amount) — click element N times (e.g. "click here 5 times" → amount=5)
- fill(element, text) — type text into an input
- scroll(amount) — scroll page down by pixels
- scroll(element, amount) — scroll inside a scrollable element by pixels
- hover(element, duration) — hover over element for N seconds (use duration 5+ for hover challenges)
- press(key) — press a key ("Enter", "ArrowUp", "Tab", etc.)
- wait(seconds) — wait N seconds
- drag(from_element, to_element) — drag and drop between elements
- draw_strokes(element) — draw shapes on a canvas element
- key_sequence(keys) — press keys in sequence ("ArrowUp ArrowDown ArrowLeft")
- js(code) — run JavaScript (last resort when no element action works)

## JSON format

Respond with ONLY a JSON array. Each action is an object:
{"action": "click", "element": 3}
{"action": "click", "element": 3, "amount": 5}
{"action": "fill", "element": 7, "text": "hello"}
{"action": "scroll", "amount": 600}
{"action": "scroll", "element": 5, "amount": 300}
{"action": "hover", "element": 5, "duration": 5}
{"action": "press", "key": "Enter"}
{"action": "wait", "seconds": 5}
{"action": "drag", "from_element": 2, "to_element": 8}
{"action": "draw_strokes", "element": 4}
{"action": "key_sequence", "keys": "ArrowUp ArrowDown"}
{"action": "js", "code": "document.querySelector('.x').click()"}

## Key principles

1. FOLLOW THE INSTRUCTION LITERALLY. The instruction tells you exactly what to do. \
If it says "click here 3 times", find the clickable element and click it 3 times. \
If it says "scroll down 500px", scroll 500px. Do what it says FIRST, then look for codes.
2. IGNORE noise buttons like "Continue Reading", "Go Forward", "Next Section", "Advance" \
— these are popup noise, not challenge elements. Focus on elements that match the instruction.
3. After completing the main action, ALWAYS click any "Reveal Code", "Complete", \
"Complete Challenge", or similar button visible in ELEMENTS.
4. For multi-step challenges (tabs, sequences, levels), complete ALL steps in one \
action list — click every tab, fill every input, click every level button.
5. For puzzles: They may have MULTIPLE parts — radio/multiple-choice questions AND a \
math input. Click the obviously-correct radio options first (labels containing "correct", \
"right choice", etc.), THEN compute the math answer, fill(element, answer) into the \
puzzle input (NOT the "Enter 6-character code" input), then click "Solve" (NOT "Submit Code"). \
IMPORTANT: Use fill() for inputs, NOT js() — React needs proper events.
6. For hover challenges: ALWAYS use the hover(element, duration) action, NEVER use \
js() for hover — JS event dispatch does not trigger React hover handlers. The hover \
target is usually a colored div/section with "hover" text. Use duration=5 or more.
7. For drag challenges: use drag(from_element, to_element) — NEVER use js() for drag.
7b. For canvas challenges: ALWAYS use draw_strokes(element), NEVER use js() for canvas \
drawing — JS synthetic events don't trigger React canvas handlers.
8. Use js() only when you need to interact with elements not in ELEMENTS \
(shadow DOM, hidden elements, iframes, etc.). Do NOT use js() to search for codes \
before trying the instruction first.
9. If previous attempts failed, try a DIFFERENT approach — don't repeat what didn't work.
10. When the instruction says "click here" — find the element whose text contains "here" \
or "Here" (it may be a div/span with onClick, not a button). Use js() to click it if \
not in ELEMENTS: js("document.querySelector('[data-wnav]').click()") or search by text.
11. Elements marked [scrollable] have overflow content — use scroll(element, amount) to scroll inside them. \
Elements marked [clickable] have cursor:pointer — they respond to click actions.
"""

STUCK_PROMPT = """\
You are analyzing a browser challenge where previous attempts FAILED. \
Look at the screenshot and page state carefully. The standard approach didn't work.

IMPORTANT: Use element indices from the ELEMENTS list for all actions. \
Use {"action": "click", "element": 0} NOT {"action": "click", "selector": "button:contains('...')"} \
— CSS :contains() does not exist. Always reference elements by their index number.

Available actions (same as primary):
- click(element), click(element, amount), fill(element, text), scroll(amount), \
scroll(element, amount), hover(element, duration), press(key), wait(seconds), \
drag(from_element, to_element), draw_strokes(element), key_sequence(keys), js(code)

Try different approaches from what failed:
1. If clicking didn't work, try the instruction literally — "click here 3 times" means click(element, 3)
2. For hover challenges: hover(element, 5) — NEVER js() for hover
3. For "click here" instructions: the target is usually a div/span, not a button
4. Hidden codes may be in: shadow DOM, iframes, HTML attributes, comments, data-* attributes, \
sessionStorage/localStorage, or React component state
5. For sequence/multi-action challenges: complete ALL actions (click, fill, scroll, hover) \
using the appropriate action types on the indexed elements

Use js() for code discovery (not interaction):
- Shadow DOM: js("(() => { function walk(r,d){if(d>10)return null; for(const e of r.querySelectorAll('*')){const s=e.shadowRoot||e.__shadow; if(s){const c=walk(s,d+1); if(c)return c;}} const m=(r.textContent||'').match(/\\\\b([A-Z0-9]{6})\\\\b/); return m?m[1]:null;} return walk(document,0); })()")
- Storage: js("[...Array(sessionStorage.length)].map((_,i)=>sessionStorage.getItem(sessionStorage.key(i))).join(' ')")
- Attributes: js("Array.from(document.querySelectorAll('*')).flatMap(e=>[...e.attributes].map(a=>a.value)).filter(v=>/^[A-Z0-9]{6}$/.test(v))[0]")

Return ONLY a JSON array of actions.\
"""


def _instruction_hints(instruction: str) -> str:
    """Generate action hints based on instruction text. Not site-specific —
    reads the instruction any challenge site would provide."""
    inst = instruction.lower()
    hints: list[str] = []
    if "hover" in inst:
        hints.append(
            "CRITICAL: Use hover(element, duration) with duration=5. "
            "Do NOT use js() for hover — JS events don't trigger React handlers."
        )
    if "drag" in inst:
        hints.append(
            "CRITICAL: Use drag(from_element, to_element). Do NOT use js() for drag."
        )
    if "canvas" in inst or "draw" in inst:
        hints.append(
            "CRITICAL: Use draw_strokes(element). Do NOT use js() for canvas drawing."
        )
    if "sequence" in inst or ("all" in inst and ("action" in inst or "step" in inst)):
        hints.append(
            "This is a multi-action challenge. Complete EACH action using the "
            "appropriate command. Look for [scrollable] and [clickable] elements. "
            "Use scroll(element, amount) for scrollable containers. "
            "For fill(element, text): type actual text like 'hello' — never empty string. "
            "For hover in sequences: use duration=2 (NOT 5) — shorter hover registers faster."
        )
    if "timing" in inst or "capture" in inst or "while" in inst and "active" in inst:
        hints.append(
            "TIMING CHALLENGE: Use js() to poll for the active state and click. Example: "
            "js(\"(async()=>{for(let i=0;i<50;i++){const btn=Array.from(document.querySelectorAll('button'))"
            ".find(b=>/capture|complete/i.test(b.textContent));if(btn){btn.click();await new Promise(r=>setTimeout(r,200));"
            "btn.click();}await new Promise(r=>setTimeout(r,200));}})()\")"
        )
    if "click here" in inst or "click the" in inst:
        # Extract click count if mentioned
        import re
        m = re.search(r"click\s+(?:here\s+)?(\d+)", inst)
        if m:
            n = m.group(1)
            hints.append(
                f"IMPORTANT: Use click(element, {n}) to click the target element "
                f"{n} times. The click target may be a div/span, not a button."
            )
    return "\n".join(hints)


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

    async def solve(
        self, state: PageState, history: list[str] | None = None
    ) -> list[Action]:
        """Primary LLM solve using compressed page state with indexed elements."""
        prompt = state.to_prompt()
        if history:
            prompt += "\n\nPREVIOUS ATTEMPTS (failed — try something different):\n"
            prompt += "\n".join(history)
        # Instruction-aware hints (placed at end of prompt, near LLM response)
        hints = _instruction_hints(state.instruction)
        if hints:
            prompt += "\n\n" + hints
        return await self._call_llm(FAST_MODEL, SYSTEM_PROMPT, prompt)

    async def solve_stuck(
        self,
        state: PageState,
        screenshot: bytes | None = None,
        history: list[str] | None = None,
    ) -> list[Action]:
        """Recovery LLM with optional screenshot for stuck puzzles."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": STUCK_PROMPT},
        ]
        prompt = state.to_prompt()
        if history:
            prompt += "\n\nPREVIOUS ATTEMPTS (all failed):\n"
            prompt += "\n".join(history)
        hints = _instruction_hints(state.instruction)
        if hints:
            prompt += "\n\n" + hints
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
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
                max_tokens=1024,
                temperature=0.0,
            )
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens

            # Defensive: handle missing/empty choices
            if not response.choices or not response.choices[0].message:
                print(f"[solver] LLM returned empty response ({model})")
                return []

            raw = response.choices[0].message.content or "[]"
            return _parse_actions(raw)
        except Exception as e:
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
        # Try to extract JSON array from mixed text/JSON response
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if arr_match:
            try:
                data = json.loads(arr_match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        data = [data]

    valid_actions = {
        "click", "fill", "scroll", "wait", "press", "js", "hover",
        "select", "drag", "key_sequence", "draw_strokes",
        "drag_fill", "canvas_draw",
    }

    actions: list[Action] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        action_type = item.get("action") or item.get("type", "")

        # Normalize legacy names
        if action_type == "canvas_draw":
            action_type = "draw_strokes"
        if action_type == "drag_fill":
            action_type = "drag"

        if action_type not in valid_actions:
            continue

        # Safely coerce element/to_element to int
        try:
            raw_el = item.get("element")
            element = int(raw_el) if raw_el is not None else None
        except (ValueError, TypeError):
            element = None
        try:
            raw_to = item.get("to_element")
            to_element = int(raw_to) if raw_to is not None else None
        except (ValueError, TypeError):
            to_element = None

        actions.append(Action(
            type=action_type,
            element=element,
            selector=item.get("selector", ""),
            value=item.get("text") or item.get("value") or item.get("code") or item.get("key") or item.get("keys") or "",
            amount=int(item.get("amount") or item.get("seconds") or 0),
            to_element=to_element,
            duration=float(item.get("duration") or 0),
        ))
    return actions
