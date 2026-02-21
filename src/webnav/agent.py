"""Orchestrator: main agent loop — general-purpose CUA, no site-specific hacks."""

from __future__ import annotations

import asyncio
import re
import time

from webnav.actions import Action
from webnav.browser import BrowserController
from webnav.config import ChallengeConfig
from webnav.executor import run as execute_action, submit_code
from webnav.extractor import find_code
from webnav.metrics import MetricsCollector
from webnav.page_cleaner import clean_page, quick_clean, reset_cleaner
from webnav.perception import ElementInfo, snapshot
from webnav.solver import Solver
from webnav.state import StateTracker


def _parse_instruction_actions(
    instruction: str, elements: list[ElementInfo],
) -> list[Action]:
    """Parse obvious action directives from instruction text.

    This is generic instruction parsing — not challenge-type matching.
    Every challenge provides an instruction telling you what to do;
    we just read common patterns: 'click here N times', 'scroll down Npx',
    'wait N seconds'.
    """
    inst = instruction.lower()
    actions: list[Action] = []

    # "click here N [more] times" — use JS for reliable targeting
    m = re.search(r"click (?:here |the \w+ )?(\d+)\s*(?:more )?times?", inst)
    if m:
        n = int(m.group(1))
        # JS finds the smallest (deepest) element containing "click here"
        # Uses fullClick() + real-time delays so React processes each click
        js = (
            "(async () => {"
            "  function fullClick(el) {"
            "    const o = {bubbles:true, cancelable:true, view:window};"
            "    el.dispatchEvent(new PointerEvent('pointerdown', o));"
            "    el.dispatchEvent(new MouseEvent('mousedown', o));"
            "    el.dispatchEvent(new PointerEvent('pointerup', o));"
            "    el.dispatchEvent(new MouseEvent('mouseup', o));"
            "    el.dispatchEvent(new MouseEvent('click', o));"
            "    el.click();"
            "  }"
            "  let best = null, bestArea = Infinity, bestChildren = Infinity;"
            "  for (const el of document.querySelectorAll('div, span, p, section, button')) {"
            "    const t = (el.textContent || '').trim();"
            "    if (!/click here/i.test(t)) continue;"
            "    const cs = getComputedStyle(el);"
            "    if (cs.display === 'none' || cs.visibility === 'hidden') continue;"
            "    const r = el.getBoundingClientRect();"
            "    if (r.width < 10 || r.height < 10) continue;"
            "    const a = r.width * r.height;"
            "    const ch = el.children.length;"
            # Prefer smaller area; if within 10% of current best, prefer fewer children (leaf)
            "    if (a < bestArea * 0.9 || (a < bestArea * 1.1 && ch < bestChildren)) {"
            "      best = el; bestArea = a; bestChildren = ch;"
            "    }"
            "  }"
            f" if (best) {{ for (let i = 0; i < {n}; i++) {{"
            "    fullClick(best);"
            # Real-time delays via __origST — timer acceleration would reduce
            # 400ms to 40ms, too fast for React state updates between clicks
            "    await new Promise(r => (window.__origST||setTimeout)(r, 400));"
            "  }"
            "  await new Promise(r => (window.__origST||setTimeout)(r, 500));"
            f"  return 'clicked {n}x on ' + best.tagName; }}"
            "  return 'no target';"
            "})()"
        )
        actions.append(Action(type="js", value=js))

    # "scroll down [at least] Npx" — always scroll the main page, not a modal.
    # Scrollable elements are used for sequence challenges, not page-level scrolls.
    m = re.search(r"scroll down (?:at least )?(\d+)\s*px", inst)
    if m:
        px = int(m.group(1))
        actions.append(Action(type="scroll", amount=px))

    # "wait[ing] [for] N seconds"
    m = re.search(r"wait(?:ing)? (?:for )?(\d+) seconds?", inst)
    if m:
        secs = int(m.group(1))
        actions.append(Action(type="wait", amount=secs))

    # "drag" / "drag-and-drop" — executor handles piece/slot finding
    if "drag" in inst:
        actions.append(Action(type="drag"))

    # "navigate through N layers/levels" — click the level/reveal button N times
    # Skip if iframe-specific handler already handles this instruction.
    # Uses Playwright trusted clicks (React requires trusted events).
    m_layers = re.search(r"(?:navigate|click) (?:through |each )?(\d+)\s*(?:nested )?(?:layer|level)", inst)
    if m_layers and "iframe" not in inst and "shadow" not in inst:
        n_layers = int(m_layers.group(1))
        for _ in range(n_layers + 1):  # +1 for safety margin
            actions.append(Action(
                type="click", element=None,
                value="layer_button",  # Marker for layer navigation
            ))

    # "hover" — find the hover target element (prefer smallest "hover here" element)
    if "hover" in inst:
        hover_el = None
        hover_candidates: list[ElementInfo] = []
        for el in elements:
            name_l = el.name.lower()
            # Skip submit/code form elements only (buttons/inputs).
            # Divs/spans with "code" in name are legit targets (e.g. "Hover here to reveal code").
            if el.tag in ("input", "button") and ("submit" in name_l or "code" in name_l):
                continue
            if "hover here" in name_l or "hover over" in name_l:
                hover_candidates.append(el)
        # Pick the best hover target: prefer clickable elements, then smallest bbox.
        # Clickable elements (cursor:pointer) are the actual React event targets,
        # not their container divs.
        if hover_candidates:
            def _hover_sort_key(e: ElementInfo) -> tuple[int, float]:
                is_clickable = "clickable" in (e.extra or "")
                area = (e.bbox["width"] * e.bbox["height"]) if e.bbox else float("inf")
                return (0 if is_clickable else 1, area)
            hover_el = min(hover_candidates, key=_hover_sort_key)
        # Fallback: any clickable element with "hover" in text
        if hover_el is None:
            for el in elements:
                name_l = el.name.lower()
                if el.tag in ("input", "button") and ("submit" in name_l or "code" in name_l):
                    continue
                if "hover" in name_l and el.extra and "clickable" in el.extra:
                    hover_el = el
                    break
        if hover_el:
            actions.append(Action(
                type="hover", element=hover_el.index, duration=1.0,
            ))
        elif "complete all" not in inst:
            # JS fallback: find hover target by text and dispatch hover events
            # Uses the executor's full hover fallback (JS coords → Playwright mouse)
            actions.append(Action(type="hover", duration=1.0))

    # Rotating code / capture challenge: click Capture N times
    if "capture" in inst:
        m_cap = re.search(r"(\d+)\s*times", inst)
        n_cap = int(m_cap.group(1)) if m_cap else 3
        capture_js = (
            f"(async () => {{"
            f"  for (let i = 0; i < {n_cap + 1}; i++) {{"
            "    const btn = Array.from(document.querySelectorAll('button'))"
            "      .find(b => /capture/i.test(b.textContent));"
            "    if (btn) btn.click();"
            "    await new Promise(r => setTimeout(r, 1000));"
            "  }"
            f"  return 'captured {n_cap + 1} times';"
            "})()"
        )
        actions.append(Action(type="js", value=capture_js))

    # Mutation challenge: click Trigger Mutation N times then Complete
    if "mutation" in inst and "trigger" in inst:
        m_count = re.search(r"(\d+)\s*(?:dom )?mutation", inst)
        n_mutations = int(m_count.group(1)) if m_count else 5
        mutation_js = (
            f"(async () => {{"
            f"  for (let i = 0; i < {n_mutations}; i++) {{"
            "    const btn = Array.from(document.querySelectorAll('button'))"
            "      .find(b => /trigger/i.test(b.textContent));"
            "    if (btn) btn.click();"
            "    await new Promise(r => setTimeout(r, 300));"
            "  }"
            "  await new Promise(r => setTimeout(r, 500));"
            "  const completeBtn = Array.from(document.querySelectorAll('button'))"
            "    .find(b => /complete/i.test(b.textContent));"
            "  if (completeBtn) completeBtn.click();"
            f"  return 'triggered {n_mutations} mutations';"
            "})()"
        )
        actions.append(Action(type="js", value=mutation_js))

    # Audio challenge: play audio, wait for Complete button to appear, click it
    if "audio" in inst and "play" in inst:
        audio_js = """(async () => {
            const playBtn = Array.from(document.querySelectorAll('button'))
                .find(b => /play/i.test(b.textContent));
            if (playBtn) playBtn.click();
            for (let i = 0; i < 20; i++) {
                await new Promise(r => (window.__origST||setTimeout)(r, 500));
                const completeBtn = Array.from(document.querySelectorAll('button'))
                    .find(b => /^complete$/i.test(b.textContent.trim()));
                if (completeBtn && completeBtn.offsetParent !== null) {
                    completeBtn.click();
                    return 'audio: played + clicked Complete';
                }
            }
            return 'audio: played but no Complete button found';
        })()"""
        actions.append(Action(type="js", value=audio_js))

    # "click [ButtonName]" / timing challenges — find and click by button name
    # General pattern: instruction says "click 'Something'" with a quoted button name
    m = re.search(r"""click\s+["'](\w+)["']""", inst)
    if m and "audio" not in inst:  # Skip for audio (handled above)
        btn_name = m.group(1)
        # For timing challenges, click repeatedly in a polling loop
        if "timing" in inst or "while" in inst or "window" in inst:
            js = (
                "(async()=>{"
                f"  const name = '{btn_name}';"
                "  for(let i=0;i<30;i++){"
                "    const btn=Array.from(document.querySelectorAll('button'))"
                "      .find(b=>b.textContent.trim().toLowerCase().includes(name.toLowerCase()));"
                "    if(btn && btn.offsetParent!==null){"
                "      btn.click();"
                "      await new Promise(r=>setTimeout(r,300));"
                "    }"
                "    await new Promise(r=>setTimeout(r,200));"
                "  }"
                "  return 'clicked '+name+' in loop';"
                "})()"
            )
            actions.append(Action(type="js", value=js))
        else:
            # Simple button click by name
            js = (
                "(() => {"
                f"  const name = '{btn_name}';"
                "  const btn = Array.from(document.querySelectorAll('button'))"
                "    .find(b => b.textContent.trim().toLowerCase().includes(name.toLowerCase()));"
                "  if (btn) { btn.click(); return 'clicked ' + name; }"
                "  return 'button not found: ' + name;"
                "})()"
            )
            actions.append(Action(type="js", value=js))

    # "video" / "seek" / "frame" — click seek buttons and complete
    if "seek" in inst or ("frame" in inst and "video" in inst):
        video_js = """(async () => {
            const frameBtn = Array.from(document.querySelectorAll('button'))
                .find(b => /frame\\s*\\d+/i.test(b.textContent));
            if (frameBtn) { frameBtn.click(); await new Promise(r => setTimeout(r, 200)); }
            for (let round = 0; round < 5; round++) {
                const seekBtns = Array.from(document.querySelectorAll('button'))
                    .filter(b => /^[+-]\\d+$/.test(b.textContent.trim()));
                for (const btn of seekBtns) {
                    btn.click();
                    await new Promise(r => setTimeout(r, 150));
                }
                const moreBtn = Array.from(document.querySelectorAll('button'))
                    .find(b => /seek.*more/i.test(b.textContent));
                if (moreBtn) { moreBtn.click(); await new Promise(r => setTimeout(r, 200)); }
                const completeBtn = Array.from(document.querySelectorAll('button'))
                    .find(b => /^(complete|reveal|done|finish)$/i.test(b.textContent.trim()));
                if (completeBtn) { completeBtn.click(); return 'video: completed after ' + (round+1) + ' rounds'; }
            }
            return 'video: seeked 5 rounds';
        })()"""
        actions.append(Action(type="js", value=video_js))

    # "canvas" / "draw" — executor can find canvas via locator
    if "canvas" in inst or ("draw" in inst and "stroke" in inst):
        canvas_el = next((el for el in elements if el.tag == "canvas"), None)
        if canvas_el:
            actions.append(Action(type="draw_strokes", element=canvas_el.index))
        else:
            actions.append(Action(type="draw_strokes"))

    # "puzzle" / "solve" — click correct radio per group, solve math, click Solve
    # Guard: "part of the puzzle" is Multi-Tab, not a puzzle challenge
    if ("puzzle" in inst or "solve" in inst) and "scroll" not in inst and "tab" not in inst:
        actions.append(Action(type="js", value=_PUZZLE_SOLVE_JS))

    # "visit all N tabs" / multi-tab challenge — click all tab buttons
    if "tab" in inst and ("visit" in inst or "click each" in inst):
        tab_js = """(async () => {
            const tabBtns = Array.from(document.querySelectorAll('button'))
                .filter(b => /^tab\\s*\\d+$/i.test(b.textContent.trim()))
                .sort((a, b) => a.textContent.localeCompare(b.textContent));
            for (const btn of tabBtns) {
                btn.click();
                await new Promise(r => setTimeout(r, 400));
            }
            await new Promise(r => setTimeout(r, 500));
            const doneBtn = Array.from(document.querySelectorAll('button'))
                .find(b => /remember|visited|all.*tab|complete|reveal/i.test(b.textContent));
            if (doneBtn) doneBtn.click();
            return 'visited ' + tabBtns.length + ' tabs';
        })()"""
        actions.append(Action(type="js", value=tab_js))

    # "iframe" / "nested levels/layers" — recursively click through nested levels
    # Also matches Shadow DOM challenge (same mechanism: click N nested elements)
    _has_nested = "level" in inst or "nested" in inst or "layer" in inst or "navigate" in inst
    if ("iframe" in inst or "shadow" in inst) and _has_nested:
        m_levels = re.search(r"(\d+)\s*(?:nested\s*)?(?:levels?|layers?)", inst)
        n_levels = int(m_levels.group(1)) if m_levels else 5
        iframe_js = f"""(async () => {{
            // Clear stale marker from previous steps
            const old = document.getElementById('wnav-discovered-code');
            if (old) old.remove();
            const maxDepth = {n_levels};
            const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+|DECODE|STRING|BASE64|PLEASE|SELECT|OPTION)$/;
            const codeRe = /\\b([A-Z0-9]{{6}})\\b/g;
            function findCode(text) {{
                const matches = (text || '').match(codeRe);
                return matches ? matches.find(c => !fp.test(c)) : null;
            }}
            function surfaceCode(code) {{
                // Inject code into a visible marker div so find_code() can pick it up
                let marker = document.getElementById('wnav-discovered-code');
                if (!marker) {{
                    marker = document.createElement('div');
                    marker.id = 'wnav-discovered-code';
                    marker.style.cssText = 'position:fixed;top:0;left:0;z-index:99999;background:white;color:black;font-size:20px;padding:10px;';
                    document.body.appendChild(marker);
                }}
                marker.textContent = code;
                return 'iframe-code:' + code;
            }}
            function fullClick(el) {{
                const r = el.getBoundingClientRect();
                const o = {{bubbles:true, cancelable:true, view:window,
                           clientX:r.x+r.width/2, clientY:r.y+r.height/2}};
                el.dispatchEvent(new PointerEvent('pointerdown', o));
                el.dispatchEvent(new MouseEvent('mousedown', o));
                el.dispatchEvent(new PointerEvent('pointerup', o));
                el.dispatchEvent(new MouseEvent('mouseup', o));
                el.dispatchEvent(new MouseEvent('click', o));
                el.click();
            }}
            function findBtn(doc) {{
                // First try buttons (iframe-style "Enter Level N" buttons)
                const btn = Array.from(doc.querySelectorAll('button'))
                    .find(b => {{
                        const t = b.textContent.trim();
                        return /enter|next|navigate|go deeper/i.test(t)
                            && !/submit code|reveal/i.test(t) && b.offsetParent !== null;
                    }});
                if (btn) return btn;
                // Then try clickable divs (shadow DOM-style levels)
                // Find deepest matching div first (innermost = next uncompleted level)
                const divs = Array.from(doc.querySelectorAll('div'))
                    .filter(b => {{
                        const s = getComputedStyle(b);
                        return s.cursor === 'pointer'
                            && /level|layer/i.test(b.textContent)
                            && b.offsetParent !== null;
                    }})
                    .reverse();  // deepest first (querySelectorAll = document order)
                // Pick the deepest one whose own heading hasn't been checked
                for (const d of divs) {{
                    // Check only the element's own direct text (not children's text)
                    const own = Array.from(d.childNodes)
                        .filter(n => n.nodeType === 3)
                        .map(n => n.textContent).join('');
                    const heading = d.querySelector('h1,h2,h3,h4,h5,h6,span');
                    const label = heading ? heading.textContent : own;
                    if (label && /level|layer/i.test(label) && !label.includes('✓'))
                        return d;
                }}
                return null;
            }}
            async function waitIframe(doc) {{
                for (let t = 0; t < 6; t++) {{
                    const iframe = doc.querySelector('iframe');
                    if (iframe && iframe.contentDocument) return iframe.contentDocument;
                    await new Promise(r => (window.__origST||setTimeout)(r, 300));
                }}
                return null;
            }}
            // Phase 1: Click buttons in main doc first (they may render inline)
            let depth = 0;
            for (let i = 0; i < maxDepth + 3; i++) {{
                const btn = findBtn(document);
                if (!btn) break;
                btn.click();
                depth++;
                await new Promise(r => (window.__origST||setTimeout)(r, 400));
                const code = findCode(document.body.innerText);
                if (code) return surfaceCode(code);
            }}
            // Phase 2: Traverse nested iframes for buttons and code
            let doc = document;
            for (let i = depth; i < maxDepth + 3; i++) {{
                const iframeDoc = await waitIframe(doc);
                if (!iframeDoc) break;
                doc = iframeDoc;
                // Check for code at this level
                const code = findCode(doc.body?.innerText);
                if (code) return surfaceCode(code);
                // Click level buttons inside this iframe
                const btn = findBtn(doc);
                if (btn) {{
                    btn.click();
                    depth++;
                    await new Promise(r => (window.__origST||setTimeout)(r, 400));
                    const code2 = findCode(doc.body?.innerText);
                    if (code2) return surfaceCode(code2);
                }}
            }}
            // Phase 3: Click extract/reveal buttons at deepest level
            // Also check main document (shadow DOM levels are in document, not iframes)
            const searchDocs = doc === document ? [document] : [doc, document];
            const revealBtns = [];
            for (const d of searchDocs) {{
                for (const b of d.querySelectorAll('button')) {{
                    if (/extract|reveal|show|code/i.test(b.textContent)
                        && !/submit code/i.test(b.textContent)
                        && b.offsetParent !== null)
                        revealBtns.push(b);
                }}
            }}
            for (const btn of revealBtns) {{
                fullClick(btn);
                await new Promise(r => (window.__origST||setTimeout)(r, 400));
            }}
            // Check deepest level after reveal clicks (code may appear here)
            if (revealBtns.length > 0) {{
                const revCode = findCode(doc.body?.innerText);
                if (revCode) return surfaceCode(revCode);
                // Also check attributes at deepest level
                for (const el of doc.querySelectorAll('*')) {{
                    for (const attr of el.attributes) {{
                        if (/^[A-Z0-9]{{6}}$/.test(attr.value) && !fp.test(attr.value))
                            return surfaceCode(attr.value);
                    }}
                }}
            }}
            // Final code check across all iframe levels
            let checkDoc = document;
            for (let i = 0; i < maxDepth + 1; i++) {{
                const code = findCode(checkDoc.body?.innerText || checkDoc.innerText);
                if (code) return surfaceCode(code);
                const iframe = checkDoc.querySelector('iframe');
                if (iframe && iframe.contentDocument) checkDoc = iframe.contentDocument;
                else break;
            }}
            // Phase 4: React fiber onComplete fallback
            // When Extract Code button is bugged (off-by-one guard), bypass via React internals
            {{
                const startEl = revealBtns[0] || findBtn(document);
                if (startEl) {{
                    const fk = Object.keys(startEl).find(k => k.startsWith('__reactFiber'));
                    if (fk) {{
                        let f = startEl[fk];
                        for (let i = 0; i < 40 && f; i++, f = f.return) {{
                            const p = f.memoizedProps || f.pendingProps;
                            if (p && typeof p.onComplete === 'function' && p.stepNum !== undefined) {{
                                try {{
                                    const r = p.onComplete({{
                                        type: "recursive_iframe",
                                        timestamp: Date.now(),
                                        data: {{ method: "recursive_iframe", numLevels: maxDepth,
                                                 currentLevel: maxDepth, levelClickTimes: {{}},
                                                 stepNum: p.stepNum }}
                                    }});
                                    if (typeof r === 'string' && /^[A-Z0-9]{{6}}$/.test(r))
                                        return surfaceCode(r);
                                }} catch(e) {{}}
                                break;
                            }}
                        }}
                    }}
                }}
            }}
            return 'iframe: navigated ' + depth + ' levels';
        }})()"""
        actions.append(Action(type="js", value=iframe_js))
        actions.append(Action(type="wait", amount=2))

    # NOTE: "complete all N actions" (sequence challenges) are handled
    # by Agent._handle_sequence() with Playwright real events, not here.

    return actions


# Auto-click visible buttons matching common reveal/complete patterns.
_AUTO_CLICK_BUTTONS_JS = """
(() => {
    const re = /^(reveal|complete|show|finish|done|check|verify|capture|submit|solve|I remember|all.*visited|all.*reveal)/i;
    const containsRe = /reveal code|complete challenge|all tabs/i;
    for (const btn of document.querySelectorAll('button')) {
        const cs = window.getComputedStyle(btn);
        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
        const text = (btn.textContent || '').trim();
        if ((re.test(text) || containsRe.test(text)) && !/submit code/i.test(text)) {
            btn.click();
        }
    }
})()
"""

# Generic JS to reveal hidden 6-char codes — pure CSS manipulation, no challenge knowledge.
_REVEAL_HIDDEN_JS = """
(() => {
    const codeRe = /^[A-Z0-9]{6}$/;
    const fp = /^(SUBMIT|SCROLL|CLICKS?|REVEAL|BUTTON|HIDDEN|STEPBAR|STEP\\d+)$/;
    for (const el of document.querySelectorAll('*')) {
        const text = (el.textContent || '').trim();
        if (codeRe.test(text) && !fp.test(text) && el.childElementCount === 0) {
            el.style.cssText = 'display:block !important; visibility:visible !important; ' +
                'opacity:1 !important; color:red !important; font-size:16px !important; ' +
                'position:static !important; width:auto !important; height:auto !important;';
        }
    }
})()
"""


# Trigger React onChange on inputs after JS sets .value directly.
# Uses the native HTMLInputElement setter to bypass React's synthetic events.
_REACT_INPUT_TRIGGER_JS = """
(() => {
    const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
    ).set;
    if (!setter) return;
    for (const input of document.querySelectorAll('input')) {
        const val = input.value;
        if (!val || !val.trim()) continue;
        const ph = (input.placeholder || '').toLowerCase();
        if (ph.includes('code') || ph.includes('character')) continue;
        try {
            // Reset React _valueTracker so React detects the value change
            if (input._valueTracker) input._valueTracker.setValue('');
            setter.call(input, val);
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        } catch(e) {}
    }
})()
"""

# --- Puzzle solve JS ---
# Generic puzzle handler: selects one correct radio per group, solves math, clicks Solve.
# Made async so we can add delays between selections and the Solve click.
_PUZZLE_SOLVE_JS = """
(async () => {
    const correctRe = /\\b(correct|right choice|this is correct|correct answer|correct choice|select this|the answer|pick this|choose this|choose me|pick me|this one|best answer|^true$|^yes$|^right$)\\b/i;
    const wrongRe = /\\b(wrong|incorrect|not this|not the|false|^no$|wrong answer|don't pick|not correct)\\b/i;
    let radioGroupCount = 0;
    let btnGroupCount = 0;
    let inp = null;  // math answer input (declared here to avoid block-scope issues)

    // React-compatible checked setter
    const checkedSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'checked')?.set;

    // 1. Handle radio groups: click + programmatically set checked via React setter
    const radios = document.querySelectorAll('input[type="radio"]');
    const groups = {};
    for (const r of radios) {
        const g = r.name || 'default';
        if (!groups[g]) groups[g] = [];
        groups[g].push(r);
    }
    for (const inputs of Object.values(groups)) {
        let selected = false;
        for (const r of inputs) {
            const label = r.closest('label') || document.querySelector('label[for="' + r.id + '"]');
            const text = (label ? label.textContent : r.value || '').toLowerCase();
            if (correctRe.test(text)) {
                r.click();
                if (label) label.click();
                // Also set checked via React setter to ensure state update
                if (checkedSetter) {
                    checkedSetter.call(r, true);
                    r.dispatchEvent(new Event('change', { bubbles: true }));
                    r.dispatchEvent(new Event('input', { bubbles: true }));
                }
                selected = true;
                break;
            }
        }
        if (!selected && inputs.length > 0) {
            for (const r of inputs) {
                const label = r.closest('label') || document.querySelector('label[for="' + r.id + '"]');
                const text = (label ? label.textContent : '').toLowerCase();
                if (!wrongRe.test(text)) {
                    r.click();
                    if (label) label.click();
                    if (checkedSetter) {
                        checkedSetter.call(r, true);
                        r.dispatchEvent(new Event('change', { bubbles: true }));
                        r.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    break;
                }
            }
        }
        radioGroupCount++;
    }

    // 2. Button-style options: click ALL buttons matching correctRe.
    //    Don't group by parent (unreliable DOM structure); just click every
    //    correct-looking option and let React handle selection state.
    const optBtns = Array.from(document.querySelectorAll('button, [role="radio"], [role="option"]'))
        .filter(b => {
            const t = (b.textContent || '').trim();
            return t.length > 0 && t.length < 50
                && !/^(submit|solve|check|verify|submit code|click here|clear|capture|complete|reveal)$/i.test(t);
        });
    // Full pointer event dispatch for React compatibility
    function fullClick(el) {
        const r = el.getBoundingClientRect();
        const o = {bubbles:true, cancelable:true, view:window,
                   clientX:r.x+r.width/2, clientY:r.y+r.height/2};
        el.dispatchEvent(new PointerEvent('pointerdown', o));
        el.dispatchEvent(new MouseEvent('mousedown', o));
        el.dispatchEvent(new PointerEvent('pointerup', o));
        el.dispatchEvent(new MouseEvent('mouseup', o));
        el.dispatchEvent(new MouseEvent('click', o));
        el.click();
    }
    // Pass 1: tag + click all correct-looking buttons (skip if wrongRe also matches —
    //         e.g. "Not this one" matches correctRe via "this one" but is wrong)
    const correctClicked = new Set();
    for (const btn of optBtns) {
        const text = (btn.textContent || '').toLowerCase().trim();
        if (correctRe.test(text) && !wrongRe.test(text)) {
            btn.setAttribute('data-puzzle-select', 'true');
            fullClick(btn);
            correctClicked.add(btn.parentElement);
            btnGroupCount++;
        }
    }
    // Pass 2: for groups with no correct button clicked, pick the first non-wrong option.
    //         Group buttons by document order: if we know how many correct buttons we found,
    //         infer group size = total buttons / estimated groups, then check each group.
    //         All buttons often share one parent, so parentElement grouping doesn't work.
    const tagged = new Set(optBtns.filter(b => b.hasAttribute('data-puzzle-select')));
    if (btnGroupCount > 0 && btnGroupCount < optBtns.length) {
        const groupSize = Math.round(optBtns.length / Math.max(btnGroupCount, 1));
        if (groupSize >= 2) {
            for (let g = 0; g < optBtns.length; g += groupSize) {
                const group = optBtns.slice(g, g + groupSize);
                const hasCorrect = group.some(b => tagged.has(b));
                if (!hasCorrect) {
                    for (const btn of group) {
                        const text = (btn.textContent || '').toLowerCase().trim();
                        if (!wrongRe.test(text)) {
                            btn.setAttribute('data-puzzle-select', 'true');
                            fullClick(btn);
                            btnGroupCount++;
                            break;
                        }
                    }
                }
            }
        }
    }

    // 3. Solve math: "What is X + Y?" or "29 + 9 = ?"
    const body = document.body.innerText;
    const mathMatch = body.match(/(?:what is |solve[: ]+|compute[: ]+|calculate[: ]+)(\\d+)\\s*([+\\-*/×÷])\\s*(\\d+)/i)
        || body.match(/(\\d+)\\s*([+\\-*/×÷])\\s*(\\d+)\\s*=\\s*\\?/);
    if (mathMatch) {
        const a = parseInt(mathMatch[1]), b = parseInt(mathMatch[3]), op = mathMatch[2];
        let result = op === '+' ? a+b : op === '-' || op === '−' ? a-b : op === '*' || op === '×' ? a*b : Math.floor(a/b);
        // Find the math answer input — try specific types first, then any
        // text input that isn't the code submission field.
        inp = document.querySelector('input[type="number"], input[inputmode="numeric"]');
        if (!inp) {
            for (const candidate of document.querySelectorAll('input[type="text"], input:not([type])')) {
                const ph = (candidate.placeholder || '').toLowerCase();
                if (ph.includes('code') || ph.includes('character') || ph.includes('6-char')) continue;
                // Skip the code submission input (typically has "Enter" placeholder)
                if (ph.includes('enter') && ph.includes('code')) continue;
                inp = candidate;
                break;
            }
        }
        if (inp) {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            // Reset React _valueTracker so React detects the value change
            if (inp._valueTracker) inp._valueTracker.setValue('');
            setter.call(inp, String(result));
            inp.dispatchEvent(new Event('input', { bubbles: true }));
            inp.dispatchEvent(new Event('change', { bubbles: true }));
            inp.setAttribute('data-puzzle-math', 'true');
        }
    }

    // 4. Also click labels for the correct options (React may listen on labels)
    for (const label of document.querySelectorAll('label')) {
        const text = (label.textContent || '').toLowerCase().trim();
        if (correctRe.test(text)) fullClick(label);
    }

    // 5. Yield to microtask queue — React 18 schedules re-renders via
    //    queueMicrotask(), so this guarantees state is committed before Solve click
    await Promise.resolve();

    // 6. Click Solve/Submit buttons — use broad matching (not exact)
    //    to catch "Solve Puzzle", "Submit Answer", etc.
    //    Exclude "Submit Code" (the code submission button).
    function clickSolveBtns() {
        for (const btn of document.querySelectorAll('button')) {
            const t = btn.textContent.trim().toLowerCase();
            if (/^submit code$/i.test(t) || /^click here$/i.test(t)) continue;
            if (/solve|check|verify/i.test(t) || /^submit$/i.test(t)) {
                btn.setAttribute('data-puzzle-solve', 'true');
                fullClick(btn);
            }
        }
    }
    clickSolveBtns();

    // 7. Yield again and retry Solve (button may only enable after React processes)
    await Promise.resolve();
    clickSolveBtns();

    // 8. Focus math input and press Enter to submit form (fallback if no Solve button)
    if (inp) {
        inp.focus();
        inp.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', bubbles:true}));
        inp.dispatchEvent(new KeyboardEvent('keypress', {key:'Enter', code:'Enter', bubbles:true}));
        inp.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', code:'Enter', bubbles:true}));
    }

    // 9. Also try submitting any parent form (requestSubmit only — submit() does
    //    a native navigation that can 404 on SPA sites)
    const form = (inp || document.querySelector('input'))?.closest('form');
    if (form && form.requestSubmit) { form.requestSubmit(); }

    return 'puzzle: ' + radioGroupCount + ' radio, ' + btnGroupCount + ' btn groups, ' + (mathMatch ? 'math=' + mathMatch[0] : 'no math');
})()
"""

# --- Sequence sub-task JS fallbacks ---
# Used when indexed elements don't include click/hover/fill/scroll targets.

# Tag each sequence sub-task target element with data-seq-target attribute.
# Returns which targets were found. Python then uses Playwright locators
# on [data-seq-target="..."] for reliable interaction (no coordinate issues).
# Tag sequence sub-task targets WITHIN the sequence challenge container.
# Scoping prevents tagging noise popup "Click Here" buttons or the code
# submission input instead of the real sequence sub-task elements.
_SEQ_TAG_TARGETS_JS = """
(() => {
    // Clear old tags
    for (const el of document.querySelectorAll('[data-seq-target]'))
        el.removeAttribute('data-seq-target');

    // Find the sequence challenge container (has "complete all N actions")
    let scope = null;
    for (const el of document.querySelectorAll('*')) {
        const t = (el.innerText || '').toLowerCase();
        if (t.includes('complete all') && t.includes('action')
            && t.includes('click me')) {
            scope = el; break;
        }
    }
    if (!scope) {
        // Fallback: look for container with "sequence" + "complete"
        for (const el of document.querySelectorAll('*')) {
            const t = (el.innerText || '').toLowerCase();
            if (t.includes('sequence') && t.includes('complete')
                && t.length < 2000) {
                scope = el; break;
            }
        }
    }
    if (!scope) return {};
    const found = {};

    // Click Me target — prefer the deepest (smallest) element to avoid
    // tagging a wrapper div when the actual handler is on a child button.
    // Two-pass: buttons first, then other elements.
    let clickTarget = null;
    for (const el of scope.querySelectorAll('button')) {
        const text = (el.textContent || '').trim().toLowerCase();
        if (text.startsWith('click me') && el.offsetParent !== null) {
            clickTarget = el; break;
        }
    }
    if (!clickTarget) {
        // Find the SMALLEST element with "click me" text
        let bestArea = Infinity;
        for (const el of scope.querySelectorAll('div, span, p')) {
            const text = (el.textContent || '').trim().toLowerCase();
            if (!text.startsWith('click me') || el.offsetParent === null) continue;
            const r = el.getBoundingClientRect();
            const area = r.width * r.height;
            if (r.width > 5 && r.height > 5 && area < bestArea) {
                clickTarget = el; bestArea = area;
            }
        }
    }
    if (clickTarget) {
        clickTarget.setAttribute('data-seq-target', 'click');
        found.click = true;
    }
    // Hover target (smallest inside container)
    let bestHover = null, bestArea = Infinity;
    for (const el of scope.querySelectorAll('div, span, section, p')) {
        const t = (el.textContent || '').toLowerCase();
        if ((t.includes('hover here') || t.includes('hover over'))
            && el.children.length < 5 && el.offsetParent !== null) {
            const r = el.getBoundingClientRect();
            const area = r.width * r.height;
            if (r.width > 10 && r.height > 10 && area < bestArea) {
                bestHover = el; bestArea = area;
            }
        }
    }
    if (bestHover) {
        bestHover.setAttribute('data-seq-target', 'hover');
        found.hover = true;
    }
    // Fillable input (inside container, not the code submission input)
    for (const inp of scope.querySelectorAll('input:not([type="hidden"])')) {
        const ph = (inp.placeholder || '').toLowerCase();
        if (ph && !ph.includes('code') && !ph.includes('character')
            && !ph.includes('enter') && !ph.startsWith('6-')) {
            inp.setAttribute('data-seq-target', 'fill');
            found.fill = true; break;
        }
    }
    // Scrollable container (inside challenge, not noise modals)
    for (const el of scope.querySelectorAll('div, section')) {
        if (el.scrollHeight > el.clientHeight + 20) {
            const cs = getComputedStyle(el);
            if ((cs.overflow + ' ' + cs.overflowY).match(/auto|scroll/)) {
                const t = (el.textContent || '').toLowerCase();
                if (t.includes('modal')) continue;
                if (t.includes('scroll') || t.includes('keep scrolling')
                    || el.scrollHeight > el.clientHeight + 50) {
                    el.setAttribute('data-seq-target', 'scroll');
                    found.scroll = true; break;
                }
            }
        }
    }
    return found;
})()
"""


# Popup modal button names to filter from element list
_MODAL_BUTTON_RE = re.compile(
    r"^(close|close \(fake\)|dismiss|accept|decline|got it|"
    r"no thanks|maybe later|not now|ok|okay|submit & continue)$",
    re.IGNORECASE,
)

# Quiz/option content patterns (noise widgets on the page)
_QUIZ_CONTENT_RE = re.compile(
    r"(correct choice|wrong option|incorrect choice|the right choice|"
    r"correct answer|not this one|pick this|choose me|select this one|"
    r"option [A-D]\b)",
    re.IGNORECASE,
)

# Floating noise divs with popup bait text
_NOISE_DIV_RE = re.compile(
    r"^(click me!?|click here!?|here!?|try this!?|moving!?|button!?|link!?)$",
    re.IGNORECASE,
)


def _filter_noise(
    elements: list[ElementInfo], instruction: str = "",
) -> list[ElementInfo]:
    """Filter noise elements from the indexed list.

    Removes popup modal buttons (Close, Dismiss, Accept, etc.) and
    quiz/option widgets. Keeps challenge elements, submit form, and
    anything with scrollable/clickable attributes.

    In "puzzle mode" (instruction mentions puzzle/solve), relaxes quiz
    filtering since the challenge itself may be a quiz.
    """
    inst_lower = instruction.lower()
    puzzle_mode = any(k in inst_lower for k in ("puzzle", "quiz", "solve", "select an option"))
    sequence_mode = any(k in inst_lower for k in ("sequence", "complete all"))

    kept: list[ElementInfo] = []
    for el in elements:
        name = el.name.strip()

        # Keep inputs, canvas, textareas, selects unconditionally
        if el.tag in ("input", "canvas", "textarea", "select"):
            kept.append(el)
            continue

        # Keep elements with scrollable/clickable extra attributes
        if "scrollable" in el.extra or "clickable" in el.extra:
            # Filter quiz option divs even if clickable (unless puzzle mode)
            if not puzzle_mode and _QUIZ_CONTENT_RE.search(name):
                continue
            # Filter floating noise divs even if clickable (unless sequence mode)
            if not sequence_mode and el.tag in ("div", "span") and _NOISE_DIV_RE.match(name):
                continue
            kept.append(el)
            continue

        # Filter modal buttons
        if el.tag == "button" and _MODAL_BUTTON_RE.match(name):
            continue

        # Filter quiz/option elements (unless puzzle mode)
        if not puzzle_mode and _QUIZ_CONTENT_RE.search(name):
            if el.tag in ("div", "label", "button"):
                continue

        # Filter unnamed buttons that are likely popup close (X) icons
        if el.tag == "button" and not name:
            continue

        # Filter floating noise divs/spans with popup bait text (unless sequence mode)
        if not sequence_mode and el.tag in ("div", "span") and _NOISE_DIV_RE.match(name):
            continue

        kept.append(el)
    return kept


class Agent:
    """General-purpose browser automation agent.

    Uses LLM reasoning for every step — no regex dispatch, no hardcoded
    patterns. Works via element indexing: the LLM picks from a numbered
    list of page elements, eliminating selector hallucination.
    """

    def __init__(
        self,
        headless: bool = True,
        config: ChallengeConfig | None = None,
    ) -> None:
        self.config = config or ChallengeConfig()
        self.browser = BrowserController(headless=headless)
        self.solver = Solver()
        self.state = StateTracker(
            total_steps=self.config.total_steps,
            max_time=self.config.max_time,
            max_step_time=self.config.max_step_time,
        )
        self.metrics = MetricsCollector()
        self._used_codes: set[str] = set()

    async def run(self) -> MetricsCollector:
        """Execute the full challenge. Returns metrics."""
        async with self.browser:
            # Navigate to landing page and click START
            await self.browser.goto(self.config.url)
            await asyncio.sleep(0.5)
            try:
                start_btn = self.browser.page.locator("text=START").first
                await start_btn.click(timeout=5000)
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[agent] Could not click START: {e}")

            # Main loop
            max_attempts = self.config.max_retries
            last_step = 0
            attempts_on_step = 0
            step_history: list[str] = []  # Track what was tried per step
            step_start_time = 0.0  # Track total time on current step
            abandoned_steps: set[int] = set()  # Steps we gave up on
            max_step_total = 22.0  # Max seconds on a single step

            while not self.state.is_over_budget():
                current_step = await self.state.detect_step_from_url(self.browser.page)
                if current_step > self.state.total_steps:
                    print("[agent] All steps completed!")
                    break

                if current_step == 0:
                    await asyncio.sleep(0.5)
                    continue

                # Skip steps we already abandoned
                if current_step in abandoned_steps:
                    break

                if current_step != last_step:
                    last_step = current_step
                    attempts_on_step = 0
                    step_history = []
                    step_start_time = time.time()

                # Time-based skip: if we've spent too long on this step, skip
                if time.time() - step_start_time > max_step_total and attempts_on_step > 0:
                    print(f"[agent] Step {current_step} time limit ({max_step_total:.0f}s) — skipping")
                    skipped = False
                    # Try skipping to step+1, step+2, step+3 (in case some steps 404)
                    for skip_offset in range(1, 4):
                        skipped = await self._try_skip_step(
                            current_step + skip_offset - 1
                        )
                        if skipped:
                            last_step = current_step + skip_offset
                            attempts_on_step = 0
                            step_history = []
                            step_start_time = time.time()
                            break
                    if skipped:
                        continue
                    # All skip offsets failed — abandon this step permanently
                    print(f"[agent] All skip offsets failed — abandoning step {current_step}")
                    abandoned_steps.add(current_step)
                    last_step = current_step + 1
                    attempts_on_step = 0
                    step_history = []
                    step_start_time = time.time()
                    continue

                try:
                    success = await asyncio.wait_for(
                        self._solve_step(
                            current_step, attempts_on_step, step_history
                        ),
                        timeout=11.0,
                    )
                except asyncio.TimeoutError:
                    print(f"[agent] Step {current_step} attempt timed out (11s)")
                    step_history.append(
                        f"attempt {attempts_on_step + 1}: TIMED OUT after 11s — "
                        "try a faster/different approach"
                    )
                    self.metrics.end_step(False)
                    success = False
                if success:
                    attempts_on_step = 0
                    step_history = []
                    step_start_time = time.time()
                    await self._wait_for_content()
                else:
                    attempts_on_step += 1
                    if attempts_on_step >= max_attempts:
                        print(f"[agent] Step {current_step} stuck after {max_attempts} attempts")
                        # Try to skip to next step via SPA navigation
                        skipped = False
                        for skip_offset in range(1, 4):
                            skipped = await self._try_skip_step(
                                current_step + skip_offset - 1
                            )
                            if skipped:
                                last_step = current_step + skip_offset
                                attempts_on_step = 0
                                step_history = []
                                step_start_time = time.time()
                                break
                        if skipped:
                            continue
                        # All skip offsets failed — abandon permanently
                        print(f"[agent] All skip offsets failed — abandoning step {current_step}")
                        abandoned_steps.add(current_step)
                        last_step = current_step + 1
                        attempts_on_step = 0
                        step_history = []
                        step_start_time = time.time()
                        continue

        self.metrics.print_report()
        return self.metrics

    async def _solve_step(
        self, step: int, attempt: int, history: list[str]
    ) -> bool:
        """Solve a single step using LLM reasoning. Returns True if completed."""
        self.state.begin_step(step)
        self.metrics.begin_step(step)
        t0 = time.time()
        print(f"\n[agent] === Step {step} (attempt {attempt + 1}) ===")

        try:
            # 0. Brief pause for React to render challenge content,
            #    then reset stale cleaner CSS and clean fixed overlays.
            if attempt == 0:
                await asyncio.sleep(0.15)
            await quick_clean(self.browser.page)

            # 1. Perceive — a11y snapshot + element indexing
            page_state = await snapshot(self.browser.page)

            # Retry perception if page hasn't fully loaded.
            # Most challenges have 5+ elements (input, submit, challenge content).
            if len(page_state.elements) < 5:
                await asyncio.sleep(0.5)
                await quick_clean(self.browser.page)
                page_state = await snapshot(self.browser.page)
                # Second retry with even longer wait
                if len(page_state.elements) < 5:
                    await asyncio.sleep(0.5)
                    page_state = await snapshot(self.browser.page)

            # Keep full element list for execution (executor resolves by
            # list index). Filter noise for the LLM prompt only.
            all_elements = page_state.elements
            filtered = _filter_noise(all_elements, page_state.instruction)

            # Complex challenges (sequence, puzzle, etc.) may need more
            # render time — React components mount asynchronously.
            # If elements are sparse despite a complex instruction, wait and re-perceive.
            _COMPLEX_RE = re.compile(
                r"(sequence|complete all \d|puzzle|service worker|split parts|drag|shadow dom|keyboard sequence)",
                re.IGNORECASE,
            )
            if _COMPLEX_RE.search(page_state.instruction) and len(filtered) < 6:
                print(f"[agent] Complex challenge with sparse elements ({len(filtered)}) — waiting for render")
                await asyncio.sleep(0.5)
                await quick_clean(self.browser.page)
                page_state = await snapshot(self.browser.page)
                all_elements = page_state.elements
                filtered = _filter_noise(all_elements, page_state.instruction)

            # Swap filtered list into page_state for prompt rendering,
            # but use all_elements for execution.
            page_state.elements = filtered
            print(f"[agent] Instruction: {page_state.instruction[:120]}")
            print(f"[agent] Elements: {len(filtered)} shown ({len(all_elements)} total)")
            for el in filtered[:30]:
                print(f"  [{el.index}] {el.tag} \"{el.name[:50]}\" {el.extra}")

            # 3. If code already visible, submit directly
            if page_state.visible_codes:
                code = next(
                    (c for c in page_state.visible_codes if c not in self._used_codes),
                    None,
                )
                if code:
                    print(f"[agent] Code already visible: {code}")
                    if await self._submit_and_check(code, step):
                        return True

            # 3b. Pre-action: execute obvious instruction directives on first attempt
            pre_acted = False
            pre_actions = []
            if attempt == 0:
                pre_actions = _parse_instruction_actions(
                    page_state.instruction, page_state.elements,
                )
                if pre_actions:
                    pre_acted = True
                    desc = ", ".join(
                        f"{a.type}({a.element})" if a.element is not None
                        else f"{a.type}({a.amount})"
                        for a in pre_actions
                    )
                    print(f"[agent] Pre-action from instruction: [{desc}]")
                    # Use longer delays for multi-action sequences
                    is_sequence = len(pre_actions) >= 3
                    inter_delay = 0.25 if is_sequence else 0.15
                    for i, a in enumerate(pre_actions):
                        # Aggressive overlay cleanup before hover to ensure
                        # CSS :hover activates on the target, not an overlay.
                        if a.type == "hover":
                            await clean_page(self.browser.page)
                            await self.browser.page.evaluate("""
                                document.querySelectorAll('*').forEach(el => {
                                    const s = getComputedStyle(el);
                                    const z = parseInt(s.zIndex);
                                    if (isNaN(z) || z < 10) return;
                                    if (s.position !== 'fixed' && s.position !== 'absolute') return;
                                    el.style.setProperty('pointer-events', 'none', 'important');
                                });
                            """)
                            await asyncio.sleep(0.05)
                        await execute_action(
                            self.browser.page, a, all_elements,
                        )
                        # After fill/js, trigger React onChange
                        if a.type in ("fill", "js"):
                            try:
                                await self.browser.page.evaluate(
                                    _REACT_INPUT_TRIGGER_JS
                                )
                            except Exception:
                                pass
                        # Delay between pre-actions for React state updates
                        if i < len(pre_actions) - 1:
                            await asyncio.sleep(inter_delay)
                    # Early extraction: check for codes immediately after
                    # pre-actions complete (before auto-click buttons which may
                    # re-hide revealed content).
                    has_hover = any(a.type == "hover" for a in pre_actions)
                    has_wait = any(a.type == "wait" for a in pre_actions)
                    if not has_hover:
                        # After wait pre-actions, new overlays may have spawned
                        # during the delay — re-clean before extraction.
                        if has_wait:
                            await asyncio.sleep(0.25)
                            await quick_clean(self.browser.page)
                        code = await find_code(self.browser.page, self._used_codes)
                        if code:
                            if await self._submit_and_check(code, step):
                                return True
                    if has_hover:
                        code = await find_code(self.browser.page, self._used_codes)
                        if code:
                            if await self._submit_and_check(code, step):
                                return True
                        # If no code yet and page says "keep hovering", re-hover
                        # (some challenges need >5s total hover time).
                        # Use textContent (not innerText) to include hidden elements.
                        try:
                            body_text = await self.browser.page.evaluate(
                                "document.body.textContent || ''"
                            )
                        except Exception:
                            body_text = ""
                        if "keep hovering" in body_text.lower() or "still hovering" in body_text.lower():
                            print("[agent] Page says keep hovering — re-hovering")
                            hover_action = next(
                                a for a in pre_actions if a.type == "hover"
                            )
                            await execute_action(
                                self.browser.page, hover_action, all_elements,
                            )
                            code = await find_code(self.browser.page, self._used_codes)
                            if code:
                                if await self._submit_and_check(code, step):
                                    return True

                    # Auto-click reveal/complete buttons after pre-actions
                    await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
                    # Puzzle: use Playwright trusted clicks as backup
                    # (JS .click() / fullClick() may not trigger React handlers)
                    has_puzzle = "puzzle" in page_state.instruction.lower()
                    if has_puzzle:
                        # Re-click tagged option buttons with trusted events
                        try:
                            tagged = self.browser.page.locator(
                                "[data-puzzle-select='true']"
                            )
                            count = await tagged.count()
                            for i in range(count):
                                try:
                                    await tagged.nth(i).click(
                                        timeout=500, force=True
                                    )
                                    await asyncio.sleep(0.1)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # 3b: Re-click checked radio buttons with trusted events
                        # (JS .click() may not register with React)
                        try:
                            radios = self.browser.page.locator(
                                "input[type='radio']:checked"
                            )
                            radio_count = await radios.count()
                            for i in range(radio_count):
                                try:
                                    await radios.nth(i).click(
                                        timeout=800, force=True
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # 3c: Re-fill math input with Playwright trusted fill
                        # JS tags the correct input with data-puzzle-math
                        try:
                            math_result = await self.browser.page.evaluate("""
                                (() => {
                                    const body = document.body.innerText;
                                    const m = body.match(/(?:what is |solve[: ]+|compute[: ]+|calculate[: ]+)(\\d+)\\s*([+\\-*/×÷])\\s*(\\d+)/i)
                                        || body.match(/(\\d+)\\s*([+\\-*/×÷])\\s*(\\d+)\\s*=\\s*\\?/);
                                    if (!m) return null;
                                    const a = parseInt(m[1]), b = parseInt(m[3]), op = m[2];
                                    return String(op === '+' ? a+b : op === '-' || op === '\\u2212' ? a-b : op === '*' || op === '\\u00d7' ? a*b : Math.floor(a/b));
                                })()
                            """)
                            if math_result:
                                try:
                                    loc = self.browser.page.locator(
                                        "[data-puzzle-math]"
                                    ).first
                                    if await loc.is_visible(timeout=500):
                                        await loc.fill(
                                            math_result, timeout=1000
                                        )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        await asyncio.sleep(0.15)
                        # Click Solve button with trusted events
                        # JS tags the correct button with data-puzzle-solve
                        try:
                            loc = self.browser.page.locator(
                                "[data-puzzle-solve='true']"
                            ).first
                            if await loc.is_visible(timeout=500):
                                await loc.click(timeout=1000)
                                await asyncio.sleep(0.15)
                        except Exception:
                            pass
                    # Poll for code — use more rounds for delayed reveals
                    has_wait = any(a.type == "wait" for a in pre_actions)
                    poll_rounds = 8 if (has_wait or has_puzzle) else 4
                    for _ in range(poll_rounds):
                        await asyncio.sleep(0.15)
                        # Skip reset_cleaner for hover challenges — restoring
                        # noise overlays kills CSS :hover and hides the code.
                        if not has_hover:
                            await reset_cleaner(self.browser.page)
                        await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
                        code = await find_code(self.browser.page, self._used_codes)
                        if code:
                            if await self._submit_and_check(code, step):
                                return True
                            break

            # 3c. Sequence challenges: Playwright real mouse events
            #      JS-dispatched events have isTrusted=false; React may ignore them.
            #      Use Playwright for trusted interactions.
            if (
                re.search(r"complete (?:all )?\d+ actions?", page_state.instruction.lower())
                and attempt <= 1
            ):
                print("[agent] Sequence challenge detected — using Playwright real events")
                await self._handle_sequence(all_elements)
                pre_acted = True
                # Auto-click reveal/complete buttons
                await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
                # Poll for code
                for _ in range(4):
                    await asyncio.sleep(0.15)
                    await reset_cleaner(self.browser.page)
                    await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
                    code = await find_code(self.browser.page, self._used_codes)
                    if code:
                        if await self._submit_and_check(code, step):
                            return True
                        break

            # 4. Reason — LLM decides actions
            #    If pre-actions ran, add them to history so LLM doesn't repeat.
            if pre_acted:
                pre_desc = ", ".join(a.type for a in pre_actions)
                history = history + [f"Pre-actions already executed: {pre_desc}. Do NOT repeat these."]
            if attempt >= 1:
                print("[agent] Retry — calling recovery LLM with screenshot")
                screenshot = await self.browser.screenshot()
                pre_tokens = self.solver.total_tokens
                actions = await self.solver.solve_stuck(
                    page_state, screenshot, history, temperature=0.3
                )
                self.metrics.record_llm_call(2, self.solver.total_tokens - pre_tokens)
            else:
                pre_tokens = self.solver.total_tokens
                actions = await self.solver.solve(page_state, history, temperature=0.0)
                self.metrics.record_llm_call(1, self.solver.total_tokens - pre_tokens)

            action_desc = ", ".join(
                f"{a.type}({a.element})" if a.element is not None else a.type
                for a in actions
            )
            print(f"[agent] LLM returned {len(actions)} actions: [{action_desc}]")

            # Filter out LLM actions that exactly duplicate pre-actions
            # (same type AND same element). Keeps actions on different elements.
            if pre_acted and actions:
                pre_pairs = {(a.type, a.element) for a in pre_actions}
                before = len(actions)
                actions = [a for a in actions if (a.type, a.element) not in pre_pairs]
                removed = before - len(actions)
                if removed:
                    print(f"[agent] Filtered {removed} duplicate actions from LLM")

            if not actions:
                if pre_acted:
                    # Pre-actions already ran, LLM returned nothing extra.
                    # Just try revealing hidden codes and go to polling.
                    actions = [Action(type="js", value=_REVEAL_HIDDEN_JS)]
                else:
                    # No pre-actions, LLM returned nothing — try instruction parse
                    fallback = _parse_instruction_actions(
                        page_state.instruction, page_state.elements,
                    )
                    if fallback:
                        actions = fallback
                    else:
                        actions = [Action(type="js", value=_REVEAL_HIDDEN_JS)]

            # 5. Execute actions (with brief delays for React state updates)
            for i, action in enumerate(actions):
                el_desc = ""
                if action.element is not None and action.element < len(all_elements):
                    ei = all_elements[action.element]
                    name = str(ei.name or "")[:40]
                    el_desc = f"element={action.element} ({ei.tag} \"{name}\")"
                elif action.element is not None:
                    el_desc = f"element={action.element}"
                else:
                    val = str(action.value or "")[:60] if action.value else str(action.amount)
                    el_desc = val
                print(f"[agent]   -> {action.type} {el_desc}")
                await execute_action(self.browser.page, action, all_elements)
                # After JS or fill actions that set input values, trigger React
                # onChange immediately (before next action clicks Submit/Solve)
                if action.type in ("js", "fill"):
                    try:
                        await self.browser.page.evaluate(_REACT_INPUT_TRIGGER_JS)
                    except Exception:
                        pass
                # Brief delay between interactive actions for React state updates
                if i < len(actions) - 1 and action.type in (
                    "click", "draw_strokes", "hover", "fill", "scroll", "drag", "js",
                ):
                    await asyncio.sleep(0.1)

            # 6. Brief pause for page reaction
            has_interaction = any(a.type in ("click", "hover", "drag", "scroll", "draw_strokes") for a in actions)
            llm_has_hover = any(a.type == "hover" for a in actions)
            await asyncio.sleep(0.25 if has_interaction else 0.1)

            # 6a. Hover-sensitive: extract code BEFORE reset_cleaner
            # (noise overlays returning kills CSS :hover → code vanishes)
            if llm_has_hover:
                code = await find_code(self.browser.page, self._used_codes)
                if code:
                    t1 = time.time()
                    print(f"[timing] total={t1-t0:.1f}s")
                    history.append(
                        f"attempt {attempt + 1}: actions=[{action_desc}], "
                        f"result={'code ' + code}"
                    )
                    self.metrics.end_step(True)
                    return await self._submit_and_check(code, step)

            # 6b. Reset cleaner BEFORE auto-click — reveal buttons may be
            #     inside clean_page-hidden containers (position:fixed z>500).
            if not llm_has_hover:
                await reset_cleaner(self.browser.page)
                await asyncio.sleep(0.05)

            # 6c. Auto-click any reveal/complete buttons that appeared after actions
            await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
            await asyncio.sleep(0.15)

            # 7. Extract code
            code = await find_code(self.browser.page, self._used_codes)

            # 8. If no code, try revealing hidden codes (CSS manipulation)
            if code is None:
                await self.browser.page.evaluate(_REVEAL_HIDDEN_JS)
                await asyncio.sleep(0.1)
                code = await find_code(self.browser.page, self._used_codes)

            # 10. Brief poll (some challenges reveal codes with delay)
            #     Each round: reset cleaner → auto-click → extract.
            if code is None and not self.state.is_step_timed_out():
                for poll_round in range(3):
                    await asyncio.sleep(0.15)
                    if not llm_has_hover:
                        await reset_cleaner(self.browser.page)
                        await asyncio.sleep(0.05)
                    await self.browser.page.evaluate(_AUTO_CLICK_BUTTONS_JS)
                    await asyncio.sleep(0.08)
                    code = await find_code(self.browser.page, self._used_codes)
                    if code:
                        break

            t1 = time.time()
            print(f"[timing] total={t1-t0:.1f}s")

            # Record what happened for retry context
            history.append(
                f"attempt {attempt + 1}: actions=[{action_desc}], "
                f"result={'code ' + code if code else 'no code found'}"
            )

            # 10. Submit code
            if code:
                if await self._submit_and_check(code, step):
                    return True
                history[-1] += f" (rejected)"
                self.state.mark_failed()
                self.metrics.end_step(False, error=f"code {code} rejected")
                print(f"[agent] Step {step} FAILED — code {code} rejected")
                return False

            # Failed — no code found
            self.state.mark_failed()
            self.metrics.end_step(False, error="no code found")
            print(f"[agent] Step {step} FAILED — no code found")
            return False

        except Exception as e:
            self.state.mark_failed()
            self.metrics.end_step(False, error=str(e))
            print(f"[agent] Step {step} ERROR: {e}")
            return False

    async def _handle_sequence(self, all_elements: list[ElementInfo]) -> None:
        """Handle sequence challenge with Playwright locators + real events.

        JS tags each target with data-seq-target="click|hover|fill|scroll",
        then we use Playwright locators (which auto-scroll and click the
        actual element) instead of raw viewport coordinates.  This avoids
        the coordinate-collapse bug and overlay-interception issues.
        """
        page = self.browser.page

        # 1. Two-phase overlay removal:
        #    a) clean_page: display:none on position:fixed z>500 noise.
        #       This removes overlays from layout/hit-testing entirely,
        #       which is critical for hover (CSS :hover won't activate if
        #       an overlay intercepts the mouse).
        #    b) pointer-events:none on any remaining high-z overlays that
        #       aren't fixed (position:absolute, etc). This lets click
        #       events pass through to targets below.
        await clean_page(page)
        await asyncio.sleep(0.05)
        # Phase b: catch remaining overlays not handled by clean_page
        await page.evaluate("""
            document.querySelectorAll('*').forEach(el => {
                if (el.style.display === 'none') return;
                const s = getComputedStyle(el);
                const z = parseInt(s.zIndex);
                if (!isNaN(z) && z > 500
                    && (s.position === 'fixed' || s.position === 'absolute')) {
                    el.style.setProperty('pointer-events', 'none', 'important');
                }
            });
        """)
        await asyncio.sleep(0.05)

        # 2. Tag targets WITHIN the sequence challenge container
        #    (scoped search prevents tagging noise popup buttons)
        found = await page.evaluate(_SEQ_TAG_TARGETS_JS)
        print(f"[agent] Sequence: tagged targets: {found}")

        done = 0

        # Each action uses BOTH Playwright (isTrusted=true) AND JS fallback
        # (dispatches directly on the element, bypassing z-order interception).

        # 1. Click — full pointer event sequence (mousedown → mouseup → click)
        #    React may listen on mousedown/pointerdown, not just click.
        if found.get("click"):
            try:
                loc = page.locator("[data-seq-target='click']").first
                await loc.scroll_into_view_if_needed(timeout=3000)
                await loc.click(timeout=3000, force=True)
                # JS fallback: full pointer event sequence
                await page.evaluate("""
                    (() => {
                        const el = document.querySelector('[data-seq-target="click"]');
                        if (!el) return;
                        const r = el.getBoundingClientRect();
                        const o = {bubbles:true, cancelable:true,
                                   clientX:r.x+r.width/2, clientY:r.y+r.height/2};
                        el.dispatchEvent(new PointerEvent('pointerdown', o));
                        el.dispatchEvent(new MouseEvent('mousedown', o));
                        el.dispatchEvent(new PointerEvent('pointerup', o));
                        el.dispatchEvent(new MouseEvent('mouseup', o));
                        el.dispatchEvent(new MouseEvent('click', o));
                    })()
                """)
                done += 1
                print("[agent] Seq: clicked target")
            except Exception as e:
                print(f"[agent] Seq: click failed: {e}")
        await asyncio.sleep(0.25)

        # 2. Hover — locator.hover(force=True) + sustained micro-movements.
        #    Noise is display:none so nothing intercepts the mouse — CSS :hover
        #    activates directly on the target.  Matches the executor's approach.
        if found.get("hover"):
            try:
                loc = page.locator("[data-seq-target='hover']").first
                await loc.scroll_into_view_if_needed(timeout=3000)
                await loc.hover(force=True, timeout=3000)
                # JS fallback: dispatch mouseenter/mouseover directly on the
                # element.  React uses event delegation that may not be ready
                # on the first page render — Playwright trusted events fire
                # but React misses them.  JS dispatchEvent is synchronous and
                # ensures React's mouseover handler runs.  Zero wall-clock
                # overhead (page.evaluate is instant vs the 2.4s hover).
                await page.evaluate("""
                    const el = document.querySelector('[data-seq-target="hover"]');
                    if (el) {
                        const r = el.getBoundingClientRect();
                        const o = {bubbles:true, cancelable:true, view:window,
                                   clientX:r.x+r.width/2, clientY:r.y+r.height/2};
                        el.dispatchEvent(new PointerEvent('pointerover', o));
                        el.dispatchEvent(new PointerEvent('pointerenter',
                            {...o, bubbles:false}));
                        el.dispatchEvent(new MouseEvent('mouseover', o));
                        el.dispatchEvent(new MouseEvent('mouseenter',
                            {...o, bubbles:false}));
                    }
                """)
                box = await asyncio.wait_for(loc.bounding_box(), timeout=3.0)
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    for i in range(5):  # ~1.0s sustained hover with micro-movements
                        await asyncio.sleep(0.2)
                        await page.mouse.move(cx + (i % 3) - 1, cy)
                    # Move away to trigger mouseleave (some React handlers
                    # check hover duration on onMouseLeave)
                    await page.mouse.move(0, 0)
                    done += 1
                    print(f"[agent] Seq: hovered ({cx:.0f}, {cy:.0f}) 2.4s + mouseleave")
                else:
                    await asyncio.sleep(1.0)
                    done += 1
                    print("[agent] Seq: hovered (no bbox, waited 1s)")
            except Exception as e:
                print(f"[agent] Seq: hover failed: {e}")
        await asyncio.sleep(0.15)

        # 3. Fill — Playwright fill() atomically clears + sets + fires events.
        #    Fallback to keyboard type if fill() fails.
        if found.get("fill"):
            try:
                loc = page.locator("[data-seq-target='fill']").first
                await loc.scroll_into_view_if_needed(timeout=3000)
                await loc.fill("hello", timeout=3000)
                await page.evaluate(_REACT_INPUT_TRIGGER_JS)
                done += 1
                print("[agent] Seq: filled 'hello' into input")
            except Exception:
                # Fallback: keyboard type
                try:
                    await page.evaluate(
                        "document.querySelector('[data-seq-target=\"fill\"]')?.focus()"
                    )
                    await asyncio.sleep(0.05)
                    await page.keyboard.press("Meta+a")
                    await page.keyboard.type("hello")
                    await page.evaluate(_REACT_INPUT_TRIGGER_JS)
                    done += 1
                    print("[agent] Seq: typed 'hello' into input (keyboard fallback)")
                except Exception as e2:
                    print(f"[agent] Seq: fill failed: {e2}")
        await asyncio.sleep(0.15)

        # 4. Scroll — Playwright wheel + JS scrollTop fallback
        if found.get("scroll"):
            try:
                loc = page.locator("[data-seq-target='scroll']").first
                await loc.scroll_into_view_if_needed(timeout=3000)
                box = await asyncio.wait_for(loc.bounding_box(), timeout=3.0)
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    await page.mouse.move(cx, cy)
                    await asyncio.sleep(0.03)
                    for _ in range(15):
                        await page.mouse.wheel(0, 200)
                        await asyncio.sleep(0.05)
                    # JS fallback: scroll to bottom + dispatch scroll event
                    await page.evaluate("""
                        const el = document.querySelector('[data-seq-target="scroll"]');
                        if (el) {
                            el.scrollTop = el.scrollHeight;
                            el.dispatchEvent(new Event('scroll', {bubbles: true}));
                        }
                    """)
                    done += 1
                    print(f"[agent] Seq: scrolled ({cx:.0f}, {cy:.0f})")
            except Exception as e:
                print(f"[agent] Seq: scroll failed: {e}")

        # 5. Try clicking Complete button and report its state
        complete_info = await page.evaluate("""
            (() => {
                const btn = Array.from(document.querySelectorAll('button'))
                    .find(b => /complete/i.test(b.textContent) && b.offsetParent !== null);
                if (btn) {
                    const text = btn.textContent.trim();
                    btn.click();
                    return 'clicked: ' + text;
                }
                return 'no Complete button found';
            })()
        """)
        print(f"[agent] Seq: completed {done}/4 actions, Complete btn: {complete_info}")

    async def _submit_and_check(self, code: str, step: int) -> bool:
        """Submit a code and check if the step advanced."""
        print(f"[agent] Submitting code: {code}")
        await clean_page(self.browser.page)
        url_before = self.browser.page.url
        submitted = await submit_code(self.browser.page, code)
        if submitted:
            try:
                await self.browser.page.wait_for_url(
                    lambda url: url != url_before, timeout=3000
                )
            except Exception:
                pass
            advanced = await self.state.check_advancement(self.browser.page, step)
            if advanced:
                self.state.mark_completed()
                self.metrics.end_step(True, code=code)
                self._used_codes.add(code)
                print(f"[agent] Step {step} PASSED")
                return True
            for _ in range(3):
                await asyncio.sleep(0.1)
                if await self.state.check_advancement(self.browser.page, step):
                    self.state.mark_completed()
                    self.metrics.end_step(True, code=code)
                    self._used_codes.add(code)
                    print(f"[agent] Step {step} PASSED")
                    return True
            # Rejected code — add to used so we don't resubmit the same wrong code
            self._used_codes.add(code)
            print(f"[agent] Step {step} submitted {code} but did not advance")
        return False

    async def _try_skip_step(self, current_step: int) -> bool:
        """Attempt to skip a stuck step via SPA navigation.

        Uses pushState + popstate event to trigger React Router navigation
        without a full page reload (which would 404 on Netlify SPA).
        """
        next_step = current_step + 1
        if next_step > self.state.total_steps:
            return False
        try:
            await self.browser.page.evaluate(f"""
                window.history.pushState({{}}, '', '/step{next_step}');
                window.dispatchEvent(new PopStateEvent('popstate'));
            """)
            await asyncio.sleep(0.5)
            actual = await self.state.detect_step_from_url(self.browser.page)
            if actual != next_step:
                print(f"[agent] Skip failed — URL shows step {actual}")
                return False
            # Verify the page actually rendered challenge content (not a 404)
            try:
                body_text = await self.browser.page.inner_text("body")
            except Exception:
                body_text = ""
            if "not found" in body_text.lower() or "broken link" in body_text.lower():
                print(f"[agent] Skip failed — got 404 page, navigating back")
                await self.browser.page.evaluate("window.history.back()")
                await asyncio.sleep(0.5)
                return False
            print(f"[agent] Skipped step {current_step} → step {next_step}")
            return True
        except Exception as e:
            print(f"[agent] Skip failed: {e}")
        return False

    async def _wait_for_content(self, timeout: float = 5.0) -> None:
        """Wait for the SPA page to render content after navigation."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                await self.browser.page.wait_for_selector("h1", timeout=1000)
                return
            except Exception:
                await asyncio.sleep(0.15)
        await asyncio.sleep(0.5)
