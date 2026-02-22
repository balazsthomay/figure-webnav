"""JS-based popup/overlay cleaner injected into each page."""

from __future__ import annotations

from playwright.async_api import Page

# Non-destructive overlay clearing:
# - Hides modal backdrops and overlay elements (display:none)
# - Disables pointer-events on blocking layers
# - Does NOT remove() elements to avoid breaking the React app
CLEANUP_JS = """
(() => {
  // Phase 1: Hide fixed-position overlay backdrops and modals.
  // Only targets position:fixed elements (not absolute) because
  // challenge content uses absolute positioning for split parts,
  // puzzle containers, etc.
  // Challenge content markers — if a fixed overlay contains these,
  // it's a challenge panel, not a noise popup.
  const challengeRe = /complete\s*\(\d|complete all|sequence|puzzle|click me|click here.*times|hover over|hover here|scroll inside|solve|draw|capture|play audio|enter level|tab \d|service worker/i;
  document.querySelectorAll('*').forEach(el => {
    const s = getComputedStyle(el);
    const z = parseInt(s.zIndex);
    if (!isNaN(z) && z > 500 && s.position === 'fixed') {
      // Check if this overlay contains challenge content
      const inner = el.innerText || '';
      const hasCanvas = el.querySelector('canvas');
      if (hasCanvas || challengeRe.test(inner)) return; // keep it
      el.setAttribute('data-wnav-noise', 'true');
      el.style.setProperty('display', 'none', 'important');
    }
  });
  // Phase 2: Remove noise popup buttons (not in overlays but still noise)
  // If >6 buttons have generic nav text, they're popup noise — hide them
  const noiseRe = /^(advance|continue|next|proceed|go forward|keep going|move on|next page|next step|next section|continue reading|continue journey|proceed forward|go)$/i;
  const noiseBtns = [];
  document.querySelectorAll('button').forEach(btn => {
    if (btn.offsetParent === null) return;
    const text = (btn.textContent || '').trim();
    if (noiseRe.test(text)) noiseBtns.push(btn);
  });
  if (noiseBtns.length > 8) {
    noiseBtns.forEach(btn => {
      btn.setAttribute('data-wnav-noise', 'true');
      btn.style.setProperty('display', 'none', 'important');
    });
  }
  // Also hide floating noise divs with popup text — but NOT if inside a challenge container
  const popupTextRe = /^(click me!?|click here!?|here!?|try this!?|moving!?|button!?|link!?)$/i;
  document.querySelectorAll('div, span').forEach(el => {
    if (el.offsetParent === null) return;
    const text = (el.textContent || '').trim();
    if (!popupTextRe.test(text)) return;
    const style = getComputedStyle(el);
    if (style.position === 'absolute' || style.position === 'fixed') {
      // Walk ancestors — if any contain challenge content, this is a real
      // sub-element (e.g. "Click Me" inside a Sequence challenge), not noise.
      let p = el.parentElement;
      let insideChallenge = false;
      for (let i = 0; i < 10 && p && p !== document.body; i++) {
        const pt = p.innerText || '';
        if (challengeRe.test(pt)) { insideChallenge = true; break; }
        p = p.parentElement;
      }
      if (insideChallenge) return;
      el.setAttribute('data-wnav-noise', 'true');
      el.style.setProperty('display', 'none', 'important');
    }
  });
  // NOTE: Do NOT scroll to top here — it breaks scroll-to-reveal challenges
})();
"""


_RESET_JS = """
(() => {
    for (const el of document.querySelectorAll('[data-wnav-noise]')) {
        el.removeAttribute('data-wnav-noise');
        el.style.removeProperty('display');
        el.style.removeProperty('pointer-events');
        el.style.removeProperty('opacity');
    }
})()
"""


async def reset_cleaner(page: Page) -> None:
    """Undo inline style modifications from previous clean_page() calls.

    Targets only !important inline styles (strong signal of cleaner origin —
    normal page styles use classes, not inline !important).
    """
    await page.evaluate(_RESET_JS)


async def clean_page(page: Page) -> None:
    """Clear popups/overlays. Single pass — double-run added no value
    (React re-renders are async and won't be caught by a 50ms sync wait)."""
    await page.evaluate(CLEANUP_JS)


# Combined reset + clean in a single evaluate() call.
# Eliminates 2 extra round-trips and 0.3s of sleeps vs calling
# reset_cleaner() + sleep + clean_page() + sleep separately.
_QUICK_CLEAN_JS = _RESET_JS.rstrip().rstrip(";") + ";\n" + CLEANUP_JS


async def quick_clean(page: Page) -> None:
    """Reset stale cleaner CSS then clean overlays — single JS round-trip."""
    await page.evaluate(_QUICK_CLEAN_JS)
    await page.wait_for_timeout(50)
