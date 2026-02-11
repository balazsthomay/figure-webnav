"""JS-based popup/overlay cleaner injected into each page."""

from __future__ import annotations

from playwright.async_api import Page

# Non-destructive overlay clearing:
# - Hides modal backdrops and overlay elements (display:none)
# - Disables pointer-events on blocking layers
# - Does NOT remove() elements to avoid breaking the React app
CLEANUP_JS = """
(() => {
  // Overlay clearing for popups and modal backdrops.
  // Challenge elements (non-fixed, non-absolute, lower z-index) are NOT affected.
  // IMPORTANT: Preserve elements containing 6-char codes or "Code revealed"
  const codeRe = /[A-Z0-9]{6}/;
  document.querySelectorAll('*').forEach(el => {
    const s = getComputedStyle(el);
    const z = parseInt(s.zIndex);
    if (!isNaN(z) && z > 500 &&
        (s.position === 'fixed' || s.position === 'absolute')) {
      // Don't hide elements that contain challenge codes or success messages
      const text = (el.textContent || '').trim();
      if (/code revealed|your code/i.test(text) ||
          (codeRe.test(text) && /bg-green|text-green|success/i.test(el.className || ''))) {
        el.style.setProperty('pointer-events', 'none', 'important');
        return;
      }
      // Hide ALL overlay elements (both large backdrops and small popups)
      el.style.setProperty('display', 'none', 'important');
    }
  });
  window.scrollTo(0, 0);
})();
"""


async def clean_page(page: Page) -> None:
    """Clear popups/overlays and scroll to top. Runs twice to catch re-renders."""
    await page.evaluate(CLEANUP_JS)
    await page.wait_for_timeout(50)
    await page.evaluate(CLEANUP_JS)
