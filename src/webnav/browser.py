"""Async Playwright browser controller."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Dialog,
    Page,
    async_playwright,
)


BASE_URL = "https://serene-frangipane-7fd25b.netlify.app"


@dataclass
class BrowserController:
    """Manages a headless Chromium browser via Playwright."""

    headless: bool = True
    _playwright: Any = field(default=None, repr=False)
    _browser: Browser | None = field(default=None, repr=False)
    _context: BrowserContext | None = field(default=None, repr=False)
    _page: Page | None = field(default=None, repr=False)

    async def __aenter__(self) -> "BrowserController":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        # Inject shadow root capture BEFORE any page JS runs.
        # Force mode:'open' so Playwright locators auto-pierce shadow roots.
        # Also store __shadow reference as fallback for page.evaluate().
        await self._context.add_init_script("""
            (() => {
                const _origAttachShadow = Element.prototype.attachShadow;
                Element.prototype.attachShadow = function(init) {
                    const shadow = _origAttachShadow.call(this, { ...init, mode: 'open' });
                    this.__shadow = shadow;
                    return shadow;
                };
            })();
        """)
        self._page = await self._context.new_page()
        self._page.on("dialog", self._handle_dialog)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @staticmethod
    async def _handle_dialog(dialog: Dialog) -> None:
        await dialog.dismiss()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not started — use async with")
        return self._page

    async def goto(self, url: str, timeout: float = 10_000) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)

    async def goto_step(self, step: int) -> None:
        await self.goto(f"{BASE_URL}/step{step}?version=3")

    async def current_url(self) -> str:
        return self.page.url

    async def aria_snapshot(self, timeout: float = 5000) -> str:
        """Return the YAML aria snapshot for the page body."""
        return await self.page.locator("body").aria_snapshot(timeout=timeout)

    async def evaluate_js(self, script: str) -> Any:
        return await self.page.evaluate(script)

    async def screenshot(self) -> bytes:
        return await self.page.screenshot()

    async def inner_text(self, selector: str = "body") -> str:
        return await self.page.inner_text(selector)

    async def wait_for_timeout(self, ms: float) -> None:
        await self.page.wait_for_timeout(ms)
