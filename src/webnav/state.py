"""State tracker: step counter, time budget, stuck detection."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from playwright.async_api import Page


@dataclass
class StateTracker:
    """Tracks agent progress through the challenge."""

    total_steps: int = 30
    max_time: float = 290.0  # 4m50s safety margin
    max_step_time: float = 15.0  # skip step if stuck >15s
    max_retries: int = 3  # stuck threshold per step

    current_step: int = 1
    steps_completed: int = 0
    steps_failed: int = 0
    start_time: float = field(default_factory=time.time)
    step_start_time: float = 0.0
    retry_count: int = 0

    def begin_step(self, step: int) -> None:
        self.current_step = step
        self.step_start_time = time.time()
        self.retry_count = 0

    def step_elapsed(self) -> float:
        return time.time() - self.step_start_time

    def total_elapsed(self) -> float:
        return time.time() - self.start_time

    def time_remaining(self) -> float:
        return max(0, self.max_time - self.total_elapsed())

    def is_over_budget(self) -> bool:
        return self.total_elapsed() >= self.max_time

    def is_step_timed_out(self) -> bool:
        return self.step_elapsed() >= self.max_step_time

    def is_stuck(self) -> bool:
        return self.retry_count >= self.max_retries

    def increment_retry(self) -> None:
        self.retry_count += 1

    def mark_completed(self) -> None:
        self.steps_completed += 1

    def mark_failed(self) -> None:
        self.steps_failed += 1

    async def detect_step_from_url(self, page: Page) -> int:
        """Parse current step number from the page URL."""
        url = page.url
        m = re.search(r"/step(\d+)", url)
        if m:
            return int(m.group(1))
        return self.current_step

    async def check_advancement(self, page: Page, expected_step: int) -> bool:
        """Check if the page has advanced past the expected step."""
        actual = await self.detect_step_from_url(page)
        return actual > expected_step

    def summary(self) -> dict:
        return {
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "total_time": round(self.total_elapsed(), 1),
            "current_step": self.current_step,
        }
