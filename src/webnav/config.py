"""Challenge configuration — parameterizes the agent for any challenge site."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChallengeConfig:
    """Configuration for the web navigation challenge."""

    url: str = "https://serene-frangipane-7fd25b.netlify.app"
    total_steps: int = 30
    max_time: float = 295.0  # 4m55s safety margin
    max_step_time: float = 15.0
    max_retries: int = 2
    code_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"\b([A-Z0-9]{6})\b")
    )
