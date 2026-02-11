"""Metrics collection and reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StepMetric:
    """Metrics for a single step."""

    step: int
    wall_time: float = 0.0
    llm_calls: int = 0
    llm_tokens: int = 0
    tier_used: int = 0  # 0 = programmatic, 1 = fast LLM, 2 = smart LLM
    success: bool = False
    code_found: str = ""
    error: str = ""


@dataclass
class MetricsCollector:
    """Collects and reports metrics across the entire run."""

    steps: list[StepMetric] = field(default_factory=list)
    run_start: float = field(default_factory=time.time)
    _step_start: float = 0.0
    _current: StepMetric | None = None

    # OpenRouter pricing (per 1M tokens, approximate)
    _COST_PER_1M: dict[str, float] = field(default_factory=lambda: {
        "google/gemini-2.5-flash-preview": 0.15,
        "anthropic/claude-haiku-4-5": 1.00,
    })

    def begin_step(self, step: int) -> None:
        self._step_start = time.time()
        self._current = StepMetric(step=step)

    def end_step(self, success: bool, code: str = "", error: str = "") -> None:
        if self._current is None:
            return
        self._current.wall_time = time.time() - self._step_start
        self._current.success = success
        self._current.code_found = code
        self._current.error = error
        self.steps.append(self._current)
        self._current = None

    def record_llm_call(self, tier: int, tokens: int) -> None:
        if self._current:
            self._current.llm_calls += 1
            self._current.llm_tokens += tokens
            self._current.tier_used = max(self._current.tier_used, tier)

    @property
    def total_wall_time(self) -> float:
        return time.time() - self.run_start

    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def total_llm_tokens(self) -> int:
        return sum(s.llm_tokens for s in self.steps)

    @property
    def steps_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.success)

    @property
    def steps_failed(self) -> int:
        return sum(1 for s in self.steps if not s.success)

    def estimated_cost(self) -> float:
        """Rough cost estimate based on token usage."""
        # Use average rate since we don't track per-model tokens here
        avg_rate = 0.30  # $/1M tokens (weighted average)
        return self.total_llm_tokens * avg_rate / 1_000_000

    def print_step_summary(self, step_metric: StepMetric) -> None:
        status = "OK" if step_metric.success else "FAIL"
        tier_label = ["T0:regex", "T1:flash", "T2:haiku"][step_metric.tier_used]
        code_info = f" code={step_metric.code_found}" if step_metric.code_found else ""
        err_info = f" err={step_metric.error}" if step_metric.error else ""
        print(
            f"  Step {step_metric.step:2d}: [{status}] "
            f"{step_metric.wall_time:5.1f}s "
            f"{tier_label} "
            f"llm={step_metric.llm_calls} tok={step_metric.llm_tokens}"
            f"{code_info}{err_info}"
        )

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("  CHALLENGE RESULTS")
        print("=" * 60)
        for s in self.steps:
            self.print_step_summary(s)
        print("-" * 60)
        print(f"  Completed: {self.steps_succeeded}/{len(self.steps)}")
        print(f"  Total time: {self.total_wall_time:.1f}s")
        print(f"  LLM calls: {self.total_llm_calls}")
        print(f"  LLM tokens: {self.total_llm_tokens}")
        print(f"  Est. cost: ${self.estimated_cost():.4f}")
        avg = self.total_wall_time / max(len(self.steps), 1)
        print(f"  Avg time/step: {avg:.1f}s")
        print("=" * 60 + "\n")
