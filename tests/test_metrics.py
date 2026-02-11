"""Tests for metrics module."""

from __future__ import annotations

from webnav.metrics import MetricsCollector, StepMetric


class TestStepMetric:
    def test_defaults(self):
        m = StepMetric(step=1)
        assert m.wall_time == 0.0
        assert m.llm_calls == 0
        assert m.success is False


class TestMetricsCollector:
    def test_begin_end_step(self):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.end_step(True, code="ABC123")
        assert len(mc.steps) == 1
        assert mc.steps[0].success is True
        assert mc.steps[0].code_found == "ABC123"
        assert mc.steps[0].wall_time > 0

    def test_end_step_without_begin(self):
        mc = MetricsCollector()
        mc.end_step(True)  # Should not crash
        assert len(mc.steps) == 0

    def test_record_llm_call(self):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.record_llm_call(tier=1, tokens=150)
        mc.end_step(True)
        assert mc.steps[0].llm_calls == 1
        assert mc.steps[0].llm_tokens == 150
        assert mc.steps[0].tier_used == 1

    def test_record_tier_escalation(self):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.record_llm_call(tier=1, tokens=100)
        mc.record_llm_call(tier=2, tokens=200)
        mc.end_step(True)
        assert mc.steps[0].tier_used == 2  # Max tier used

    def test_aggregate_properties(self):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.record_llm_call(1, 100)
        mc.end_step(True)
        mc.begin_step(2)
        mc.record_llm_call(1, 200)
        mc.end_step(False)

        assert mc.steps_succeeded == 1
        assert mc.steps_failed == 1
        assert mc.total_llm_calls == 2
        assert mc.total_llm_tokens == 300

    def test_estimated_cost(self):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.record_llm_call(1, 1_000_000)
        mc.end_step(True)
        cost = mc.estimated_cost()
        assert cost > 0

    def test_print_report(self, capsys):
        mc = MetricsCollector()
        mc.begin_step(1)
        mc.end_step(True, code="ABC123")
        mc.begin_step(2)
        mc.end_step(False, error="no code found")
        mc.print_report()
        output = capsys.readouterr().out
        assert "CHALLENGE RESULTS" in output
        assert "Completed: 1/2" in output
        assert "ABC123" in output
