"""Tests for pipeline.Executor."""
import pytest
from circuitforge_core.pipeline.executor import Executor, ExecutionResult
from circuitforge_core.pipeline.models import CrystallizedWorkflow, Step


def _wf(steps=None) -> CrystallizedWorkflow:
    return CrystallizedWorkflow(
        workflow_id="osprey:ivr_navigate:abc",
        product="osprey",
        task_type="ivr_navigate",
        input_hash="abc",
        steps=[Step("dtmf", {"digits": "1"}), Step("dtmf", {"digits": "2"})]
              if steps is None else steps,
        crystallized_at="2026-04-08T00:00:00+00:00",
        run_ids=["r1"],
        approval_count=1,
        avg_review_duration_ms=8000,
        all_output_unmodified=True,
    )


def _ok_step(_step):
    return True, "ok"


def _fail_step(_step):
    return False, None


def _raise_step(_step):
    raise RuntimeError("hardware error")


def _llm():
    return "llm-output"


class TestExecutor:
    def test_all_steps_succeed(self):
        ex = Executor(step_fn=_ok_step, llm_fn=_llm)
        result = ex.execute(_wf())
        assert result.success is True
        assert result.used_deterministic is True
        assert len(result.step_results) == 2

    def test_failed_step_triggers_llm_fallback(self):
        ex = Executor(step_fn=_fail_step, llm_fn=_llm)
        result = ex.execute(_wf())
        assert result.success is True
        assert result.used_deterministic is False
        assert result.llm_output == "llm-output"

    def test_raising_step_triggers_llm_fallback(self):
        ex = Executor(step_fn=_raise_step, llm_fn=_llm)
        result = ex.execute(_wf())
        assert result.success is True
        assert result.used_deterministic is False

    def test_llm_fallback_disabled_returns_failure(self):
        ex = Executor(step_fn=_fail_step, llm_fn=_llm, llm_fallback=False)
        result = ex.execute(_wf())
        assert result.success is False
        assert "disabled" in (result.error or "")

    def test_run_with_fallback_no_workflow_calls_llm(self):
        ex = Executor(step_fn=_ok_step, llm_fn=_llm)
        result = ex.run_with_fallback(workflow=None)
        assert result.success is True
        assert result.used_deterministic is False
        assert result.llm_output == "llm-output"

    def test_run_with_fallback_uses_workflow_when_given(self):
        ex = Executor(step_fn=_ok_step, llm_fn=_llm)
        result = ex.run_with_fallback(workflow=_wf())
        assert result.used_deterministic is True

    def test_llm_fn_raises_returns_failure(self):
        def _bad_llm():
            raise ValueError("no model")

        ex = Executor(step_fn=_fail_step, llm_fn=_bad_llm)
        result = ex.execute(_wf())
        assert result.success is False
        assert "no model" in (result.error or "")

    def test_workflow_id_preserved_in_result(self):
        ex = Executor(step_fn=_ok_step, llm_fn=_llm)
        result = ex.execute(_wf())
        assert result.workflow_id == "osprey:ivr_navigate:abc"

    def test_empty_workflow_succeeds_immediately(self):
        ex = Executor(step_fn=_ok_step, llm_fn=_llm)
        result = ex.execute(_wf(steps=[]))
        assert result.success is True
        assert result.step_results == []
