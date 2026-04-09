# circuitforge_core/pipeline/executor.py — deterministic execution with LLM fallback
#
# MIT — orchestration logic only; calls product-supplied callables.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import CrystallizedWorkflow, Step

log = logging.getLogger(__name__)


@dataclass
class StepResult:
    step: Step
    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class ExecutionResult:
    """Result of running a workflow (deterministic or LLM-assisted).

    Attributes
    ----------
    success:
        True if all steps completed without error.
    used_deterministic:
        True if a crystallized workflow was used; False if LLM was called.
    step_results:
        Per-step outcomes from the deterministic path.
    llm_output:
        Raw output from the LLM fallback path, if used.
    workflow_id:
        ID of the workflow used, or None for LLM path.
    error:
        Error message if the run failed entirely.
    """
    success: bool
    used_deterministic: bool
    step_results: list[StepResult] = field(default_factory=list)
    llm_output: Any = None
    workflow_id: str | None = None
    error: str | None = None


# ── Executor ──────────────────────────────────────────────────────────────────

class Executor:
    """Runs crystallized workflows with transparent LLM fallback.

    Parameters
    ----------
    step_fn:
        Called for each Step: ``step_fn(step) -> (success, output)``.
        The product supplies this — it knows how to turn a Step into a real
        action (DTMF dial, HTTP call, form field write, etc.).
    llm_fn:
        Called when no workflow matches or a step fails: ``llm_fn() -> output``.
        Products wire this to ``cf_core.llm.router`` or equivalent.
    llm_fallback:
        If False, raise RuntimeError instead of calling llm_fn on miss.
    """

    def __init__(
        self,
        step_fn: Callable[[Step], tuple[bool, Any]],
        llm_fn: Callable[[], Any],
        llm_fallback: bool = True,
    ) -> None:
        self._step_fn = step_fn
        self._llm_fn = llm_fn
        self._llm_fallback = llm_fallback

    def execute(
        self,
        workflow: CrystallizedWorkflow,
    ) -> ExecutionResult:
        """Run *workflow* deterministically.

        If a step fails, falls back to LLM (if ``llm_fallback`` is enabled).
        """
        step_results: list[StepResult] = []
        for step in workflow.steps:
            try:
                success, output = self._step_fn(step)
            except Exception as exc:
                log.warning("step %s raised: %s", step.action, exc)
                success, output = False, None
                error_str = str(exc)
            else:
                error_str = None if success else "step_fn returned success=False"

            step_results.append(StepResult(step=step, success=success,
                                           output=output, error=error_str))
            if not success:
                log.info(
                    "workflow %s: step %s failed — triggering LLM fallback",
                    workflow.workflow_id, step.action,
                )
                return self._llm_fallback_result(
                    step_results, workflow.workflow_id
                )

        log.info("workflow %s: all %d steps succeeded",
                 workflow.workflow_id, len(workflow.steps))
        return ExecutionResult(
            success=True,
            used_deterministic=True,
            step_results=step_results,
            workflow_id=workflow.workflow_id,
        )

    def run_with_fallback(
        self,
        workflow: CrystallizedWorkflow | None,
    ) -> ExecutionResult:
        """Run *workflow* if provided; otherwise call the LLM directly."""
        if workflow is None:
            return self._llm_fallback_result([], workflow_id=None)
        return self.execute(workflow)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _llm_fallback_result(
        self,
        partial_steps: list[StepResult],
        workflow_id: str | None,
    ) -> ExecutionResult:
        if not self._llm_fallback:
            return ExecutionResult(
                success=False,
                used_deterministic=True,
                step_results=partial_steps,
                workflow_id=workflow_id,
                error="LLM fallback disabled and deterministic path failed.",
            )
        try:
            llm_output = self._llm_fn()
        except Exception as exc:
            return ExecutionResult(
                success=False,
                used_deterministic=False,
                step_results=partial_steps,
                workflow_id=workflow_id,
                error=f"LLM fallback raised: {exc}",
            )
        return ExecutionResult(
            success=True,
            used_deterministic=False,
            step_results=partial_steps,
            llm_output=llm_output,
            workflow_id=workflow_id,
        )
