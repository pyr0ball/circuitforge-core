# circuitforge_core/pipeline/crystallizer.py — promote approved runs → workflows
#
# MIT — pure logic, no inference backends.
from __future__ import annotations

import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .models import CrystallizedWorkflow, PipelineRun, Step
from .recorder import Recorder

log = logging.getLogger(__name__)

# Minimum milliseconds of review that counts as "genuine".
# Runs shorter than this are accepted but trigger a warning.
_RUBBER_STAMP_THRESHOLD_MS = 5_000


@dataclass
class CrystallizerConfig:
    """Tuning knobs for one product/task-type pair.

    threshold:
        Minimum number of approved runs required before crystallization.
        Osprey sets this to 1 (first successful IVR navigation is enough);
        Peregrine uses 3+ for cover-letter templates.
    min_review_ms:
        Approved runs with review_duration_ms below this value generate a
        warning.  Set to 0 to silence the check (tests, automated approvals).
    strategy:
        ``"most_recent"`` — use the latest approved run's steps verbatim.
        ``"majority"`` — pick each step by majority vote across runs (requires
        runs to have the same step count; falls back to most_recent otherwise).
    """
    threshold: int = 3
    min_review_ms: int = _RUBBER_STAMP_THRESHOLD_MS
    strategy: Literal["most_recent", "majority"] = "most_recent"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _majority_steps(runs: list[PipelineRun]) -> list[Step] | None:
    """Return majority-voted steps, or None if run lengths differ."""
    lengths = {len(r.steps) for r in runs}
    if len(lengths) != 1:
        return None
    n = lengths.pop()
    result: list[Step] = []
    for i in range(n):
        counter: Counter[str] = Counter()
        step_by_action: dict[str, Step] = {}
        for r in runs:
            s = r.steps[i]
            counter[s.action] += 1
            step_by_action[s.action] = s
        winner = counter.most_common(1)[0][0]
        result.append(step_by_action[winner])
    return result


def _check_review_quality(runs: list[PipelineRun],
                          min_review_ms: int) -> None:
    """Warn if any run has a suspiciously short review duration."""
    if min_review_ms <= 0:
        return
    flagged = [r for r in runs if r.review_duration_ms < min_review_ms]
    if flagged:
        ids = ", ".join(r.run_id for r in flagged)
        warnings.warn(
            f"Crystallizing from {len(flagged)} run(s) with review_duration_ms "
            f"< {min_review_ms} ms — possible rubber-stamp approval: [{ids}]. "
            "Verify these were genuinely human-reviewed before deployment.",
            stacklevel=3,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def should_crystallize(runs: list[PipelineRun],
                       config: CrystallizerConfig) -> bool:
    """Return True if *runs* meet the threshold for crystallization."""
    approved = [r for r in runs if r.approved]
    return len(approved) >= config.threshold


def crystallize(runs: list[PipelineRun],
                config: CrystallizerConfig,
                existing_version: int = 0) -> CrystallizedWorkflow:
    """Promote *runs* into a CrystallizedWorkflow.

    Raises
    ------
    ValueError
        If fewer approved runs than ``config.threshold``, or if the runs
        span more than one (product, task_type, input_hash) triple.
    """
    approved = [r for r in runs if r.approved]
    if len(approved) < config.threshold:
        raise ValueError(
            f"Need {config.threshold} approved runs, got {len(approved)}."
        )

    # Validate homogeneity
    products = {r.product for r in approved}
    task_types = {r.task_type for r in approved}
    hashes = {r.input_hash for r in approved}
    if len(products) != 1 or len(task_types) != 1 or len(hashes) != 1:
        raise ValueError(
            "All runs must share the same product, task_type, and input_hash. "
            f"Got products={products}, task_types={task_types}, hashes={hashes}."
        )

    product = products.pop()
    task_type = task_types.pop()
    input_hash = hashes.pop()

    _check_review_quality(approved, config.min_review_ms)

    # Pick canonical steps
    if config.strategy == "majority":
        steps = _majority_steps(approved) or approved[-1].steps
    else:
        steps = sorted(approved, key=lambda r: r.timestamp)[-1].steps

    avg_ms = sum(r.review_duration_ms for r in approved) // len(approved)
    all_unmodified = all(not r.output_modified for r in approved)

    workflow_id = f"{product}:{task_type}:{input_hash[:12]}"
    return CrystallizedWorkflow(
        workflow_id=workflow_id,
        product=product,
        task_type=task_type,
        input_hash=input_hash,
        steps=steps,
        crystallized_at=datetime.now(timezone.utc).isoformat(),
        run_ids=[r.run_id for r in approved],
        approval_count=len(approved),
        avg_review_duration_ms=avg_ms,
        all_output_unmodified=all_unmodified,
        version=existing_version + 1,
    )


def evaluate_new_run(
    run: PipelineRun,
    recorder: Recorder,
    config: CrystallizerConfig,
    existing_version: int = 0,
) -> CrystallizedWorkflow | None:
    """Record *run* and return a new workflow if the threshold is now met.

    Products call this after each human-approved execution.  Returns a
    ``CrystallizedWorkflow`` if crystallization was triggered, ``None``
    otherwise.
    """
    recorder.record(run)
    if not run.approved:
        return None

    all_runs = recorder.load_approved(run.product, run.task_type, run.input_hash)
    if not should_crystallize(all_runs, config):
        log.debug(
            "pipeline: %d/%d approved runs for %s:%s — not yet crystallizing",
            len(all_runs), config.threshold, run.product, run.task_type,
        )
        return None

    workflow = crystallize(all_runs, config, existing_version=existing_version)
    log.info(
        "pipeline: crystallized %s after %d approvals",
        workflow.workflow_id, workflow.approval_count,
    )
    return workflow
