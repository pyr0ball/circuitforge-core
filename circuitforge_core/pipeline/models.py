# circuitforge_core/pipeline/models.py — crystallization data models
#
# MIT — protocol and model types only; no inference backends.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Utilities ─────────────────────────────────────────────────────────────────

def hash_input(features: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of *features*.

    Sorts keys before serialising so insertion order doesn't affect the hash.
    Only call this on already-normalised, PII-free feature dicts — the hash is
    opaque but the source dict should never contain raw user data.
    """
    canonical = json.dumps(features, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Step ──────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """One atomic action in a deterministic workflow.

    The ``action`` string is product-defined (e.g. ``"dtmf"``, ``"field_fill"``,
    ``"api_call"``).  ``params`` carries action-specific values; ``description``
    is a plain-English summary for the approval UI.
    """
    action: str
    params: dict[str, Any]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "params": self.params,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(action=d["action"], params=d.get("params", {}),
                   description=d.get("description", ""))


# ── PipelineRun ───────────────────────────────────────────────────────────────

@dataclass
class PipelineRun:
    """Record of one LLM-assisted execution — the raw material for crystallization.

    Fields
    ------
    run_id:
        UUID or unique string identifying this run.
    product:
        CF product code (``"osprey"``, ``"falcon"``, ``"peregrine"`` …).
    task_type:
        Product-defined task category (``"ivr_navigate"``, ``"form_fill"`` …).
    input_hash:
        SHA-256 of normalised, PII-free input features.  Never store raw input.
    steps:
        Ordered list of Steps the LLM proposed.
    approved:
        True if a human approved this run before execution.
    review_duration_ms:
        Wall-clock milliseconds between displaying the proposal and the approval
        click.  Values under ~5 000 ms indicate a rubber-stamp — the
        crystallizer may reject runs with suspiciously short reviews.
    output_modified:
        True if the user edited any step before approving.  Modifications suggest
        the LLM proposal was imperfect; too-easy crystallization from unmodified
        runs may mean the task is already deterministic and the LLM is just
        echoing a fixed pattern.
    timestamp:
        ISO 8601 UTC creation time.
    llm_model:
        Model ID that generated the steps, e.g. ``"llama3:8b-instruct"``.
    metadata:
        Freeform dict for product-specific extra fields.
    """

    run_id: str
    product: str
    task_type: str
    input_hash: str
    steps: list[Step]
    approved: bool
    review_duration_ms: int
    output_modified: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    llm_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "product": self.product,
            "task_type": self.task_type,
            "input_hash": self.input_hash,
            "steps": [s.to_dict() for s in self.steps],
            "approved": self.approved,
            "review_duration_ms": self.review_duration_ms,
            "output_modified": self.output_modified,
            "timestamp": self.timestamp,
            "llm_model": self.llm_model,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineRun":
        return cls(
            run_id=d["run_id"],
            product=d["product"],
            task_type=d["task_type"],
            input_hash=d["input_hash"],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            approved=d["approved"],
            review_duration_ms=d["review_duration_ms"],
            output_modified=d.get("output_modified", False),
            timestamp=d.get("timestamp", ""),
            llm_model=d.get("llm_model"),
            metadata=d.get("metadata", {}),
        )


# ── CrystallizedWorkflow ──────────────────────────────────────────────────────

@dataclass
class CrystallizedWorkflow:
    """A deterministic workflow promoted from N approved PipelineRuns.

    Once crystallized, the executor runs ``steps`` directly — no LLM required
    unless an edge case is encountered.

    Fields
    ------
    workflow_id:
        Unique identifier (typically ``{product}:{task_type}:{input_hash[:12]}``).
    product / task_type / input_hash:
        Same semantics as PipelineRun; the hash is the lookup key.
    steps:
        Canonical deterministic step sequence (majority-voted or most-recent,
        per CrystallizerConfig.strategy).
    crystallized_at:
        ISO 8601 UTC timestamp.
    run_ids:
        IDs of the source PipelineRuns that contributed to this workflow.
    approval_count:
        Number of approved runs that went into crystallization.
    avg_review_duration_ms:
        Mean review_duration_ms across all source runs — low values are a
        warning sign that approvals may not have been genuine.
    all_output_unmodified:
        True if every contributing run had output_modified=False.  Combined with
        a very short avg_review_duration_ms this can flag workflows that may
        have crystallized from rubber-stamp approvals.
    active:
        Whether this workflow is in use.  Set to False to disable without
        deleting the record.
    version:
        Increments each time the workflow is re-crystallized from new runs.
    """

    workflow_id: str
    product: str
    task_type: str
    input_hash: str
    steps: list[Step]
    crystallized_at: str
    run_ids: list[str]
    approval_count: int
    avg_review_duration_ms: int
    all_output_unmodified: bool
    active: bool = True
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "product": self.product,
            "task_type": self.task_type,
            "input_hash": self.input_hash,
            "steps": [s.to_dict() for s in self.steps],
            "crystallized_at": self.crystallized_at,
            "run_ids": self.run_ids,
            "approval_count": self.approval_count,
            "avg_review_duration_ms": self.avg_review_duration_ms,
            "all_output_unmodified": self.all_output_unmodified,
            "active": self.active,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CrystallizedWorkflow":
        return cls(
            workflow_id=d["workflow_id"],
            product=d["product"],
            task_type=d["task_type"],
            input_hash=d["input_hash"],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            crystallized_at=d["crystallized_at"],
            run_ids=d.get("run_ids", []),
            approval_count=d["approval_count"],
            avg_review_duration_ms=d["avg_review_duration_ms"],
            all_output_unmodified=d.get("all_output_unmodified", True),
            active=d.get("active", True),
            version=d.get("version", 1),
            metadata=d.get("metadata", {}),
        )
