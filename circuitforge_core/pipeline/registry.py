# circuitforge_core/pipeline/registry.py — workflow lookup
#
# MIT — file I/O and matching logic only.
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from .models import CrystallizedWorkflow

log = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".config" / "circuitforge" / "pipeline" / "workflows"


class Registry:
    """Loads and matches CrystallizedWorkflows from the local filesystem.

    Layout::

        {root}/{product}/{task_type}/{workflow_id}.json

    Exact matching is always available.  Products that need fuzzy/semantic
    matching can supply a ``similarity_fn`` — a callable that takes two input
    hashes and returns a float in [0, 1].  The registry returns the first
    active workflow whose similarity score meets ``fuzzy_threshold``.
    """

    def __init__(
        self,
        root: Path | None = None,
        similarity_fn: Callable[[str, str], float] | None = None,
        fuzzy_threshold: float = 0.8,
    ) -> None:
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._similarity_fn = similarity_fn
        self._fuzzy_threshold = fuzzy_threshold

    # ── Write ─────────────────────────────────────────────────────────────────

    def register(self, workflow: CrystallizedWorkflow) -> Path:
        """Persist *workflow* and return the path written."""
        dest = self._path_for(workflow.product, workflow.task_type,
                              workflow.workflow_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(workflow.to_dict(), indent=2), encoding="utf-8")
        log.info("registered workflow %s (v%d)", workflow.workflow_id,
                 workflow.version)
        return dest

    def deactivate(self, workflow_id: str, product: str,
                   task_type: str) -> bool:
        """Set ``active=False`` on a stored workflow.  Returns True if found."""
        path = self._path_for(product, task_type, workflow_id)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        data["active"] = False
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("deactivated workflow %s", workflow_id)
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_all(self, product: str, task_type: str) -> list[CrystallizedWorkflow]:
        """Return all (including inactive) workflows for *(product, task_type)*."""
        directory = self._root / product / task_type
        if not directory.is_dir():
            return []
        workflows: list[CrystallizedWorkflow] = []
        for p in directory.glob("*.json"):
            try:
                workflows.append(
                    CrystallizedWorkflow.from_dict(json.loads(p.read_text()))
                )
            except Exception:
                log.warning("skipping unreadable workflow file %s", p)
        return workflows

    # ── Match ─────────────────────────────────────────────────────────────────

    def match(self, product: str, task_type: str,
              input_hash: str) -> CrystallizedWorkflow | None:
        """Return the active workflow for an exact input_hash match, or None."""
        for wf in self.load_all(product, task_type):
            if wf.active and wf.input_hash == input_hash:
                log.debug("registry exact match: %s", wf.workflow_id)
                return wf
        return None

    def fuzzy_match(self, product: str, task_type: str,
                    input_hash: str) -> CrystallizedWorkflow | None:
        """Return a workflow above the similarity threshold, or None.

        Requires a ``similarity_fn`` to have been supplied at construction.
        If none was provided, raises ``RuntimeError``.
        """
        if self._similarity_fn is None:
            raise RuntimeError(
                "fuzzy_match() requires a similarity_fn — none was supplied "
                "to Registry.__init__()."
            )
        best: CrystallizedWorkflow | None = None
        best_score = 0.0
        for wf in self.load_all(product, task_type):
            if not wf.active:
                continue
            score = self._similarity_fn(wf.input_hash, input_hash)
            if score >= self._fuzzy_threshold and score > best_score:
                best = wf
                best_score = score
        if best:
            log.debug("registry fuzzy match: %s (score=%.2f)", best.workflow_id,
                      best_score)
        return best

    def find(self, product: str, task_type: str,
             input_hash: str) -> CrystallizedWorkflow | None:
        """Exact match first; fuzzy match second (if similarity_fn is set)."""
        exact = self.match(product, task_type, input_hash)
        if exact:
            return exact
        if self._similarity_fn is not None:
            return self.fuzzy_match(product, task_type, input_hash)
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path_for(self, product: str, task_type: str,
                  workflow_id: str) -> Path:
        safe_id = workflow_id.replace(":", "_")
        return self._root / product / task_type / f"{safe_id}.json"
