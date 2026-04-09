# circuitforge_core/pipeline/recorder.py — write and load PipelineRun records
#
# MIT — local file I/O only; no inference.
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from .models import PipelineRun

log = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".config" / "circuitforge" / "pipeline" / "runs"


class Recorder:
    """Writes PipelineRun JSON records to a local directory tree.

    Layout::

        {root}/{product}/{task_type}/{run_id}.json

    The recorder is intentionally append-only — it never deletes or modifies
    existing records.  Old runs accumulate as an audit trail; products that
    want retention limits should prune the directory themselves.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root else _DEFAULT_ROOT

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, run: PipelineRun) -> Path:
        """Persist *run* to disk and return the file path written."""
        dest = self._path_for(run.product, run.task_type, run.run_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        log.debug("recorded pipeline run %s → %s", run.run_id, dest)
        return dest

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_runs(self, product: str, task_type: str) -> list[PipelineRun]:
        """Return all runs for *(product, task_type)*, newest-first."""
        directory = self._root / product / task_type
        if not directory.is_dir():
            return []
        runs: list[PipelineRun] = []
        for p in directory.glob("*.json"):
            try:
                runs.append(PipelineRun.from_dict(json.loads(p.read_text())))
            except Exception:
                log.warning("skipping unreadable run file %s", p)
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs

    def load_approved(self, product: str, task_type: str,
                      input_hash: str) -> list[PipelineRun]:
        """Return approved runs that match *input_hash*, newest-first."""
        return [
            r for r in self.load_runs(product, task_type)
            if r.approved and r.input_hash == input_hash
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path_for(self, product: str, task_type: str, run_id: str) -> Path:
        return self._root / product / task_type / f"{run_id}.json"
