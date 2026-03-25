"""
SQLite-backed staging queue for CircuitForge pipeline tasks.
Full implementation deferred — stub raises NotImplementedError.
"""
from __future__ import annotations
from typing import Any


class StagingDB:
    """
    Staging queue for background pipeline tasks (search polling, score updates, etc.)
    Stub: raises NotImplementedError until wired up in a product.
    """

    def enqueue(self, task_type: str, payload: dict[str, Any]) -> None:
        """Add a task to the staging queue."""
        raise NotImplementedError(
            "StagingDB.enqueue() is not yet implemented. "
            "Background task pipeline is a v0.2+ feature."
        )

    def dequeue(self) -> tuple[str, dict[str, Any]] | None:
        """Fetch the next pending task. Returns (task_type, payload) or None."""
        raise NotImplementedError(
            "StagingDB.dequeue() is not yet implemented."
        )
