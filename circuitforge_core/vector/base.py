"""
circuitforge_core.vector.base — VectorStore ABC and shared types.

Concrete implementations: LocalSQLiteVecStore (local), QdrantStore (cloud Paid tier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class VectorMatch:
    """A single result from a vector similarity search."""

    entry_id: str
    score: float  # lower is better (L2 / cosine distance)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def upsert(
        self, entry_id: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        """Insert or replace a vector and its metadata."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Return the top_k nearest vectors. Optional metadata filter applied post-search."""

    @abstractmethod
    def delete(self, entry_id: str) -> None:
        """Remove a single vector by string ID. No-op if not found."""

    @abstractmethod
    def delete_where(self, filter_metadata: dict[str, Any]) -> int:
        """Remove all vectors whose metadata matches all key-value pairs. Returns count removed.

        Raises ValueError if filter_metadata is empty (would delete entire store).
        """
