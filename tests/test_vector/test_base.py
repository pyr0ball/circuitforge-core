"""Tests for VectorStore ABC and VectorMatch."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from circuitforge_core.vector.base import VectorMatch, VectorStore


class _ConcreteStore(VectorStore):
    """Minimal in-memory implementation for testing the ABC contract."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, entry_id: str, vector: list[float], metadata: dict) -> None:
        self._data[entry_id] = (vector, metadata)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[VectorMatch]:
        results = [
            VectorMatch(entry_id=k, score=0.0, metadata=v[1])
            for k, v in self._data.items()
        ]
        if filter_metadata:
            results = [
                r
                for r in results
                if all(r.metadata.get(k) == val for k, val in filter_metadata.items())
            ]
        return results[:top_k]

    def delete(self, entry_id: str) -> None:
        self._data.pop(entry_id, None)

    def delete_where(self, filter_metadata: dict) -> int:
        to_remove = [
            k
            for k, (_, meta) in self._data.items()
            if all(meta.get(fk) == fv for fk, fv in filter_metadata.items())
        ]
        for k in to_remove:
            del self._data[k]
        return len(to_remove)


def test_vector_match_is_frozen():
    match = VectorMatch(entry_id="a", score=0.1, metadata={})
    with pytest.raises(FrozenInstanceError):
        match.score = 0.5  # type: ignore[misc]


def test_vector_match_metadata_is_not_mutable():
    match = VectorMatch(entry_id="a", score=0.1, metadata={"k": "v"})
    assert isinstance(match.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        match.metadata["k"] = "changed"  # type: ignore[index]


def test_upsert_and_query():
    store = _ConcreteStore()
    store.upsert("chunk-1", [0.1, 0.2], {"doc_id": "book-a", "page": 1})
    results = store.query([0.1, 0.2])
    assert len(results) == 1
    assert results[0].entry_id == "chunk-1"
    assert results[0].metadata["page"] == 1


def test_query_filter_metadata():
    store = _ConcreteStore()
    store.upsert("c1", [0.1], {"doc_id": "book-a"})
    store.upsert("c2", [0.2], {"doc_id": "book-b"})
    results = store.query([0.1], filter_metadata={"doc_id": "book-a"})
    assert len(results) == 1
    assert results[0].entry_id == "c1"


def test_delete():
    store = _ConcreteStore()
    store.upsert("x", [0.1], {})
    store.delete("x")
    assert store.query([0.1]) == []


def test_delete_where():
    store = _ConcreteStore()
    store.upsert("c1", [0.1], {"doc_id": "book-a"})
    store.upsert("c2", [0.2], {"doc_id": "book-a"})
    store.upsert("c3", [0.3], {"doc_id": "book-b"})
    count = store.delete_where({"doc_id": "book-a"})
    assert count == 2
    assert len(store.query([0.1])) == 1


def test_cannot_instantiate_abc_directly():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]
