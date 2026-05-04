# tests/test_vector/test_sqlite_vec.py
"""Integration tests for LocalSQLiteVecStore (uses a real in-memory sqlite-vec DB)."""

from __future__ import annotations

import pytest

from circuitforge_core.vector.sqlite_vec import LocalSQLiteVecStore

DIMS = 4  # small dimension for tests


@pytest.fixture
def store(tmp_path) -> LocalSQLiteVecStore:
    return LocalSQLiteVecStore(db_path=tmp_path / "vecs.db", dimensions=DIMS)


def _vec(val: float) -> list[float]:
    return [val] * DIMS


def test_upsert_and_query_returns_match(store):
    store.upsert("doc-1::p1", _vec(0.1), {"doc_id": "doc-1", "page": 1})
    results = store.query(_vec(0.1), top_k=5)
    assert len(results) == 1
    assert results[0].entry_id == "doc-1::p1"
    assert results[0].metadata["page"] == 1


def test_upsert_replaces_existing(store):
    store.upsert("chunk-1", _vec(0.1), {"page": 1})
    store.upsert("chunk-1", _vec(0.2), {"page": 99})
    results = store.query(_vec(0.2), top_k=5)
    assert results[0].metadata["page"] == 99


def test_query_respects_top_k(store):
    for i in range(5):
        store.upsert(f"chunk-{i}", _vec(float(i) * 0.1), {"i": i})
    results = store.query(_vec(0.0), top_k=2)
    assert len(results) == 2


def test_filter_metadata(store):
    store.upsert("c1", _vec(0.1), {"doc_id": "book-a"})
    store.upsert("c2", _vec(0.2), {"doc_id": "book-b"})
    results = store.query(_vec(0.1), filter_metadata={"doc_id": "book-a"})
    assert all(r.metadata["doc_id"] == "book-a" for r in results)


def test_delete(store):
    store.upsert("x", _vec(0.5), {})
    store.delete("x")
    assert store.query(_vec(0.5)) == []


def test_delete_where(store):
    store.upsert("c1", _vec(0.1), {"doc_id": "book-a"})
    store.upsert("c2", _vec(0.2), {"doc_id": "book-a"})
    store.upsert("c3", _vec(0.3), {"doc_id": "book-b"})
    count = store.delete_where({"doc_id": "book-a"})
    assert count == 2
    assert len(store.query(_vec(0.1))) == 1


def test_delete_nonexistent_is_noop(store):
    store.delete("does-not-exist")  # should not raise


def test_empty_query_returns_empty(store):
    assert store.query(_vec(0.1)) == []


def test_delete_where_raises_on_empty_filter(store):
    store.upsert("c1", _vec(0.1), {"doc_id": "book-a"})
    with pytest.raises(ValueError, match="empty"):
        store.delete_where({})
