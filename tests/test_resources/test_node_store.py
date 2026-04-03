# tests/test_resources/test_node_store.py
"""Unit tests for NodeStore — SQLite persistence layer for known agent nodes."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from circuitforge_core.resources.coordinator.node_store import NodeStore


@pytest.fixture
def store(tmp_path: Path) -> NodeStore:
    return NodeStore(db_path=tmp_path / "test-nodes.db")


def test_upsert_and_all(store: NodeStore) -> None:
    store.upsert("heimdall", "http://127.0.0.1:7701")
    rows = store.all()
    assert len(rows) == 1
    assert rows[0] == ("heimdall", "http://127.0.0.1:7701")


def test_upsert_updates_url(store: NodeStore) -> None:
    store.upsert("navi", "http://10.1.10.10:7701")
    store.upsert("navi", "http://10.1.10.10:7702")
    rows = store.all()
    assert len(rows) == 1
    assert rows[0][1] == "http://10.1.10.10:7702"


def test_multiple_nodes(store: NodeStore) -> None:
    store.upsert("heimdall", "http://127.0.0.1:7701")
    store.upsert("navi", "http://10.1.10.10:7701")
    store.upsert("strahl", "http://10.1.10.20:7701")
    assert len(store.all()) == 3


def test_remove(store: NodeStore) -> None:
    store.upsert("heimdall", "http://127.0.0.1:7701")
    store.upsert("navi", "http://10.1.10.10:7701")
    store.remove("navi")
    ids = [r[0] for r in store.all()]
    assert "navi" not in ids
    assert "heimdall" in ids


def test_prune_stale_removes_old_entries(store: NodeStore) -> None:
    # Insert a node with a last_seen in the distant past
    store._conn.execute(
        "INSERT INTO known_nodes (node_id, agent_url, last_seen) VALUES (?, ?, ?)",
        ("ghost", "http://dead:7701", time.time() - 40 * 86400),
    )
    store._conn.commit()
    store.upsert("live", "http://live:7701")

    removed = store.prune_stale(max_age_days=30)
    assert removed == 1
    ids = [r[0] for r in store.all()]
    assert "ghost" not in ids
    assert "live" in ids


def test_prune_stale_keeps_recent(store: NodeStore) -> None:
    store.upsert("recent", "http://recent:7701")
    removed = store.prune_stale(max_age_days=30)
    assert removed == 0
    assert len(store.all()) == 1


def test_all_empty(store: NodeStore) -> None:
    assert store.all() == []


def test_db_persists_across_instances(tmp_path: Path) -> None:
    """Data written by one NodeStore instance is visible to a new one on the same file."""
    db = tmp_path / "shared.db"
    s1 = NodeStore(db_path=db)
    s1.upsert("navi", "http://10.1.10.10:7701")
    s1.close()

    s2 = NodeStore(db_path=db)
    rows = s2.all()
    assert len(rows) == 1
    assert rows[0][0] == "navi"
    s2.close()
