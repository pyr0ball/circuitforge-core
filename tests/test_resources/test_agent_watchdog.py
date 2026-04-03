# tests/test_resources/test_agent_watchdog.py
"""
Tests for AgentSupervisor watchdog behaviour:
  - restore_from_store() reloads known nodes from NodeStore on startup
  - register() persists to NodeStore
  - restored nodes start offline and come online after a successful poll
  - NodeStore=None path is a no-op (backwards compatibility)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.node_store import NodeStore


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> NodeStore:
    return NodeStore(db_path=tmp_path / "nodes.db")


@pytest.fixture
def supervisor(store: NodeStore) -> AgentSupervisor:
    return AgentSupervisor(lease_manager=LeaseManager(), node_store=store)


@pytest.fixture
def supervisor_no_store() -> AgentSupervisor:
    return AgentSupervisor(lease_manager=LeaseManager(), node_store=None)


# ── register() persists ───────────────────────────────────────────────────────

def test_register_persists_to_store(supervisor: AgentSupervisor, store: NodeStore) -> None:
    supervisor.register("heimdall", "http://127.0.0.1:7701")
    rows = store.all()
    assert len(rows) == 1
    assert rows[0] == ("heimdall", "http://127.0.0.1:7701")


def test_register_updates_url_in_store(supervisor: AgentSupervisor, store: NodeStore) -> None:
    supervisor.register("navi", "http://10.1.10.10:7701")
    supervisor.register("navi", "http://10.1.10.10:9999")
    rows = store.all()
    assert len(rows) == 1
    assert rows[0][1] == "http://10.1.10.10:9999"


def test_register_without_store_does_not_crash(supervisor_no_store: AgentSupervisor) -> None:
    supervisor_no_store.register("heimdall", "http://127.0.0.1:7701")
    assert supervisor_no_store.get_node_info("heimdall") is not None


# ── restore_from_store() ──────────────────────────────────────────────────────

def test_restore_loads_known_nodes(tmp_path: Path) -> None:
    """Nodes written by a previous supervisor session are restored into a fresh one."""
    db = tmp_path / "nodes.db"

    # Session 1: register two nodes
    s1 = NodeStore(db_path=db)
    sup1 = AgentSupervisor(lease_manager=LeaseManager(), node_store=s1)
    sup1.register("navi", "http://10.1.10.10:7701")
    sup1.register("strahl", "http://10.1.10.20:7701")

    # Session 2: fresh supervisor, same DB
    s2 = NodeStore(db_path=db)
    sup2 = AgentSupervisor(lease_manager=LeaseManager(), node_store=s2)
    restored = sup2.restore_from_store()

    assert restored == 2
    assert sup2.get_node_info("navi") is not None
    assert sup2.get_node_info("strahl") is not None


def test_restore_marks_nodes_offline(tmp_path: Path) -> None:
    """Restored nodes start offline — they haven't been polled yet."""
    db = tmp_path / "nodes.db"

    s1 = NodeStore(db_path=db)
    AgentSupervisor(lease_manager=LeaseManager(), node_store=s1).register(
        "navi", "http://10.1.10.10:7701"
    )

    s2 = NodeStore(db_path=db)
    sup2 = AgentSupervisor(lease_manager=LeaseManager(), node_store=s2)
    sup2.restore_from_store()

    assert sup2.online_agents() == {}


def test_restore_returns_zero_without_store() -> None:
    sup = AgentSupervisor(lease_manager=LeaseManager(), node_store=None)
    assert sup.restore_from_store() == 0


def test_restore_skips_already_registered(tmp_path: Path) -> None:
    """Nodes manually registered before restore_from_store() are not duplicated."""
    db = tmp_path / "nodes.db"
    store = NodeStore(db_path=db)
    store.upsert("heimdall", "http://127.0.0.1:7701")

    sup = AgentSupervisor(lease_manager=LeaseManager(), node_store=store)
    sup.register("heimdall", "http://127.0.0.1:7701")  # already in memory
    restored = sup.restore_from_store()

    assert restored == 0  # already present, not double-counted


# ── restored node comes online after poll ─────────────────────────────────────

@pytest.mark.asyncio
async def test_restored_node_comes_online_after_poll(tmp_path: Path) -> None:
    """After restore, a successful poll_agent() brings the node online."""
    db = tmp_path / "nodes.db"
    store = NodeStore(db_path=db)
    store.upsert("navi", "http://10.1.10.10:7701")

    sup = AgentSupervisor(lease_manager=LeaseManager(), node_store=store)
    sup.restore_from_store()

    # Stub poll_agent to succeed
    gpu_payload = {"gpus": [{"gpu_id": 0, "name": "RTX 4000",
                              "vram_total_mb": 8192, "vram_used_mb": 512, "vram_free_mb": 7680}]}
    resident_payload = {"residents": []}

    mock_resp_gpu = MagicMock()
    mock_resp_gpu.raise_for_status = MagicMock()
    mock_resp_gpu.json.return_value = gpu_payload

    mock_resp_res = MagicMock()
    mock_resp_res.is_success = True
    mock_resp_res.json.return_value = resident_payload

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[mock_resp_gpu, mock_resp_res])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("circuitforge_core.resources.coordinator.agent_supervisor.httpx.AsyncClient",
               return_value=mock_client):
        result = await sup.poll_agent("navi")

    assert result is True
    assert "navi" in sup.online_agents()
