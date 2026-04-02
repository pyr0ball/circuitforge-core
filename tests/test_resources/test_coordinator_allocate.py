import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from circuitforge_core.resources.coordinator.app import create_coordinator_app
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.coordinator.agent_supervisor import AgentRecord
from circuitforge_core.resources.models import GpuInfo, NodeInfo


def _make_supervisor_mock(online: bool = True):
    sup = MagicMock()
    record = AgentRecord(node_id="heimdall", agent_url="http://heimdall:7701")
    record.gpus = [GpuInfo(0, "RTX 4000", 8192, 0, 8192)]
    record.online = online
    sup.online_agents.return_value = {"heimdall": record} if online else {}
    sup.get_node_info.return_value = NodeInfo(
        node_id="heimdall",
        agent_url="http://heimdall:7701",
        gpus=record.gpus,
        last_heartbeat=0.0,
    )
    return sup


@pytest.fixture
def alloc_client():
    lm = LeaseManager()
    pr = ProfileRegistry()
    sup = _make_supervisor_mock()
    app = create_coordinator_app(lease_manager=lm, profile_registry=pr, agent_supervisor=sup)
    return TestClient(app), sup


def test_allocate_returns_allocation_id_and_url(alloc_client):
    client, sup = alloc_client
    with patch("httpx.AsyncClient") as mock_http:
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_resp.json.return_value = {"running": True, "url": "http://heimdall:8000"}
        mock_http.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)

        resp = client.post("/api/services/vllm/allocate", json={
            "model_candidates": ["Ouro-1.4B"],
            "ttl_s": 300.0,
            "caller": "test",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert "allocation_id" in data
    assert data["service"] == "vllm"
    assert data["node_id"] == "heimdall"
    assert data["url"] == "http://heimdall:8000"


def test_allocate_returns_503_when_no_online_nodes(alloc_client):
    client, sup = alloc_client
    sup.online_agents.return_value = {}
    resp = client.post("/api/services/vllm/allocate", json={"model_candidates": ["Ouro-1.4B"]})
    assert resp.status_code == 503


def test_allocate_returns_422_for_empty_candidates(alloc_client):
    client, _ = alloc_client
    resp = client.post("/api/services/vllm/allocate", json={"model_candidates": []})
    assert resp.status_code == 422


def test_allocate_returns_422_for_unknown_service(alloc_client):
    client, _ = alloc_client
    resp = client.post("/api/services/cf-made-up/allocate", json={"model_candidates": ["x"]})
    assert resp.status_code == 422
