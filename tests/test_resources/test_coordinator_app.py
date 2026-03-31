import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from circuitforge_core.resources.coordinator.app import create_coordinator_app
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.models import GpuInfo, NodeInfo


@pytest.fixture
def coordinator_client():
    lease_manager = LeaseManager()
    lease_manager.register_gpu("heimdall", 0, 8192)
    profile_registry = ProfileRegistry()
    supervisor = MagicMock()
    supervisor.all_nodes.return_value = [
        NodeInfo(
            node_id="heimdall",
            agent_url="http://localhost:7701",
            gpus=[GpuInfo(gpu_id=0, name="RTX 4000",
                          vram_total_mb=8192, vram_used_mb=0, vram_free_mb=8192)],
            last_heartbeat=0.0,
        )
    ]
    supervisor.get_node_info.return_value = NodeInfo(
        node_id="heimdall",
        agent_url="http://localhost:7701",
        gpus=[],
        last_heartbeat=0.0,
    )
    app = create_coordinator_app(
        lease_manager=lease_manager,
        profile_registry=profile_registry,
        agent_supervisor=supervisor,
    )
    return TestClient(app), lease_manager


def test_health_returns_ok(coordinator_client):
    client, _ = coordinator_client
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_get_nodes_returns_list(coordinator_client):
    client, _ = coordinator_client
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "heimdall"


def test_get_profiles_returns_public_profiles(coordinator_client):
    client, _ = coordinator_client
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()["profiles"]]
    assert "single-gpu-8gb" in names


def test_post_lease_grants_lease(coordinator_client):
    client, _ = coordinator_client
    resp = client.post("/api/leases", json={
        "node_id": "heimdall", "gpu_id": 0,
        "mb": 2048, "service": "peregrine", "priority": 1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["lease"]["mb_granted"] == 2048
    assert data["lease"]["holder_service"] == "peregrine"
    assert "lease_id" in data["lease"]


def test_delete_lease_releases_it(coordinator_client):
    client, _ = coordinator_client
    resp = client.post("/api/leases", json={
        "node_id": "heimdall", "gpu_id": 0,
        "mb": 2048, "service": "peregrine", "priority": 1,
    })
    lease_id = resp.json()["lease"]["lease_id"]
    del_resp = client.delete(f"/api/leases/{lease_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["released"] is True


def test_delete_unknown_lease_returns_404(coordinator_client):
    client, _ = coordinator_client
    resp = client.delete("/api/leases/nonexistent-id")
    assert resp.status_code == 404


def test_get_leases_returns_active_leases(coordinator_client):
    client, _ = coordinator_client
    client.post("/api/leases", json={
        "node_id": "heimdall", "gpu_id": 0,
        "mb": 1024, "service": "kiwi", "priority": 2,
    })
    resp = client.get("/api/leases")
    assert resp.status_code == 200
    assert len(resp.json()["leases"]) == 1
