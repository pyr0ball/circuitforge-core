import pytest
from unittest.mock import MagicMock
from pathlib import Path
from fastapi.testclient import TestClient
from circuitforge_core.resources.coordinator.app import create_coordinator_app
from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.models import GpuInfo, NodeInfo
from circuitforge_core.resources.profiles.schema import load_profile


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


def test_dashboard_serves_html(coordinator_client):
    """GET / returns the dashboard HTML page."""
    client, _ = coordinator_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # Verify key structural markers are present (without asserting exact markup)
    assert "cf-orch" in resp.text
    assert "/api/nodes" in resp.text
    assert "/api/leases" in resp.text


def test_online_agents_excludes_offline():
    lm = LeaseManager()
    sup = AgentSupervisor(lm)
    sup.register("online_node", "http://a:7701")
    sup.register("offline_node", "http://b:7701")
    sup._agents["online_node"].online = True
    sup._agents["offline_node"].online = False
    result = sup.online_agents()
    assert "online_node" in result
    assert "offline_node" not in result


def test_resident_keys_returns_set_of_node_service():
    lm = LeaseManager()
    lm.set_residents_for_node("heimdall", [("vllm", "Ouro-1.4B"), ("ollama", None)])
    keys = lm.resident_keys()
    assert keys == {"heimdall:vllm", "heimdall:ollama"}


def test_single_gpu_8gb_profile_has_idle_stop_after_s():
    profile = load_profile(
        Path("circuitforge_core/resources/profiles/public/single-gpu-8gb.yaml")
    )
    vllm_svc = profile.services.get("vllm")
    assert vllm_svc is not None
    assert hasattr(vllm_svc, "idle_stop_after_s")
    assert vllm_svc.idle_stop_after_s == 600
