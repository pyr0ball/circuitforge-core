"""Integration test: full lease → eviction → re-grant cycle.

Runs coordinator in-process (no subprocesses, no real nvidia-smi).
Uses TestClient for HTTP, mocks AgentSupervisor to return fixed node state.
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.app import create_coordinator_app
from circuitforge_core.resources.models import GpuInfo, NodeInfo


@pytest.fixture
def system():
    """Create an in-process coordinator system with 8GB GPU and mock supervisor."""
    lease_manager = LeaseManager()
    lease_manager.register_gpu("local", 0, 8192)

    mock_supervisor = MagicMock(spec=AgentSupervisor)
    mock_supervisor.all_nodes.return_value = [
        NodeInfo(
            node_id="local",
            agent_url="http://localhost:7701",
            gpus=[GpuInfo(
                gpu_id=0,
                name="RTX 4000",
                vram_total_mb=8192,
                vram_used_mb=0,
                vram_free_mb=8192,
            )],
            last_heartbeat=0.0,
        )
    ]
    mock_supervisor.get_node_info.return_value = NodeInfo(
        node_id="local",
        agent_url="http://localhost:7701",
        gpus=[],
        last_heartbeat=0.0,
    )

    profile_registry = ProfileRegistry()
    app = create_coordinator_app(
        lease_manager=lease_manager,
        profile_registry=profile_registry,
        agent_supervisor=mock_supervisor,
    )
    client = TestClient(app)
    return client, lease_manager


def test_full_lease_cycle(system):
    """Test: grant, verify, release, verify gone."""
    client, _ = system

    # Grant a lease
    resp = client.post("/api/leases", json={
        "node_id": "local",
        "gpu_id": 0,
        "mb": 4096,
        "service": "peregrine",
        "priority": 1,
    })
    assert resp.status_code == 200
    lease_data = resp.json()["lease"]
    lease_id = lease_data["lease_id"]
    assert lease_data["mb_granted"] == 4096
    assert lease_data["holder_service"] == "peregrine"

    # Verify it appears in active leases
    resp = client.get("/api/leases")
    assert resp.status_code == 200
    leases = resp.json()["leases"]
    assert any(l["lease_id"] == lease_id for l in leases)

    # Release it
    resp = client.delete(f"/api/leases/{lease_id}")
    assert resp.status_code == 200
    assert resp.json()["released"] is True

    # Verify it's gone
    resp = client.get("/api/leases")
    assert resp.status_code == 200
    leases = resp.json()["leases"]
    assert not any(l["lease_id"] == lease_id for l in leases)


def test_vram_exhaustion_returns_503(system):
    """Test: fill GPU, then request with no eviction candidates returns 503."""
    client, _ = system

    # Fill GPU 0 with high-priority lease
    resp = client.post("/api/leases", json={
        "node_id": "local",
        "gpu_id": 0,
        "mb": 8000,
        "service": "vllm",
        "priority": 1,
    })
    assert resp.status_code == 200

    # Try to get more VRAM with same priority (no eviction candidates)
    resp = client.post("/api/leases", json={
        "node_id": "local",
        "gpu_id": 0,
        "mb": 2000,
        "service": "kiwi",
        "priority": 1,
    })
    assert resp.status_code == 503
    assert "Insufficient VRAM" in resp.json()["detail"]


def test_auto_detect_profile_for_8gb():
    """Test: ProfileRegistry auto-detects single-gpu-8gb for 8GB GPU."""
    registry = ProfileRegistry()
    gpu = GpuInfo(
        gpu_id=0,
        name="RTX 4000",
        vram_total_mb=8192,
        vram_used_mb=0,
        vram_free_mb=8192,
    )
    profile = registry.auto_detect([gpu])
    assert profile.name == "single-gpu-8gb"
    # Verify profile has services configured
    assert hasattr(profile, "services")


def test_node_endpoint_shows_nodes(system):
    """Test: GET /api/nodes returns the mocked node."""
    client, _ = system
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "local"
    assert nodes[0]["agent_url"] == "http://localhost:7701"
    assert len(nodes[0]["gpus"]) == 1
    assert nodes[0]["gpus"][0]["name"] == "RTX 4000"


def test_profiles_endpoint_returns_public_profiles(system):
    """Test: GET /api/profiles returns standard public profiles."""
    client, _ = system
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    profiles = resp.json()["profiles"]
    names = [p["name"] for p in profiles]
    # Verify common public profiles are present
    assert "single-gpu-8gb" in names
    assert "single-gpu-6gb" in names
    assert "single-gpu-2gb" in names


def test_multiple_leases_tracked_independently(system):
    """Test: multiple active leases are tracked correctly."""
    client, _ = system

    # Grant lease 1
    resp1 = client.post("/api/leases", json={
        "node_id": "local",
        "gpu_id": 0,
        "mb": 2048,
        "service": "peregrine",
        "priority": 2,
    })
    assert resp1.status_code == 200
    lease1_id = resp1.json()["lease"]["lease_id"]

    # Grant lease 2
    resp2 = client.post("/api/leases", json={
        "node_id": "local",
        "gpu_id": 0,
        "mb": 2048,
        "service": "kiwi",
        "priority": 2,
    })
    assert resp2.status_code == 200
    lease2_id = resp2.json()["lease"]["lease_id"]

    # Both should be in active leases
    resp = client.get("/api/leases")
    leases = resp.json()["leases"]
    lease_ids = [l["lease_id"] for l in leases]
    assert lease1_id in lease_ids
    assert lease2_id in lease_ids
    assert len(leases) == 2

    # Release lease 1
    resp = client.delete(f"/api/leases/{lease1_id}")
    assert resp.status_code == 200

    # Only lease 2 should remain
    resp = client.get("/api/leases")
    leases = resp.json()["leases"]
    lease_ids = [l["lease_id"] for l in leases]
    assert lease1_id not in lease_ids
    assert lease2_id in lease_ids
    assert len(leases) == 1


def test_delete_nonexistent_lease_returns_404(system):
    """Test: deleting a nonexistent lease returns 404."""
    client, _ = system
    resp = client.delete("/api/leases/nonexistent-lease-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_health_endpoint_returns_ok(system):
    """Test: GET /api/health returns status ok."""
    client, _ = system
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
