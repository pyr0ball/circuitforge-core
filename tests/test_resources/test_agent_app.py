from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from circuitforge_core.resources.agent.app import create_agent_app
from circuitforge_core.resources.models import GpuInfo
from circuitforge_core.resources.agent.eviction_executor import EvictionResult

MOCK_GPUS = [
    GpuInfo(
        gpu_id=0,
        name="RTX 4000",
        vram_total_mb=8192,
        vram_used_mb=1024,
        vram_free_mb=7168,
    ),
]


@pytest.fixture
def agent_client():
    mock_monitor = MagicMock()
    mock_monitor.poll.return_value = MOCK_GPUS
    mock_executor = MagicMock()
    app = create_agent_app(
        node_id="heimdall",
        monitor=mock_monitor,
        executor=mock_executor,
    )
    return TestClient(app), mock_monitor, mock_executor


def test_health_returns_ok(agent_client):
    client, _, _ = agent_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["node_id"] == "heimdall"


def test_gpu_info_returns_gpu_list(agent_client):
    client, _, _ = agent_client
    resp = client.get("/gpu-info")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["gpus"]) == 1
    assert data["gpus"][0]["gpu_id"] == 0
    assert data["gpus"][0]["name"] == "RTX 4000"
    assert data["gpus"][0]["vram_free_mb"] == 7168


def test_evict_calls_executor(agent_client):
    client, _, mock_executor = agent_client
    mock_executor.evict_pid.return_value = EvictionResult(
        success=True, method="sigterm", message="done"
    )
    resp = client.post("/evict", json={"pid": 1234, "grace_period_s": 5.0})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    mock_executor.evict_pid.assert_called_once_with(pid=1234, grace_period_s=5.0)


def test_evict_requires_pid(agent_client):
    client, _, _ = agent_client
    resp = client.post("/evict", json={"grace_period_s": 5.0})
    assert resp.status_code == 422
