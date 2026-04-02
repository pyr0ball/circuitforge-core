import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.service_registry import ServiceRegistry, ServiceInstance


def test_build_idle_stop_config_empty_without_registry():
    lm = LeaseManager()
    supervisor = AgentSupervisor(lease_manager=lm)
    assert supervisor._build_idle_stop_config() == {}


def test_build_idle_stop_config_from_profiles():
    lm = LeaseManager()
    mock_svc = MagicMock()
    mock_svc.idle_stop_after_s = 600
    mock_profile = MagicMock()
    mock_profile.services = {"vllm": mock_svc}
    mock_profile_registry = MagicMock()
    mock_profile_registry.list_public.return_value = [mock_profile]

    supervisor = AgentSupervisor(lease_manager=lm, profile_registry=mock_profile_registry)
    config = supervisor._build_idle_stop_config()
    assert config == {"vllm": 600}


@pytest.mark.asyncio
async def test_run_idle_sweep_posts_stop():
    lm = LeaseManager()
    service_registry = ServiceRegistry()

    # Upsert instance as running, then allocate + release to transition it to idle
    service_registry.upsert_instance(
        service="vllm",
        node_id="heimdall",
        gpu_id=0,
        state="running",
        model="test-model",
        url="http://heimdall:8000",
    )
    alloc = service_registry.allocate(
        service="vllm",
        node_id="heimdall",
        gpu_id=0,
        model="test-model",
        url="http://heimdall:8000",
        caller="test",
        ttl_s=300.0,
    )
    service_registry.release(alloc.allocation_id)

    # Backdate idle_since so it exceeds the timeout
    import dataclasses
    key = "vllm:heimdall:0"
    inst = service_registry._instances[key]
    service_registry._instances[key] = dataclasses.replace(inst, idle_since=time.time() - 700)

    mock_profile_registry = MagicMock()
    mock_svc = MagicMock()
    mock_svc.idle_stop_after_s = 600
    mock_profile = MagicMock()
    mock_profile.services = {"vllm": mock_svc}
    mock_profile_registry.list_public.return_value = [mock_profile]

    supervisor = AgentSupervisor(
        lease_manager=lm,
        service_registry=service_registry,
        profile_registry=mock_profile_registry,
    )
    supervisor.register("heimdall", "http://heimdall:7701")

    posted_urls = []

    async def fake_http_post(url: str) -> bool:
        posted_urls.append(url)
        return True

    supervisor._http_post = fake_http_post
    await supervisor._run_idle_sweep()

    assert len(posted_urls) == 1
    assert posted_urls[0] == "http://heimdall:7701/services/vllm/stop"


@pytest.mark.asyncio
async def test_run_idle_sweep_skips_without_registry():
    lm = LeaseManager()
    supervisor = AgentSupervisor(lease_manager=lm)
    # Should return immediately without error
    await supervisor._run_idle_sweep()
