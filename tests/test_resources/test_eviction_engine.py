import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from circuitforge_core.resources.coordinator.eviction_engine import EvictionEngine
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager


@pytest.fixture
def lease_manager():
    mgr = LeaseManager()
    mgr.register_gpu("heimdall", 0, 8192)
    return mgr


@pytest.fixture
def engine(lease_manager):
    return EvictionEngine(lease_manager=lease_manager, eviction_timeout_s=0.1)


@pytest.mark.asyncio
async def test_request_lease_grants_when_vram_available(engine, lease_manager):
    lease = await engine.request_lease(
        node_id="heimdall", gpu_id=0, mb=4096,
        service="peregrine", priority=1,
        agent_url="http://localhost:7701",
    )
    assert lease is not None
    assert lease.mb_granted == 4096


@pytest.mark.asyncio
async def test_request_lease_evicts_and_grants(engine, lease_manager):
    # Pre-fill with a low-priority lease
    big_lease = await lease_manager.try_grant(
        "heimdall", 0, 7000, "comfyui", priority=4
    )
    assert big_lease is not None

    # Mock the agent eviction call
    with patch(
        "circuitforge_core.resources.coordinator.eviction_engine.EvictionEngine._call_agent_evict",
        new_callable=AsyncMock,
    ) as mock_evict:
        mock_evict.return_value = True
        # Simulate the comfyui lease being released (as if the agent evicted it)
        asyncio.get_event_loop().call_later(
            0.05, lambda: asyncio.ensure_future(lease_manager.release(big_lease.lease_id))
        )
        lease = await engine.request_lease(
            node_id="heimdall", gpu_id=0, mb=4096,
            service="peregrine", priority=1,
            agent_url="http://localhost:7701",
        )
    assert lease is not None
    assert lease.holder_service == "peregrine"


@pytest.mark.asyncio
async def test_request_lease_returns_none_when_no_eviction_candidates(engine):
    await engine.lease_manager.try_grant("heimdall", 0, 6000, "vllm", priority=1)
    # Requesting 4GB but no lower-priority leases exist
    lease = await engine.request_lease(
        node_id="heimdall", gpu_id=0, mb=4096,
        service="kiwi", priority=2,
        agent_url="http://localhost:7701",
    )
    assert lease is None
