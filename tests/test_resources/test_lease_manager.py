import asyncio
import pytest
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager


@pytest.fixture
def mgr():
    m = LeaseManager()
    m.register_gpu(node_id="heimdall", gpu_id=0, total_mb=8192)
    return m


@pytest.mark.asyncio
async def test_grant_succeeds_when_vram_available(mgr):
    lease = await mgr.try_grant(
        node_id="heimdall", gpu_id=0, mb=4096,
        service="peregrine", priority=1
    )
    assert lease is not None
    assert lease.mb_granted == 4096
    assert lease.node_id == "heimdall"
    assert lease.gpu_id == 0


@pytest.mark.asyncio
async def test_grant_fails_when_vram_insufficient(mgr):
    await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=7000,
                         service="vllm", priority=1)
    lease = await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=2000,
                                 service="kiwi", priority=2)
    assert lease is None


@pytest.mark.asyncio
async def test_release_frees_vram(mgr):
    lease = await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=7000,
                                 service="vllm", priority=1)
    assert lease is not None
    released = await mgr.release(lease.lease_id)
    assert released is True
    lease2 = await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=7000,
                                   service="comfyui", priority=4)
    assert lease2 is not None


@pytest.mark.asyncio
async def test_release_unknown_lease_returns_false(mgr):
    result = await mgr.release("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_eviction_candidates_returns_lower_priority_leases(mgr):
    await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=3000,
                         service="comfyui", priority=4)
    await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=2000,
                         service="ollama", priority=1)
    candidates = mgr.get_eviction_candidates(
        node_id="heimdall", gpu_id=0,
        needed_mb=3000, requester_priority=2
    )
    assert len(candidates) == 1
    assert candidates[0].holder_service == "comfyui"


@pytest.mark.asyncio
async def test_list_leases_for_gpu(mgr):
    await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=1024,
                         service="peregrine", priority=1)
    await mgr.try_grant(node_id="heimdall", gpu_id=0, mb=512,
                         service="kiwi", priority=2)
    leases = mgr.list_leases(node_id="heimdall", gpu_id=0)
    assert len(leases) == 2


def test_register_gpu_sets_total(mgr):
    assert mgr.gpu_total_mb("heimdall", 0) == 8192


def test_used_mb_tracks_grants():
    mgr = LeaseManager()
    mgr.register_gpu("heimdall", 0, 8192)
    asyncio.run(mgr.try_grant("heimdall", 0, 3000, "a", 1))
    asyncio.run(mgr.try_grant("heimdall", 0, 2000, "b", 2))
    assert mgr.used_mb("heimdall", 0) == 5000
