import time
from circuitforge_core.resources.models import VRAMLease, GpuInfo, NodeInfo


def test_vram_lease_create_assigns_unique_ids():
    lease_a = VRAMLease.create(gpu_id=0, node_id="heimdall", mb=4096,
                                service="peregrine", priority=1)
    lease_b = VRAMLease.create(gpu_id=0, node_id="heimdall", mb=4096,
                                service="peregrine", priority=1)
    assert lease_a.lease_id != lease_b.lease_id


def test_vram_lease_create_with_ttl_sets_expiry():
    before = time.time()
    lease = VRAMLease.create(gpu_id=0, node_id="heimdall", mb=2048,
                              service="kiwi", priority=2, ttl_s=60.0)
    after = time.time()
    assert before + 60.0 <= lease.expires_at <= after + 60.0


def test_vram_lease_create_no_ttl_has_zero_expiry():
    lease = VRAMLease.create(gpu_id=0, node_id="heimdall", mb=1024,
                              service="snipe", priority=2)
    assert lease.expires_at == 0.0


def test_vram_lease_is_immutable():
    lease = VRAMLease.create(gpu_id=0, node_id="heimdall", mb=1024,
                              service="snipe", priority=2)
    import pytest
    with pytest.raises((AttributeError, TypeError)):
        lease.mb_granted = 999  # type: ignore


def test_gpu_info_fields():
    info = GpuInfo(gpu_id=0, name="RTX 4000", vram_total_mb=8192,
                   vram_used_mb=2048, vram_free_mb=6144)
    assert info.vram_free_mb == 6144


def test_node_info_fields():
    gpu = GpuInfo(gpu_id=0, name="RTX 4000", vram_total_mb=8192,
                  vram_used_mb=0, vram_free_mb=8192)
    node = NodeInfo(node_id="heimdall", agent_url="http://localhost:7701",
                    gpus=[gpu], last_heartbeat=time.time())
    assert node.node_id == "heimdall"
    assert len(node.gpus) == 1
