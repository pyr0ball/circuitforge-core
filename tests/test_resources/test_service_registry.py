import time
import dataclasses
import pytest
from circuitforge_core.resources.coordinator.service_registry import (
    ServiceRegistry, ServiceAllocation, ServiceInstance,
)


@pytest.fixture
def registry():
    return ServiceRegistry()


def test_allocate_creates_allocation(registry):
    alloc = registry.allocate(
        service="vllm", node_id="heimdall", gpu_id=0,
        model="Ouro-1.4B", url="http://heimdall:8000",
        caller="test", ttl_s=300.0,
    )
    assert alloc.service == "vllm"
    assert alloc.node_id == "heimdall"
    assert alloc.allocation_id  # non-empty UUID string


def test_active_allocations_count(registry):
    registry.allocate("vllm", "heimdall", 0, "M", "http://h:8000", "a", 300.0)
    registry.allocate("vllm", "heimdall", 0, "M", "http://h:8000", "b", 300.0)
    assert registry.active_allocations("vllm", "heimdall", 0) == 2


def test_release_decrements_count(registry):
    alloc = registry.allocate("vllm", "heimdall", 0, "M", "http://h:8000", "a", 300.0)
    registry.release(alloc.allocation_id)
    assert registry.active_allocations("vllm", "heimdall", 0) == 0


def test_release_nonexistent_returns_false(registry):
    assert registry.release("nonexistent-id") is False


def test_upsert_instance_sets_running_state(registry):
    registry.upsert_instance("vllm", "heimdall", 0, state="running",
                              model="Ouro-1.4B", url="http://heimdall:8000")
    instances = registry.all_instances()
    assert len(instances) == 1
    assert instances[0].state == "running"


def test_release_last_alloc_marks_instance_idle(registry):
    registry.upsert_instance("vllm", "heimdall", 0, state="running",
                              model="Ouro-1.4B", url="http://heimdall:8000")
    alloc = registry.allocate("vllm", "heimdall", 0, "Ouro-1.4B", "http://heimdall:8000", "a", 300.0)
    registry.release(alloc.allocation_id)
    instance = registry.all_instances()[0]
    assert instance.state == "idle"
    assert instance.idle_since is not None


def test_new_alloc_on_idle_instance_marks_it_running(registry):
    registry.upsert_instance("vllm", "heimdall", 0, state="idle",
                              model="M", url="http://h:8000")
    registry.allocate("vllm", "heimdall", 0, "M", "http://h:8000", "x", 300.0)
    assert registry.all_instances()[0].state == "running"
