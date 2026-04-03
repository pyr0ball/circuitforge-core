import pytest
from circuitforge_core.resources.coordinator.node_selector import select_node
from circuitforge_core.resources.coordinator.agent_supervisor import AgentRecord
from circuitforge_core.resources.models import GpuInfo
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry


def _make_agent(node_id: str, free_mb: int, online: bool = True) -> AgentRecord:
    r = AgentRecord(node_id=node_id, agent_url=f"http://{node_id}:7701")
    r.gpus = [GpuInfo(gpu_id=0, name="RTX", vram_total_mb=8192,
                      vram_used_mb=8192 - free_mb, vram_free_mb=free_mb)]
    r.online = online
    return r


def test_selects_node_with_most_free_vram():
    agents = {
        "a": _make_agent("a", free_mb=2000),
        "b": _make_agent("b", free_mb=6000),
    }
    registry = ProfileRegistry()
    result = select_node(agents, "vllm", registry, resident_keys=set())
    assert result == ("b", 0)


def test_prefers_warm_node_even_with_less_free_vram():
    agents = {
        "a": _make_agent("a", free_mb=2000),
        "b": _make_agent("b", free_mb=6000),
    }
    registry = ProfileRegistry()
    result = select_node(agents, "vllm", registry, resident_keys={"a:vllm"})
    assert result == ("a", 0)


def test_excludes_offline_nodes():
    agents = {
        "a": _make_agent("a", free_mb=8000, online=False),
        "b": _make_agent("b", free_mb=2000, online=True),
    }
    registry = ProfileRegistry()
    result = select_node(agents, "vllm", registry, resident_keys=set())
    assert result == ("b", 0)


def test_returns_none_when_no_node_has_profile_for_service():
    agents = {"a": _make_agent("a", free_mb=8000)}
    registry = ProfileRegistry()
    result = select_node(agents, "cf-nonexistent-service", registry, resident_keys=set())
    assert result is None


def test_returns_none_when_no_agents():
    registry = ProfileRegistry()
    result = select_node({}, "vllm", registry, resident_keys=set())
    assert result is None


def test_prefers_node_that_fully_fits_service_over_one_that_does_not():
    """can_fit requires free_mb >= service max_mb (full ceiling, not half).
    9500 MB guarantees above all profile ceilings (max is 9000); 1000 MB is below all.
    """
    agents = {
        "a": _make_agent("a", free_mb=1000),
        "b": _make_agent("b", free_mb=9500),
    }
    registry = ProfileRegistry()
    result = select_node(agents, "vllm", registry, resident_keys=set())
    # "b" is the only node in the preferred (can_fit) pool
    assert result == ("b", 0)


def test_falls_back_to_best_effort_when_no_node_fully_fits():
    """When nothing can_fit, select_node returns the best-VRAM node as fallback."""
    agents = {
        "a": _make_agent("a", free_mb=1000),
        "b": _make_agent("b", free_mb=2000),
    }
    registry = ProfileRegistry()
    # Neither has enough free VRAM; fallback picks highest effective_free_mb
    result = select_node(agents, "vllm", registry, resident_keys=set())
    assert result == ("b", 0)
