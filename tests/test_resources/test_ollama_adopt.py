# tests/test_resources/test_ollama_adopt.py
"""
Tests for the Ollama adopt-if-running path:
  - ProcessSpec: adopt and health_path fields parsed from YAML
  - ServiceManager.start(): adopt path claims running service; falls through if not running
  - ServiceManager.is_running(): adopt path uses health probe, not proc table
  - ServiceInstance.health_path persists through upsert_instance
  - Probe loop uses inst.health_path instead of hardcoded /health
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.resources.agent.service_manager import ServiceManager
from circuitforge_core.resources.coordinator.service_registry import ServiceRegistry
from circuitforge_core.resources.profiles.schema import GpuProfile, ProcessSpec, ServiceProfile, load_profile


# ── ProcessSpec schema ────────────────────────────────────────────────────────

def test_process_spec_defaults():
    spec = ProcessSpec(exec_path="/usr/local/bin/ollama")
    assert spec.adopt is False
    assert spec.health_path == "/health"


def test_process_spec_adopt_fields():
    spec = ProcessSpec(
        exec_path="/usr/local/bin/ollama",
        adopt=True,
        health_path="/api/tags",
        port=11434,
        host_port=11434,
    )
    assert spec.adopt is True
    assert spec.health_path == "/api/tags"


def test_profile_yaml_parses_adopt(tmp_path: Path):
    yaml_text = """\
schema_version: 1
name: test
services:
  ollama:
    max_mb: 4096
    priority: 1
    managed:
      type: process
      adopt: true
      exec_path: /usr/local/bin/ollama
      args_template: serve
      port: 11434
      host_port: 11434
      health_path: /api/tags
"""
    p = tmp_path / "profile.yaml"
    p.write_text(yaml_text)
    profile = load_profile(p)
    spec = profile.services["ollama"].managed
    assert isinstance(spec, ProcessSpec)
    assert spec.adopt is True
    assert spec.health_path == "/api/tags"
    assert spec.host_port == 11434


# ── ServiceManager adopt path ─────────────────────────────────────────────────

def _make_manager_with_ollama(advertise_host: str = "127.0.0.1") -> ServiceManager:
    profile = GpuProfile(
        schema_version=1,
        name="test",
        services={
            "ollama": ServiceProfile(
                max_mb=4096,
                priority=1,
                managed=ProcessSpec(
                    exec_path="/usr/local/bin/ollama",
                    args_template="serve",
                    port=11434,
                    host_port=11434,
                    adopt=True,
                    health_path="/api/tags",
                ),
            )
        },
    )
    return ServiceManager(node_id="heimdall", profile=profile, advertise_host=advertise_host)


def test_start_adopt_claims_running_service():
    """When Ollama is already healthy, start() returns its URL without spawning a process."""
    mgr = _make_manager_with_ollama()
    with patch.object(mgr, "_probe_health", return_value=True) as mock_probe:
        url = mgr.start("ollama", gpu_id=0, params={})
    assert url == "http://127.0.0.1:11434"
    mock_probe.assert_called_once_with(11434, "/api/tags")
    assert "ollama" not in mgr._procs  # no subprocess spawned


def test_start_adopt_spawns_when_not_running():
    """When Ollama is not yet running, start() spawns it normally."""
    mgr = _make_manager_with_ollama()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with patch.object(mgr, "_probe_health", return_value=False), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        url = mgr.start("ollama", gpu_id=0, params={})

    assert url == "http://127.0.0.1:11434"
    mock_popen.assert_called_once()
    assert "ollama" in mgr._procs


def test_is_running_adopt_uses_health_probe():
    """is_running() for adopt=True services checks the health endpoint, not the proc table."""
    mgr = _make_manager_with_ollama()
    with patch.object(mgr, "_probe_health", return_value=True):
        assert mgr.is_running("ollama") is True
    with patch.object(mgr, "_probe_health", return_value=False):
        assert mgr.is_running("ollama") is False


def test_probe_health_returns_true_on_200():
    mgr = _make_manager_with_ollama()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: mock_resp
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert mgr._probe_health(11434, "/api/tags") is True


def test_probe_health_returns_false_on_connection_error():
    mgr = _make_manager_with_ollama()
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert mgr._probe_health(11434, "/api/tags") is False


# ── ServiceRegistry health_path ───────────────────────────────────────────────

def test_upsert_instance_stores_health_path():
    reg = ServiceRegistry()
    inst = reg.upsert_instance(
        service="ollama", node_id="heimdall", gpu_id=0,
        state="running", model=None, url="http://127.0.0.1:11434",
        health_path="/api/tags",
    )
    assert inst.health_path == "/api/tags"


def test_upsert_instance_default_health_path():
    reg = ServiceRegistry()
    inst = reg.upsert_instance(
        service="vllm", node_id="heimdall", gpu_id=0,
        state="starting", model="qwen", url="http://127.0.0.1:8000",
    )
    assert inst.health_path == "/health"


def test_all_gpu_profiles_have_ollama_managed_block():
    """Sanity check: all public GPU profiles now have a managed block for ollama."""
    from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
    registry = ProfileRegistry()
    for profile in registry.list_public():
        svc = profile.services.get("ollama")
        if svc is None:
            continue  # profile may not define ollama
        assert svc.managed is not None, f"{profile.name}: ollama missing managed block"
        assert isinstance(svc.managed, ProcessSpec)
        assert svc.managed.adopt is True, f"{profile.name}: ollama adopt should be True"
        assert svc.managed.health_path == "/api/tags", f"{profile.name}: wrong health_path"
