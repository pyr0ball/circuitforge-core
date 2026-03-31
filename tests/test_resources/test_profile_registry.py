# tests/test_resources/test_profile_registry.py
import pytest
from unittest.mock import MagicMock

from circuitforge_core.resources.profiles.schema import (
    GpuProfile, ServiceProfile, load_profile
)
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry


def test_load_8gb_profile(tmp_path):
    yaml_content = """
schema_version: 1
name: single-gpu-8gb
vram_total_mb: 8192
eviction_timeout_s: 10.0
services:
  vllm:
    max_mb: 5120
    priority: 1
  cf-vision:
    max_mb: 2048
    priority: 2
    shared: true
    max_concurrent: 3
"""
    profile_file = tmp_path / "test.yaml"
    profile_file.write_text(yaml_content)
    profile = load_profile(profile_file)

    assert profile.name == "single-gpu-8gb"
    assert profile.schema_version == 1
    assert profile.vram_total_mb == 8192
    assert profile.eviction_timeout_s == 10.0
    assert "vllm" in profile.services
    assert profile.services["vllm"].max_mb == 5120
    assert profile.services["vllm"].priority == 1
    assert profile.services["cf-vision"].shared is True
    assert profile.services["cf-vision"].max_concurrent == 3


def test_load_profile_rejects_wrong_schema_version(tmp_path):
    yaml_content = "schema_version: 99\nname: future\n"
    profile_file = tmp_path / "future.yaml"
    profile_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="schema_version"):
        load_profile(profile_file)


def test_service_profile_defaults():
    svc = ServiceProfile(max_mb=1024, priority=2)
    assert svc.shared is False
    assert svc.max_concurrent == 1
    assert svc.always_on is False
    assert svc.backend is None
    assert svc.consumers == []


def test_profile_registry_loads_public_profiles():
    registry = ProfileRegistry()
    profiles = registry.list_public()
    names = [p.name for p in profiles]
    assert "single-gpu-8gb" in names
    assert "single-gpu-6gb" in names
    assert "single-gpu-2gb" in names


def test_profile_registry_auto_detect_selects_8gb():
    registry = ProfileRegistry()
    mock_gpus = [
        MagicMock(vram_total_mb=8192),
    ]
    profile = registry.auto_detect(mock_gpus)
    assert profile.name == "single-gpu-8gb"


def test_profile_registry_auto_detect_selects_6gb():
    registry = ProfileRegistry()
    mock_gpus = [MagicMock(vram_total_mb=6144)]
    profile = registry.auto_detect(mock_gpus)
    assert profile.name == "single-gpu-6gb"


def test_profile_registry_auto_detect_selects_2gb():
    registry = ProfileRegistry()
    mock_gpus = [MagicMock(vram_total_mb=2048)]
    profile = registry.auto_detect(mock_gpus)
    assert profile.name == "single-gpu-2gb"


def test_profile_registry_load_from_path(tmp_path):
    yaml_content = (
        "schema_version: 1\nname: custom\n"
        "vram_total_mb: 12288\neviction_timeout_s: 5.0\n"
    )
    p = tmp_path / "custom.yaml"
    p.write_text(yaml_content)
    registry = ProfileRegistry()
    profile = registry.load(p)
    assert profile.name == "custom"
    assert profile.vram_total_mb == 12288
