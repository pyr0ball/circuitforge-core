# tests/test_resources/test_profile_registry.py
import pytest
from circuitforge_core.resources.profiles.schema import (
    GpuProfile, ServiceProfile, load_profile
)


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
