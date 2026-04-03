# tests/test_hardware/test_generator.py
"""Tests for the LLMConfig profile generator."""
import pytest

from circuitforge_core.hardware.generator import generate_profile
from circuitforge_core.hardware.models import HardwareSpec


def _spec(vram_mb: int, gpu_vendor: str = "nvidia") -> HardwareSpec:
    return HardwareSpec(
        vram_mb=vram_mb, ram_mb=32768, gpu_count=1 if vram_mb > 0 else 0,
        gpu_vendor=gpu_vendor,
    )


class TestGenerateProfile:
    def test_cpu_only_no_vllm_backend(self):
        config = generate_profile(_spec(0, "cpu"))
        assert config.profile_name == "cpu-16gb"
        assert "vllm" not in config.backends
        assert config.backends["ollama"].enabled

    def test_cpu_only_fallback_order_ollama_only(self):
        config = generate_profile(_spec(0, "cpu"))
        assert config.fallback_order == ["ollama"]

    def test_6gb_has_vllm_backend(self):
        config = generate_profile(_spec(6144))
        assert "vllm" in config.backends
        assert config.backends["vllm"].enabled
        assert config.backends["vllm"].model_candidates

    def test_6gb_has_vision_service(self):
        config = generate_profile(_spec(6144))
        assert "vision_service" in config.backends

    def test_6gb_has_docuvision_service(self):
        config = generate_profile(_spec(6144))
        assert "docuvision_service" in config.backends

    def test_4gb_no_docuvision(self):
        config = generate_profile(_spec(4096))
        assert "docuvision_service" not in config.backends

    def test_vllm_tier_fallback_order_prefers_vllm(self):
        config = generate_profile(_spec(8192))
        assert config.fallback_order[0] == "vllm"

    def test_16gb_profile_name(self):
        config = generate_profile(_spec(16384))
        assert config.profile_name == "single-gpu-16gb"

    def test_to_dict_roundtrip(self):
        config = generate_profile(_spec(8192))
        d = config.to_dict()
        assert "backends" in d
        assert "fallback_order" in d
        assert "ollama" in d["backends"]
        assert d["backends"]["ollama"]["enabled"] is True

    def test_custom_ollama_url(self):
        config = generate_profile(_spec(8192), ollama_url="http://10.0.0.1:11434")
        assert config.backends["ollama"].url == "http://10.0.0.1:11434"

    def test_vllm_candidates_populated(self):
        config = generate_profile(_spec(8192))
        assert len(config.backends["vllm"].model_candidates) >= 1

    def test_vision_fallback_includes_vision_service(self):
        config = generate_profile(_spec(8192))
        assert "vision_service" in config.vision_fallback_order
