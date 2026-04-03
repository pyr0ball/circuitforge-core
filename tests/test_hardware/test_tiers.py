# tests/test_hardware/test_tiers.py
"""Tests for VRAM tier ladder selection."""
import pytest

from circuitforge_core.hardware.tiers import VRAM_TIERS, select_tier


class TestSelectTier:
    def test_zero_vram_returns_cpu_tier(self):
        tier = select_tier(0)
        assert tier.profile_name == "cpu-16gb"
        assert "vllm" not in tier.services

    def test_2gb_gpu(self):
        tier = select_tier(2048)
        assert tier.profile_name == "single-gpu-2gb"
        assert "ollama" in tier.services

    def test_4gb_gpu(self):
        tier = select_tier(4096)
        assert tier.profile_name == "single-gpu-4gb"
        assert "cf-vision" in tier.services
        assert "vllm" not in tier.services

    def test_6gb_gpu(self):
        tier = select_tier(6144)
        assert tier.profile_name == "single-gpu-6gb"
        assert "vllm" in tier.services
        assert "cf-docuvision" in tier.services

    def test_8gb_gpu(self):
        tier = select_tier(8192)
        assert tier.profile_name == "single-gpu-8gb"
        assert "vllm" in tier.services

    def test_16gb_gpu(self):
        tier = select_tier(16384)
        assert tier.profile_name == "single-gpu-16gb"
        assert "cf-embed" in tier.services
        assert "cf-classify" in tier.services

    def test_24gb_gpu(self):
        tier = select_tier(24576)
        assert tier.profile_name == "single-gpu-24gb"
        assert "comfyui" in tier.services

    def test_boundary_exact_6gb(self):
        """Exactly 6GB should land in the 6GB tier, not 4GB."""
        tier = select_tier(6000)
        assert tier.profile_name == "single-gpu-6gb"

    def test_boundary_just_below_6gb(self):
        tier = select_tier(4999)
        assert tier.profile_name == "single-gpu-4gb"

    def test_extremely_large_vram_returns_top_tier(self):
        tier = select_tier(80 * 1024)  # 80GB (H100)
        assert tier.profile_name == "single-gpu-24gb"

    def test_all_tiers_have_ollama(self):
        for t in VRAM_TIERS:
            assert "ollama" in t.services, f"{t.profile_name} missing ollama"

    def test_vllm_tiers_have_candidates(self):
        for t in VRAM_TIERS:
            if "vllm" in t.services:
                assert t.vllm_candidates, f"{t.profile_name} has vllm but no candidates"
