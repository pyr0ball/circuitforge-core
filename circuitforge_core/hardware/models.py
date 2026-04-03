# circuitforge_core/hardware/models.py
"""Data models for hardware detection and LLM configuration output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    """Describes a user's hardware as detected or manually entered."""

    vram_mb: int          # total VRAM per primary GPU (0 = CPU-only)
    ram_mb: int           # total system RAM
    gpu_count: int        # number of GPUs
    gpu_vendor: str       # "nvidia" | "amd" | "apple" | "cpu"
    gpu_name: str = ""    # human-readable card name, e.g. "RTX 4080"
    cuda_version: str = ""   # e.g. "12.4" (empty if not CUDA)
    rocm_version: str = ""   # e.g. "5.7" (empty if not ROCm)


@dataclass
class LLMBackendConfig:
    """Configuration for a single LLM backend."""

    enabled: bool
    url: str
    model: str = ""
    model_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"enabled": self.enabled, "url": self.url}
        if self.model:
            d["model"] = self.model
        if self.model_candidates:
            d["model_candidates"] = self.model_candidates
        return d


@dataclass
class LLMConfig:
    """
    Ready-to-serialize llm.yaml configuration.

    Matches the schema consumed by LLMRouter in circuitforge products.
    """

    profile_name: str       # e.g. "single-gpu-8gb" — matches a public GpuProfile
    backends: dict[str, LLMBackendConfig] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    research_fallback_order: list[str] = field(default_factory=list)
    vision_fallback_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backends": {k: v.to_dict() for k, v in self.backends.items()},
            "fallback_order": self.fallback_order,
            "research_fallback_order": self.research_fallback_order,
            "vision_fallback_order": self.vision_fallback_order,
        }
