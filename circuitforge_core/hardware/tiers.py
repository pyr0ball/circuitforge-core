# circuitforge_core/hardware/tiers.py
"""
VRAM tier ladder and model catalog.

Tiers map hardware VRAM (per-GPU) to:
  - profile_name: matching public GpuProfile YAML in profiles/public/
  - ollama_model:  best default Ollama model for this tier
  - vllm_candidates: ordered list of HF model dirs to try via cf-orch/vllm
  - services: which cf-* managed services are available at this tier
  - llm_max_params: rough upper bound, for human-readable display
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VramTier:
    vram_min_mb: int        # inclusive lower bound (0 = CPU-only)
    vram_max_mb: int        # exclusive upper bound (use sys.maxsize for the top tier)
    profile_name: str       # public GpuProfile YAML stem
    ollama_model: str       # e.g. "qwen2.5:7b-instruct-q4_k_m"
    vllm_candidates: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    llm_max_params: str = ""   # human-readable, e.g. "7b-q4"


# Ordered from smallest to largest — first match wins in select_tier().
VRAM_TIERS: list[VramTier] = [
    VramTier(
        vram_min_mb=0,
        vram_max_mb=1,
        profile_name="cpu-16gb",
        ollama_model="qwen2.5:1.5b-instruct-q4_k_m",
        vllm_candidates=[],
        services=["ollama", "cf-stt", "cf-tts"],
        llm_max_params="3b-q4",
    ),
    VramTier(
        vram_min_mb=1,
        vram_max_mb=3_000,
        profile_name="single-gpu-2gb",
        ollama_model="qwen2.5:1.5b-instruct-q4_k_m",
        vllm_candidates=[],
        services=["ollama"],
        llm_max_params="1.5b",
    ),
    VramTier(
        vram_min_mb=3_000,
        vram_max_mb=5_000,
        profile_name="single-gpu-4gb",
        ollama_model="qwen2.5:3b-instruct-q4_k_m",
        vllm_candidates=[],
        services=["ollama", "cf-vision", "cf-stt", "cf-tts"],
        llm_max_params="3b",
    ),
    VramTier(
        vram_min_mb=5_000,
        vram_max_mb=7_000,
        profile_name="single-gpu-6gb",
        ollama_model="qwen2.5:7b-instruct-q4_k_m",
        vllm_candidates=["Qwen2.5-3B-Instruct", "Phi-4-mini-instruct"],
        services=["ollama", "vllm", "cf-vision", "cf-docuvision", "cf-stt", "cf-tts"],
        llm_max_params="7b-q4",
    ),
    VramTier(
        vram_min_mb=7_000,
        vram_max_mb=12_000,
        profile_name="single-gpu-8gb",
        ollama_model="qwen2.5:7b-instruct",
        vllm_candidates=["Qwen2.5-3B-Instruct", "Phi-4-mini-instruct"],
        services=["ollama", "vllm", "cf-vision", "cf-docuvision", "cf-stt", "cf-tts", "cf-musicgen"],
        llm_max_params="8b",
    ),
    VramTier(
        vram_min_mb=12_000,
        vram_max_mb=20_000,
        profile_name="single-gpu-16gb",
        ollama_model="qwen2.5:14b-instruct-q4_k_m",
        vllm_candidates=["Qwen2.5-14B-Instruct", "Qwen2.5-3B-Instruct", "Phi-4-mini-instruct"],
        services=["ollama", "vllm", "cf-vision", "cf-docuvision", "cf-stt", "cf-tts",
                  "cf-musicgen", "cf-embed", "cf-classify"],
        llm_max_params="14b",
    ),
    VramTier(
        vram_min_mb=20_000,
        vram_max_mb=10 ** 9,
        profile_name="single-gpu-24gb",
        ollama_model="qwen2.5:32b-instruct-q4_k_m",
        vllm_candidates=["Qwen2.5-14B-Instruct", "Qwen2.5-3B-Instruct", "Phi-4-mini-instruct"],
        services=["ollama", "vllm", "cf-vision", "cf-docuvision", "cf-stt", "cf-tts",
                  "cf-musicgen", "cf-embed", "cf-classify", "comfyui"],
        llm_max_params="32b-q4",
    ),
]


def select_tier(vram_mb: int) -> VramTier:
    """Return the best matching tier for the given per-GPU VRAM in MB."""
    for tier in VRAM_TIERS:
        if tier.vram_min_mb <= vram_mb < tier.vram_max_mb:
            return tier
    # Fallback: return the top tier for unusually large VRAM
    return VRAM_TIERS[-1]
