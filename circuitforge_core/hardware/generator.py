# circuitforge_core/hardware/generator.py
"""
Profile generator: HardwareSpec → LLMConfig.

`generate_profile()` is the main entry point. It selects the appropriate
VRAM tier, builds a complete LLMConfig dict ready to write as llm.yaml,
and returns the matching public GpuProfile name for orch use.
"""
from __future__ import annotations

from .models import HardwareSpec, LLMBackendConfig, LLMConfig
from .tiers import select_tier


# Default backend URLs — overridable for non-standard setups
_OLLAMA_URL = "http://localhost:11434"
_VLLM_URL = "http://localhost:8000"
_VISION_URL = "http://localhost:8002"
_DOCUVISION_URL = "http://localhost:8003"


def generate_profile(
    spec: HardwareSpec,
    *,
    ollama_url: str = _OLLAMA_URL,
    vllm_url: str = _VLLM_URL,
    vision_url: str = _VISION_URL,
    docuvision_url: str = _DOCUVISION_URL,
) -> LLMConfig:
    """
    Map a HardwareSpec to an LLMConfig.

    Returns an LLMConfig whose `profile_name` matches a public GpuProfile YAML
    in `circuitforge_core/resources/profiles/public/` so the orch can load the
    correct service allocation profile automatically.
    """
    tier = select_tier(spec.vram_mb)
    has_vllm = "vllm" in tier.services
    has_vision = "cf-vision" in tier.services
    has_docuvision = "cf-docuvision" in tier.services

    backends: dict[str, LLMBackendConfig] = {}

    # Ollama is always available (CPU fallback)
    backends["ollama"] = LLMBackendConfig(
        enabled=True,
        url=ollama_url,
        model=tier.ollama_model,
    )

    # vllm — only on GPU tiers that can fit a model
    if has_vllm and tier.vllm_candidates:
        backends["vllm"] = LLMBackendConfig(
            enabled=True,
            url=vllm_url,
            model_candidates=list(tier.vllm_candidates),
        )

    # Vision service
    if has_vision:
        backends["vision_service"] = LLMBackendConfig(
            enabled=True,
            url=vision_url,
        )

    # Docuvision service
    if has_docuvision:
        backends["docuvision_service"] = LLMBackendConfig(
            enabled=True,
            url=docuvision_url,
        )

    # Fallback order: prefer vllm over ollama when available (faster for batch)
    if has_vllm:
        fallback = ["vllm", "ollama"]
        research_fallback = ["vllm", "ollama"]
    else:
        fallback = ["ollama"]
        research_fallback = ["ollama"]

    vision_fallback = (
        ["vision_service"] if has_vision else []
    ) + ["ollama"]

    return LLMConfig(
        profile_name=tier.profile_name,
        backends=backends,
        fallback_order=fallback,
        research_fallback_order=research_fallback,
        vision_fallback_order=vision_fallback,
    )
