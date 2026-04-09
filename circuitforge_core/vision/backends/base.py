# circuitforge_core/vision/backends/base.py — VisionBackend Protocol + factory
#
# MIT licensed. The Protocol and mock are always importable without GPU deps.
# Real backends require optional extras:
#   pip install -e "circuitforge-core[vision-siglip]"   # SigLIP (default, ~1.4 GB VRAM)
#   pip install -e "circuitforge-core[vision-vlm]"      # Full VLM (e.g. moondream, LLaVA)
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisionResult:
    """
    Standard result from any VisionBackend call.

    classify() → labels + scores populated; embedding/caption may be None.
    embed()    → embedding populated; labels/scores empty.
    caption()  → caption populated; labels/scores empty; embedding None.
    """
    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    embedding: list[float] | None = None
    caption: str | None = None
    model: str = ""

    def top(self, n: int = 1) -> list[tuple[str, float]]:
        """Return the top-n (label, score) pairs sorted by descending score."""
        paired = sorted(zip(self.labels, self.scores), key=lambda x: x[1], reverse=True)
        return paired[:n]


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class VisionBackend(Protocol):
    """
    Abstract interface for vision backends.

    All backends load their model once at construction time.

    SigLIP backends implement classify() and embed() but raise NotImplementedError
    for caption().  VLM backends implement caption() and a prompt-based classify()
    but raise NotImplementedError for embed().
    """

    def classify(self, image: bytes, labels: list[str]) -> VisionResult:
        """
        Zero-shot image classification.

        labels: candidate text descriptions; scores are returned in the same order.
        SigLIP uses sigmoid similarity; VLM prompts for each label.
        """
        ...

    def embed(self, image: bytes) -> VisionResult:
        """
        Return an image embedding vector.

        Available on SigLIP backends.  Raises NotImplementedError on VLM backends.
        embedding is a list of floats with length == model hidden dim.
        """
        ...

    def caption(self, image: bytes, prompt: str = "") -> VisionResult:
        """
        Generate a text description of the image.

        Available on VLM backends.  Raises NotImplementedError on SigLIP backends.
        prompt is an optional instruction; defaults to a generic description request.
        """
        ...

    @property
    def model_name(self) -> str:
        """Identifier for the loaded model (HuggingFace ID or path stem)."""
        ...

    @property
    def vram_mb(self) -> int:
        """Approximate VRAM footprint in MB. Used by cf-orch service registry."""
        ...

    @property
    def supports_embed(self) -> bool:
        """True if embed() is implemented (SigLIP backends)."""
        ...

    @property
    def supports_caption(self) -> bool:
        """True if caption() is implemented (VLM backends)."""
        ...


# ── Factory ───────────────────────────────────────────────────────────────────

def make_vision_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
    device: str = "cuda",
    dtype: str = "float16",
) -> VisionBackend:
    """
    Return a VisionBackend for the given model.

    mock=True or CF_VISION_MOCK=1  → MockVisionBackend (no GPU, no model file needed)
    backend="siglip"               → SigLIPBackend (default; classify + embed)
    backend="vlm"                  → VLMBackend (caption + prompt-based classify)

    Auto-detection: if model_path contains "siglip" → SigLIPBackend;
    otherwise defaults to siglip unless backend is explicitly "vlm".

    device and dtype are forwarded to the real backends and ignored by mock.
    """
    use_mock = mock if mock is not None else os.environ.get("CF_VISION_MOCK", "") == "1"
    if use_mock:
        from circuitforge_core.vision.backends.mock import MockVisionBackend
        return MockVisionBackend(model_name=model_path)

    resolved = backend or os.environ.get("CF_VISION_BACKEND", "")
    if not resolved:
        # Auto-detect from model path
        resolved = "vlm" if _looks_like_vlm(model_path) else "siglip"

    if resolved == "siglip":
        from circuitforge_core.vision.backends.siglip import SigLIPBackend
        return SigLIPBackend(model_path=model_path, device=device, dtype=dtype)

    if resolved == "vlm":
        from circuitforge_core.vision.backends.vlm import VLMBackend
        return VLMBackend(model_path=model_path, device=device, dtype=dtype)

    raise ValueError(
        f"Unknown vision backend {resolved!r}. "
        "Expected 'siglip' or 'vlm'. Set CF_VISION_BACKEND or pass backend= explicitly."
    )


def _looks_like_vlm(model_path: str) -> bool:
    """Heuristic: names associated with generative VLMs."""
    _vlm_hints = ("llava", "moondream", "qwen-vl", "qwenvl", "idefics",
                  "cogvlm", "internvl", "phi-3-vision", "phi3vision",
                  "dolphin", "paligemma")
    lower = model_path.lower()
    return any(h in lower for h in _vlm_hints)
