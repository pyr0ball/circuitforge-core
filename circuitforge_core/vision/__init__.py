"""
circuitforge_core.vision — Managed vision service module.

Quick start (mock mode — no GPU or model required):

    import os; os.environ["CF_VISION_MOCK"] = "1"
    from circuitforge_core.vision import classify, embed

    result = classify(image_bytes, labels=["cat", "dog", "bird"])
    print(result.top(1))          # [("cat", 0.82)]

    emb = embed(image_bytes)
    print(len(emb.embedding))     # 1152  (so400m hidden dim)

Real inference (SigLIP — default, ~1.4 GB VRAM):

    export CF_VISION_MODEL=google/siglip-so400m-patch14-384
    from circuitforge_core.vision import classify

Full VLM inference (caption + VQA):

    export CF_VISION_BACKEND=vlm
    export CF_VISION_MODEL=vikhyatk/moondream2
    from circuitforge_core.vision import caption

Per-request backend (bypasses process singleton):

    from circuitforge_core.vision import make_backend
    vlm = make_backend("vikhyatk/moondream2", backend="vlm")
    result = vlm.caption(image_bytes, prompt="What text appears in this image?")

cf-orch service profile:

    service_type: cf-vision
    max_mb:       1536 (siglip-so400m); 2200 (moondream2); 14500 (llava-7b)
    max_concurrent: 4   (siglip); 1 (vlm)
    shared:       true
    managed:
      exec:       python -m circuitforge_core.vision.app
      args:       --model <path> --backend siglip --port {port} --gpu-id {gpu_id}
      port:       8006
      health:     /health
"""
from __future__ import annotations

import os

from circuitforge_core.vision.backends.base import (
    VisionBackend,
    VisionResult,
    make_vision_backend,
)
from circuitforge_core.vision.backends.mock import MockVisionBackend

_backend: VisionBackend | None = None


def _get_backend() -> VisionBackend:
    global _backend
    if _backend is None:
        model_path = os.environ.get("CF_VISION_MODEL", "mock")
        mock = model_path == "mock" or os.environ.get("CF_VISION_MOCK", "") == "1"
        _backend = make_vision_backend(model_path, mock=mock)
    return _backend


def classify(image: bytes, labels: list[str]) -> VisionResult:
    """Zero-shot image classification using the process-level backend."""
    return _get_backend().classify(image, labels)


def embed(image: bytes) -> VisionResult:
    """Image embedding using the process-level backend (SigLIP only)."""
    return _get_backend().embed(image)


def caption(image: bytes, prompt: str = "") -> VisionResult:
    """Image captioning / VQA using the process-level backend (VLM only)."""
    return _get_backend().caption(image, prompt)


def make_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
    device: str = "cuda",
    dtype: str = "float16",
) -> VisionBackend:
    """
    Create a one-off VisionBackend without affecting the process singleton.

    Useful when a product needs both SigLIP (routing) and a VLM (captioning)
    in the same process, or when testing different models side-by-side.
    """
    return make_vision_backend(
        model_path, backend=backend, mock=mock, device=device, dtype=dtype
    )


__all__ = [
    "VisionBackend",
    "VisionResult",
    "MockVisionBackend",
    "classify",
    "embed",
    "caption",
    "make_backend",
]
