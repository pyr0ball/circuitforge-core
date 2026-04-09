# circuitforge_core/vision/backends/mock.py — MockVisionBackend
#
# Deterministic stub for tests and CI. No GPU, no model files required.
from __future__ import annotations

import math

from circuitforge_core.vision.backends.base import VisionBackend, VisionResult


class MockVisionBackend:
    """
    Mock VisionBackend for testing.

    classify() returns uniform scores normalised to 1/n per label.
    embed()    returns a unit vector of length 512 (all values 1/sqrt(512)).
    caption()  returns a canned string.
    """

    def __init__(self, model_name: str = "mock") -> None:
        self._model_name = model_name

    # ── VisionBackend Protocol ─────────────────────────────────────────────────

    def classify(self, image: bytes, labels: list[str]) -> VisionResult:
        n = max(len(labels), 1)
        return VisionResult(
            labels=list(labels),
            scores=[1.0 / n] * len(labels),
            model=self._model_name,
        )

    def embed(self, image: bytes) -> VisionResult:
        dim = 512
        val = 1.0 / math.sqrt(dim)
        return VisionResult(embedding=[val] * dim, model=self._model_name)

    def caption(self, image: bytes, prompt: str = "") -> VisionResult:
        return VisionResult(
            caption="A mock image description for testing purposes.",
            model=self._model_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vram_mb(self) -> int:
        return 0

    @property
    def supports_embed(self) -> bool:
        return True

    @property
    def supports_caption(self) -> bool:
        return True


# Verify protocol compliance at import time (catches missing methods early).
assert isinstance(MockVisionBackend(), VisionBackend)
