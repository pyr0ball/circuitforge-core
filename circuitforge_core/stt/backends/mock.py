# circuitforge_core/stt/backends/mock.py — MockSTTBackend
#
# MIT licensed. No GPU, no model file required.
# Used in tests and CI, and when CF_STT_MOCK=1.
from __future__ import annotations

from circuitforge_core.stt.backends.base import STTBackend, STTResult


class MockSTTBackend:
    """
    Deterministic mock STT backend for testing.

    Returns a fixed transcript so tests can assert on the response shape
    without needing a GPU or a model file.
    """

    def __init__(
        self,
        model_name: str = "mock",
        fixed_text: str = "mock transcription",
        fixed_confidence: float = 0.95,
    ) -> None:
        self._model_name = model_name
        self._fixed_text = fixed_text
        self._fixed_confidence = fixed_confidence

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        confidence_threshold: float = STTResult.CONFIDENCE_DEFAULT_THRESHOLD,
    ) -> STTResult:
        return STTResult(
            text=self._fixed_text,
            confidence=self._fixed_confidence,
            below_threshold=self._fixed_confidence < confidence_threshold,
            language=language or "en",
            duration_s=float(len(audio)) / 32000,  # rough estimate: 16kHz 16-bit mono
            model=self._model_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vram_mb(self) -> int:
        return 0


# Satisfy the Protocol at import time (no GPU needed)
assert isinstance(MockSTTBackend(), STTBackend)
