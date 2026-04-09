# circuitforge_core/stt/backends/base.py — STTBackend Protocol + factory
#
# MIT licensed. The Protocol and mock are always importable without GPU deps.
# Real backends require optional extras:
#   pip install -e "circuitforge-core[stt-faster-whisper]"
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class STTSegment:
    """Word- or phrase-level segment (included when the backend supports it)."""
    start_s: float
    end_s: float
    text: str
    confidence: float


@dataclass(frozen=True)
class STTResult:
    """
    Standard result from any STTBackend.transcribe() call.

    confidence is normalised to 0.0–1.0 regardless of the backend's native metric.
    below_threshold is True when confidence < the configured threshold (default 0.75).
    This flag is safety-critical for products like Osprey: DTMF must NOT be sent
    when below_threshold is True.
    """
    text: str
    confidence: float                       # 0.0–1.0
    below_threshold: bool
    language: str | None = None
    duration_s: float | None = None
    segments: list[STTSegment] = field(default_factory=list)
    model: str = ""

    CONFIDENCE_DEFAULT_THRESHOLD: float = 0.75


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class STTBackend(Protocol):
    """
    Abstract interface for speech-to-text backends.

    All backends load their model once at construction time and are safe to
    call concurrently (the model weights are read-only after load).
    """

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        confidence_threshold: float = STTResult.CONFIDENCE_DEFAULT_THRESHOLD,
    ) -> STTResult:
        """Synchronous transcription. audio is raw PCM or any format ffmpeg understands."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier for the loaded model (path stem or size name)."""
        ...

    @property
    def vram_mb(self) -> int:
        """Approximate VRAM footprint in MB. Used by cf-orch service registry."""
        ...


# ── Factory ───────────────────────────────────────────────────────────────────

def make_stt_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
) -> STTBackend:
    """
    Return an STTBackend for the given model.

    mock=True or CF_STT_MOCK=1  → MockSTTBackend (no GPU, no model file needed)
    backend="faster-whisper"    → FasterWhisperBackend (default)

    device and compute_type are passed through to the backend and ignored by mock.
    """
    use_mock = mock if mock is not None else os.environ.get("CF_STT_MOCK", "") == "1"
    if use_mock:
        from circuitforge_core.stt.backends.mock import MockSTTBackend
        return MockSTTBackend(model_name=model_path)

    resolved = backend or os.environ.get("CF_STT_BACKEND", "faster-whisper")
    if resolved == "faster-whisper":
        from circuitforge_core.stt.backends.faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend(
            model_path=model_path, device=device, compute_type=compute_type
        )

    raise ValueError(
        f"Unknown STT backend {resolved!r}. "
        "Expected 'faster-whisper'. Set CF_STT_BACKEND or pass backend= explicitly."
    )
