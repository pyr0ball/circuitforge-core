"""
circuitforge_core.stt — Speech-to-text service module.

Quick start (mock mode — no GPU or model required):

    import os; os.environ["CF_STT_MOCK"] = "1"
    from circuitforge_core.stt import transcribe

    result = transcribe(open("audio.wav", "rb").read())
    print(result.text, result.confidence)

Real inference (faster-whisper):

    export CF_STT_MODEL=/Library/Assets/LLM/whisper/models/Whisper/faster-whisper/models--Systran--faster-whisper-medium/snapshots/<hash>
    from circuitforge_core.stt import transcribe

cf-orch service profile:

    service_type: cf-stt
    max_mb:       1024 (medium); 600 (base/small)
    max_concurrent: 3
    shared:       true
    managed:
      exec:       python -m circuitforge_core.stt.app
      args:       --model <path> --port {port} --gpu-id {gpu_id}
      port:       8004
      health:     /health
"""
from __future__ import annotations

import os

from circuitforge_core.stt.backends.base import (
    STTBackend,
    STTResult,
    STTSegment,
    make_stt_backend,
)
from circuitforge_core.stt.backends.mock import MockSTTBackend

_backend: STTBackend | None = None


def _get_backend() -> STTBackend:
    global _backend
    if _backend is None:
        model_path = os.environ.get("CF_STT_MODEL", "mock")
        mock = model_path == "mock" or os.environ.get("CF_STT_MOCK", "") == "1"
        _backend = make_stt_backend(model_path, mock=mock)
    return _backend


def transcribe(
    audio: bytes,
    *,
    language: str | None = None,
    confidence_threshold: float = STTResult.CONFIDENCE_DEFAULT_THRESHOLD,
) -> STTResult:
    """Transcribe audio bytes using the process-level backend."""
    return _get_backend().transcribe(
        audio, language=language, confidence_threshold=confidence_threshold
    )


def reset_backend() -> None:
    """Reset the process-level singleton. Test teardown only."""
    global _backend
    _backend = None


__all__ = [
    "STTBackend",
    "STTResult",
    "STTSegment",
    "MockSTTBackend",
    "make_stt_backend",
    "transcribe",
    "reset_backend",
]
