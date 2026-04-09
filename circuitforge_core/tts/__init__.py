"""
circuitforge_core.tts — Text-to-speech service module.

Quick start (mock mode — no GPU or model required):

    import os; os.environ["CF_TTS_MOCK"] = "1"
    from circuitforge_core.tts import synthesize

    result = synthesize("Hello world")
    open("out.ogg", "wb").write(result.audio_bytes)

Real inference (chatterbox-turbo):

    export CF_TTS_MODEL=/Library/Assets/LLM/chatterbox/hub/models--ResembleAI--chatterbox-turbo/snapshots/<hash>
    from circuitforge_core.tts import synthesize

cf-orch service profile:

    service_type: cf-tts
    max_mb:       768
    max_concurrent: 1
    shared:       true
    managed:
      exec:       python -m circuitforge_core.tts.app
      args:       --model <path> --port {port} --gpu-id {gpu_id}
      port:       8005
      health:     /health
"""
from __future__ import annotations

import os

from circuitforge_core.tts.backends.base import (
    AudioFormat,
    TTSBackend,
    TTSResult,
    make_tts_backend,
)
from circuitforge_core.tts.backends.mock import MockTTSBackend

_backend: TTSBackend | None = None


def _get_backend() -> TTSBackend:
    global _backend
    if _backend is None:
        model_path = os.environ.get("CF_TTS_MODEL", "mock")
        mock = model_path == "mock" or os.environ.get("CF_TTS_MOCK", "") == "1"
        _backend = make_tts_backend(model_path, mock=mock)
    return _backend


def synthesize(
    text: str,
    *,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    audio_prompt: bytes | None = None,
    format: AudioFormat = "ogg",
) -> TTSResult:
    """Synthesize speech from text using the process-level backend."""
    return _get_backend().synthesize(
        text,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        audio_prompt=audio_prompt,
        format=format,
    )


def reset_backend() -> None:
    """Reset the process-level singleton. Test teardown only."""
    global _backend
    _backend = None


__all__ = [
    "AudioFormat",
    "TTSBackend",
    "TTSResult",
    "MockTTSBackend",
    "make_tts_backend",
    "synthesize",
    "reset_backend",
]
