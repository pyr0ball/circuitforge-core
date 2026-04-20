"""
Tests for the mock MusicGen backend and shared audio encode/decode utilities.

All tests run without a GPU or AudioCraft install.
"""
import io
import wave

import pytest
from circuitforge_core.musicgen.backends.base import (
    MODEL_MELODY,
    MODEL_SMALL,
    MusicGenBackend,
    MusicContinueResult,
    make_musicgen_backend,
)
from circuitforge_core.musicgen.backends.mock import MockMusicGenBackend


# ── Mock backend ──────────────────────────────────────────────────────────────


def test_mock_satisfies_protocol():
    backend = MockMusicGenBackend()
    assert isinstance(backend, MusicGenBackend)


def test_mock_model_name():
    assert MockMusicGenBackend().model_name == "mock"


def test_mock_vram_mb():
    assert MockMusicGenBackend().vram_mb == 0


def _silent_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    n = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def test_mock_returns_result():
    backend = MockMusicGenBackend()
    result = backend.continue_audio(_silent_wav(), duration_s=5.0)
    assert isinstance(result, MusicContinueResult)


def test_mock_duration_matches_request():
    backend = MockMusicGenBackend()
    result = backend.continue_audio(_silent_wav(), duration_s=7.5)
    assert result.duration_s == 7.5


def test_mock_returns_valid_wav():
    backend = MockMusicGenBackend()
    result = backend.continue_audio(_silent_wav(), duration_s=2.0)
    assert result.format == "wav"
    buf = io.BytesIO(result.audio_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getnframes() > 0


def test_mock_sample_rate():
    backend = MockMusicGenBackend()
    result = backend.continue_audio(_silent_wav())
    assert result.sample_rate == 32000


def test_mock_prompt_duration_passthrough():
    backend = MockMusicGenBackend()
    result = backend.continue_audio(_silent_wav(), prompt_duration_s=8.0)
    assert result.prompt_duration_s == 8.0


def test_mock_description_ignored():
    backend = MockMusicGenBackend()
    # Should not raise regardless of description
    result = backend.continue_audio(_silent_wav(), description="upbeat jazz")
    assert result is not None


# ── make_musicgen_backend factory ─────────────────────────────────────────────


def test_factory_returns_mock_when_flag_set():
    backend = make_musicgen_backend(mock=True)
    assert isinstance(backend, MockMusicGenBackend)


def test_factory_mock_for_mock_model_name():
    backend = make_musicgen_backend(model_name="mock", mock=True)
    assert isinstance(backend, MockMusicGenBackend)
