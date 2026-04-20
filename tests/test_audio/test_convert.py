import numpy as np
import pytest
from circuitforge_core.audio.convert import pcm_to_float32, float32_to_pcm, bytes_to_float32


def _silence_pcm(n_samples: int = 1024) -> bytes:
    return (np.zeros(n_samples, dtype=np.int16)).tobytes()


def _sine_pcm(freq_hz: float = 440.0, sample_rate: int = 16000, duration_s: float = 0.1) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
    return samples.tobytes()


def test_pcm_to_float32_silence():
    result = pcm_to_float32(_silence_pcm())
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_pcm_to_float32_range():
    # Full-scale positive int16 -> ~1.0
    pcm = np.array([32767], dtype=np.int16).tobytes()
    result = pcm_to_float32(pcm)
    assert abs(result[0] - 1.0) < 0.001

    # Full-scale negative int16 -> ~-1.0
    pcm = np.array([-32768], dtype=np.int16).tobytes()
    result = pcm_to_float32(pcm)
    assert abs(result[0] - (-32768 / 32767)) < 0.001


def test_pcm_roundtrip():
    original = _sine_pcm()
    float_audio = pcm_to_float32(original)
    recovered = float32_to_pcm(float_audio)
    # Roundtrip through float32 introduces tiny quantisation error — within 1 LSB
    orig_arr = np.frombuffer(original, dtype=np.int16)
    recv_arr = np.frombuffer(recovered, dtype=np.int16)
    assert np.max(np.abs(orig_arr.astype(np.int32) - recv_arr.astype(np.int32))) <= 1


def test_float32_to_pcm_clips():
    # Values outside [-1.0, 1.0] must be clipped, not wrap.
    # int16 is asymmetric: max=32767, min=-32768. Scaling by 32767 means
    # -1.0 → -32767, not -32768 — that's expected and correct.
    audio = np.array([2.0, -2.0], dtype=np.float32)
    result = float32_to_pcm(audio)
    samples = np.frombuffer(result, dtype=np.int16)
    assert samples[0] == 32767
    assert samples[1] == -32767


def test_bytes_to_float32_alias():
    pcm = _sine_pcm()
    assert np.allclose(bytes_to_float32(pcm), pcm_to_float32(pcm))
