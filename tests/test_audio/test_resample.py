import numpy as np
import pytest
from circuitforge_core.audio.resample import resample


def _sine(freq_hz: float, sample_rate: int, duration_s: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_same_rate_is_noop():
    audio = _sine(440.0, 16000)
    result = resample(audio, 16000, 16000)
    assert np.allclose(result, audio, atol=1e-5)


def test_output_length_correct():
    audio = _sine(440.0, 16000, duration_s=1.0)
    result = resample(audio, 16000, 8000)
    assert len(result) == 8000


def test_upsample_output_length():
    audio = _sine(440.0, 8000, duration_s=1.0)
    result = resample(audio, 8000, 16000)
    assert len(result) == 16000


def test_output_dtype_float32():
    audio = _sine(440.0, 16000)
    result = resample(audio, 16000, 8000)
    assert result.dtype == np.float32


def test_energy_preserved_approximately():
    # RMS should be approximately the same after resampling a sine wave
    audio = _sine(440.0, 16000, duration_s=1.0)
    result = resample(audio, 16000, 8000)
    rms_in = float(np.sqrt(np.mean(audio ** 2)))
    rms_out = float(np.sqrt(np.mean(result ** 2)))
    assert abs(rms_in - rms_out) < 0.05  # within 5% — resampling is not power-preserving exactly
