import numpy as np
from circuitforge_core.audio.gate import is_silent, rms


def test_silence_detected():
    audio = np.zeros(1600, dtype=np.float32)
    assert is_silent(audio) is True


def test_speech_level_not_silent():
    # Sine at 0.1 amplitude — well above 0.005 RMS
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
    assert is_silent(audio) is False


def test_just_below_threshold():
    # RMS exactly at 0.004 — should be silent
    audio = np.full(1600, 0.004, dtype=np.float32)
    assert is_silent(audio, rms_threshold=0.005) is True


def test_just_above_threshold():
    audio = np.full(1600, 0.006, dtype=np.float32)
    assert is_silent(audio, rms_threshold=0.005) is False


def test_empty_array_is_silent():
    assert is_silent(np.array([], dtype=np.float32)) is True


def test_rms_zero_for_silence():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0


def test_rms_nonzero_for_signal():
    audio = np.ones(100, dtype=np.float32) * 0.5
    assert abs(rms(audio) - 0.5) < 1e-6
