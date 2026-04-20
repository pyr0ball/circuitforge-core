"""
Audio resampling — change sample rate of a float32 audio array.

Uses scipy.signal.resample_poly when available (high-quality, anti-aliased).
Falls back to linear interpolation via numpy when scipy is absent — acceptable
for 16kHz speech but not for music.
"""
from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """Resample audio from one sample rate to another.

    Args:
        audio:   float32 ndarray, shape (samples,) or (channels, samples).
        from_hz: Source sample rate in Hz.
        to_hz:   Target sample rate in Hz.

    Returns:
        Resampled float32 ndarray at to_hz.
    """
    if from_hz == to_hz:
        return audio.astype(np.float32)

    try:
        from scipy.signal import resample_poly  # type: ignore[import]
        from math import gcd
        g = gcd(from_hz, to_hz)
        up, down = to_hz // g, from_hz // g
        return resample_poly(audio.astype(np.float32), up, down, axis=-1)
    except ImportError:
        # Numpy linear interpolation fallback — lower quality but no extra deps.
        # Adequate for 16kHz ↔ 8kHz conversion on speech; avoid for music.
        n_out = int(len(audio) * to_hz / from_hz)
        x_old = np.linspace(0, 1, len(audio), endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        return np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.float32)
