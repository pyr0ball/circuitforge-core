"""
Energy gate — silence detection via RMS amplitude.
"""
from __future__ import annotations

import numpy as np

# Default threshold extracted from cf-voice stt.py.
# Signals below this RMS level are considered silent.
_DEFAULT_RMS_THRESHOLD = 0.005


def is_silent(
    audio: np.ndarray,
    *,
    rms_threshold: float = _DEFAULT_RMS_THRESHOLD,
) -> bool:
    """Return True when the audio clip is effectively silent.

    Uses root-mean-square amplitude as the energy estimate. This is a fast
    frame-level gate — not a VAD model. Use it to skip inference on empty
    audio frames before they hit a more expensive transcription or
    classification pipeline.

    Args:
        audio:         float32 ndarray, values in [-1.0, 1.0].
        rms_threshold: Clips with RMS below this value are silent.
                       Default 0.005 is conservative — genuine speech at
                       normal mic levels sits well above this.

    Returns:
        True if silent, False if the clip contains meaningful signal.
    """
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return rms < rms_threshold


def rms(audio: np.ndarray) -> float:
    """Return the RMS amplitude of an audio array."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
