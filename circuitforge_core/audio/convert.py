"""
PCM / float32 conversion utilities.

All functions operate on raw audio bytes or numpy arrays. No torch dependency.

Standard pipeline:
    bytes (int16 PCM) -> float32 ndarray -> signal processing -> bytes (int16 PCM)
"""
from __future__ import annotations

import numpy as np


def pcm_to_float32(pcm_bytes: bytes, *, dtype: np.dtype = np.int16) -> np.ndarray:
    """Convert raw PCM bytes to a float32 numpy array in [-1.0, 1.0].

    Args:
        pcm_bytes: Raw PCM audio bytes.
        dtype:     Sample dtype of the input. Default: int16 (standard mic input).

    Returns:
        float32 ndarray, values in [-1.0, 1.0].
    """
    scale = np.iinfo(dtype).max
    return np.frombuffer(pcm_bytes, dtype=dtype).astype(np.float32) / scale


def bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Alias for pcm_to_float32 with default int16 dtype.

    Matches the naming used in cf-voice context.py for easier migration.
    """
    return pcm_to_float32(pcm_bytes)


def float32_to_pcm(audio: np.ndarray, *, dtype: np.dtype = np.int16) -> bytes:
    """Convert a float32 ndarray in [-1.0, 1.0] to raw PCM bytes.

    Clips to [-1.0, 1.0] before scaling to prevent wraparound distortion.

    Args:
        audio: float32 ndarray, values nominally in [-1.0, 1.0].
        dtype: Target PCM sample dtype. Default: int16.

    Returns:
        Raw PCM bytes.
    """
    scale = np.iinfo(dtype).max
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * scale).astype(dtype).tobytes()
