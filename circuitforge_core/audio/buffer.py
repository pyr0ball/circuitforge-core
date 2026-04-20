"""
ChunkAccumulator — collect fixed-size audio chunks into a classify window.

Used by cf-voice and Linnet to gather N × 100ms frames before firing
a classification pass. The window size trades latency against context:
a 2-second window (20 × 100ms) gives the classifier enough signal to
detect tone/affect reliably without lagging the conversation.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ChunkAccumulator:
    """Accumulate audio chunks and flush when the window is full.

    Args:
        window_chunks: Number of chunks to collect before is_ready() is True.
        dtype:         numpy dtype of the accumulated array. Default float32.
    """

    def __init__(self, window_chunks: int, *, dtype: np.dtype = np.float32) -> None:
        if window_chunks < 1:
            raise ValueError(f"window_chunks must be >= 1, got {window_chunks}")
        self._window = window_chunks
        self._dtype = dtype
        self._buf: deque[np.ndarray] = deque()

    def accumulate(self, chunk: np.ndarray) -> None:
        """Add a chunk to the buffer. Oldest chunks are dropped once the
        buffer exceeds window_chunks to bound memory."""
        self._buf.append(chunk.astype(self._dtype))
        while len(self._buf) > self._window:
            self._buf.popleft()

    def is_ready(self) -> bool:
        """True when window_chunks have been accumulated."""
        return len(self._buf) >= self._window

    def flush(self) -> np.ndarray:
        """Concatenate accumulated chunks and reset the buffer.

        Returns:
            float32 ndarray of concatenated audio.

        Raises:
            RuntimeError: if fewer than window_chunks have been accumulated.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Not enough chunks accumulated: have {len(self._buf)}, "
                f"need {self._window}. Check is_ready() before calling flush()."
            )
        result = np.concatenate(list(self._buf), axis=-1).astype(self._dtype)
        self._buf.clear()
        return result

    def reset(self) -> None:
        """Discard all buffered audio without returning it."""
        self._buf.clear()

    @property
    def chunk_count(self) -> int:
        """Current number of buffered chunks."""
        return len(self._buf)
