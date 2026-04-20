import numpy as np
import pytest
from circuitforge_core.audio.buffer import ChunkAccumulator


def _chunk(value: float = 0.0, size: int = 1600) -> np.ndarray:
    return np.full(size, value, dtype=np.float32)


def test_not_ready_initially():
    acc = ChunkAccumulator(window_chunks=3)
    assert acc.is_ready() is False
    assert acc.chunk_count == 0


def test_ready_after_window_filled():
    acc = ChunkAccumulator(window_chunks=3)
    for _ in range(3):
        acc.accumulate(_chunk())
    assert acc.is_ready() is True
    assert acc.chunk_count == 3


def test_flush_returns_concatenated():
    acc = ChunkAccumulator(window_chunks=2)
    acc.accumulate(_chunk(0.1, size=100))
    acc.accumulate(_chunk(0.2, size=100))
    result = acc.flush()
    assert result.shape == (200,)
    assert np.allclose(result[:100], 0.1)
    assert np.allclose(result[100:], 0.2)


def test_flush_clears_buffer():
    acc = ChunkAccumulator(window_chunks=2)
    acc.accumulate(_chunk())
    acc.accumulate(_chunk())
    acc.flush()
    assert acc.chunk_count == 0
    assert acc.is_ready() is False


def test_flush_raises_when_not_ready():
    acc = ChunkAccumulator(window_chunks=3)
    acc.accumulate(_chunk())
    with pytest.raises(RuntimeError, match="Not enough chunks"):
        acc.flush()


def test_reset_clears_buffer():
    acc = ChunkAccumulator(window_chunks=2)
    acc.accumulate(_chunk())
    acc.accumulate(_chunk())
    acc.reset()
    assert acc.chunk_count == 0


def test_oldest_dropped_when_overfilled():
    # Accumulate more than window_chunks — oldest should be evicted
    acc = ChunkAccumulator(window_chunks=2)
    acc.accumulate(_chunk(1.0, size=10))  # will be evicted
    acc.accumulate(_chunk(2.0, size=10))
    acc.accumulate(_chunk(3.0, size=10))
    assert acc.chunk_count == 2
    result = acc.flush()
    assert np.allclose(result[:10], 2.0)
    assert np.allclose(result[10:], 3.0)


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        ChunkAccumulator(window_chunks=0)
