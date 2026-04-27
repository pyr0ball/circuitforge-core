import numpy as np
import pytest
from circuitforge_core.input.gestures.normalizer import normalize_hand


def _synthetic_hand(scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """21 landmarks, wrist at offset, middle MCP at offset + (scale, 0, 0)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # All landmarks start at the offset (roughly at the wrist)
    for i in range(21):
        pts[i] = [offset, 0.0, 0.0]
    # Then define a few key landmarks relative to wrist
    pts[0] = [offset, 0.0, 0.0]  # wrist
    pts[9] = [offset + scale, 0.0, 0.0]  # middle MCP at distance scale from wrist
    pts[1] = [offset + 0.1 * scale, 0.05 * scale, 0.0]  # thumb
    pts[5] = [offset + 0.4 * scale, 0.2 * scale, 0.0]  # index
    return pts


def test_output_shape():
    pts = _synthetic_hand()
    result = normalize_hand(pts)
    assert result.shape == (63,)


def test_translation_invariance():
    pts_a = _synthetic_hand(offset=0.0)
    pts_b = _synthetic_hand(offset=5.0)
    np.testing.assert_allclose(normalize_hand(pts_a), normalize_hand(pts_b), atol=1e-5)


def test_scale_invariance():
    pts_small = _synthetic_hand(scale=0.5)
    pts_large = _synthetic_hand(scale=2.0)
    np.testing.assert_allclose(
        normalize_hand(pts_small), normalize_hand(pts_large), atol=1e-5
    )


def test_zero_scale_does_not_crash():
    """All landmarks at same point — degenerate hand. Should return zeros, not raise."""
    pts = np.zeros((21, 3), dtype=np.float32)
    result = normalize_hand(pts)
    assert result.shape == (63,)
    assert not np.any(np.isnan(result))


def test_dtype_is_float32():
    pts = _synthetic_hand()
    result = normalize_hand(pts)
    assert result.dtype == np.float32
