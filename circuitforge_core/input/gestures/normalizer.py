"""
Landmark normalization for MediaPipe hand landmarks.

Converts raw (21, 3) landmark array into a 63-element translation- and
scale-invariant feature vector suitable for gesture classifiers.
"""
import numpy as np


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """
    Normalize 21 MediaPipe hand landmarks into a scale/translation-invariant
    63-element feature vector.

    Steps:
        1. Translate so wrist (landmark 0) is at origin.
        2. Scale so distance from wrist to middle-finger MCP (landmark 9) = 1.0.
           If that distance is near-zero (degenerate hand), return zeros.
        3. Flatten to shape (63,).

    Args:
        points: (21, 3) float32 array — raw MediaPipe landmark coords.

    Returns:
        (63,) float32 feature vector.
    """
    pts = points.astype(np.float32).copy()
    pts -= pts[0]                          # translate: wrist → origin
    scale = float(np.linalg.norm(pts[9])) # wrist-to-middle-MCP distance
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()
