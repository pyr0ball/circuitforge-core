"""
cf_input.gestures — camera capture, hand detection, landmark normalization.

Public API:
    CameraCapture      — OpenCV frame source
    HandsDetector      — MediaPipe Hands wrapper
    HandLandmarks      — immutable detected hand dataclass
    normalize_hand()   — scale/translation-invariant feature vector
"""

from circuitforge_core.input.gestures.camera import CameraCapture
from circuitforge_core.input.gestures.hands import HandLandmarks, HandsDetector
from circuitforge_core.input.gestures.normalizer import normalize_hand

__all__ = ["CameraCapture", "HandLandmarks", "HandsDetector", "normalize_hand"]
