"""
MediaPipe Hands wrapper.

Produces immutable HandLandmarks dataclasses from RGB video frames.
The caller is responsible for BGR→RGB conversion before passing frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class HandLandmarks:
    """Immutable snapshot of one detected hand."""

    points: np.ndarray  # shape (21, 3) — x, y, z in [0,1] normalized image space
    handedness: str  # 'Left' | 'Right' (mirror of physical hand)
    confidence: float  # [0.0, 1.0]


class HandsDetector:
    """
    Thin wrapper around mediapipe.solutions.hands.Hands.

    Usage:
        detector = HandsDetector()
        for frame_bgr in camera.frames():
            frame_rgb = frame_bgr[:, :, ::-1]
            hands = detector.detect(frame_rgb)
            for hand in hands:
                vec = normalize_hand(hand.points)
                ...
        detector.close()

    Or use as a context manager:
        with HandsDetector() as detector:
            ...
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb_frame: np.ndarray) -> list[HandLandmarks]:
        """
        Run hand detection on one RGB frame.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB image.

        Returns:
            List of HandLandmarks, one per detected hand (up to max_hands).
            Empty list if no hands detected.
        """
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return []
        out: list[HandLandmarks] = []
        for lm, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            points = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            points.flags.writeable = False  # enforce immutability of stored array
            out.append(
                HandLandmarks(
                    points=points,
                    handedness=hand.classification[0].label,
                    confidence=float(hand.classification[0].score),
                )
            )
        return out

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> HandsDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
