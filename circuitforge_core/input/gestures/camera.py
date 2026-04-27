"""
OpenCV camera capture — context manager wrapping VideoCapture.

Yields BGR frames. Callers convert to RGB before passing to HandsDetector:
    frame_rgb = frame_bgr[:, :, ::-1]
"""

from __future__ import annotations

from typing import Iterator

import cv2


class CameraCapture:
    """
    Thin wrapper around cv2.VideoCapture.

    Usage:
        with CameraCapture(device_index=0) as cam:
            for frame_bgr in cam.frames():
                process(frame_bgr)
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()

    def frames(self) -> Iterator:
        """Yield BGR uint8 frames until camera fails or caller breaks."""
        while self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> CameraCapture:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
