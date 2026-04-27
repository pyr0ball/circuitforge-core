import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@patch("circuitforge_core.input.gestures.camera.cv2")
def test_is_open_reflects_videocapture_state(mock_cv2):
    from circuitforge_core.input.gestures.camera import CameraCapture

    mock_cv2.VideoCapture.return_value.isOpened.return_value = True
    cam = CameraCapture()
    assert cam.is_open is True

    mock_cv2.VideoCapture.return_value.isOpened.return_value = False
    cam2 = CameraCapture()
    assert cam2.is_open is False


@patch("circuitforge_core.input.gestures.camera.cv2")
def test_frames_yields_until_read_fails(mock_cv2):
    from circuitforge_core.input.gestures.camera import CameraCapture

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, frame),
        (True, frame),
        (False, None),  # triggers break
    ]
    mock_cv2.VideoCapture.return_value = mock_cap

    cam = CameraCapture()
    collected = list(cam.frames())
    assert len(collected) == 2


@patch("circuitforge_core.input.gestures.camera.cv2")
def test_context_manager_calls_release(mock_cv2):
    from circuitforge_core.input.gestures.camera import CameraCapture

    mock_cap = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap

    with CameraCapture() as cam:
        pass

    mock_cap.release.assert_called_once()
