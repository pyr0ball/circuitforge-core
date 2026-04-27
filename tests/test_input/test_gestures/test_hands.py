import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from circuitforge_core.input.gestures.hands import HandsDetector, HandLandmarks


def _make_mock_results(n_hands: int = 1):
    """Build a fake mediapipe result object with n_hands detected."""
    mock_results = MagicMock()
    if n_hands == 0:
        mock_results.multi_hand_landmarks = None
        mock_results.multi_handedness = None
        return mock_results

    hand_landmarks = []
    handedness_list = []
    for i in range(n_hands):
        lm = MagicMock()
        lm.landmark = [
            MagicMock(x=float(j) / 100, y=float(j) / 200, z=0.0) for j in range(21)
        ]
        hand_landmarks.append(lm)

        hand = MagicMock()
        hand.classification = [
            MagicMock(label="Right" if i == 0 else "Left", score=0.95)
        ]
        handedness_list.append(hand)

    mock_results.multi_hand_landmarks = hand_landmarks
    mock_results.multi_handedness = handedness_list
    return mock_results


@patch("circuitforge_core.input.gestures.hands.mp")
def test_detect_returns_empty_when_no_hands(mock_mp):
    mock_mp.solutions.hands.Hands.return_value.process.return_value = (
        _make_mock_results(0)
    )
    detector = HandsDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)
    assert results == []


@patch("circuitforge_core.input.gestures.hands.mp")
def test_detect_returns_one_hand(mock_mp):
    mock_mp.solutions.hands.Hands.return_value.process.return_value = (
        _make_mock_results(1)
    )
    detector = HandsDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)
    assert len(results) == 1
    h = results[0]
    assert isinstance(h, HandLandmarks)
    assert h.points.shape == (21, 3)
    assert h.handedness == "Right"
    assert 0.0 <= h.confidence <= 1.0


@patch("circuitforge_core.input.gestures.hands.mp")
def test_detect_returns_two_hands(mock_mp):
    mock_mp.solutions.hands.Hands.return_value.process.return_value = (
        _make_mock_results(2)
    )
    detector = HandsDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)
    assert len(results) == 2


@patch("circuitforge_core.input.gestures.hands.mp")
def test_handlandmarks_is_immutable(mock_mp):
    mock_mp.solutions.hands.Hands.return_value.process.return_value = (
        _make_mock_results(1)
    )
    detector = HandsDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)[0]
    with pytest.raises((AttributeError, TypeError)):
        result.handedness = (
            "Left"  # frozen dataclass must reject attribute reassignment
        )
    with pytest.raises(ValueError):
        result.points[0] = np.array(
            [1.0, 2.0, 3.0]
        )  # writeable=False must reject in-place mutation


@patch("circuitforge_core.input.gestures.hands.mp")
def test_full_pipeline_hands_to_normalized_vector(mock_mp):
    """Detect hand → normalize landmarks → get 63-element vector."""
    from circuitforge_core.input.gestures.normalizer import normalize_hand

    mock_mp.solutions.hands.Hands.return_value.process.return_value = (
        _make_mock_results(1)
    )
    detector = HandsDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = detector.detect(frame)
    assert len(hands) == 1
    vec = normalize_hand(hands[0].points)
    assert vec.shape == (63,)
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec))
