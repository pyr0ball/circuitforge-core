"""
Tests for cf-vision backends (mock) and factory routing.

Real SigLIP/VLM backends are not tested here — they require GPU + model downloads.
The mock backend exercises the full Protocol surface so we can verify the contract
without hardware dependencies.
"""
from __future__ import annotations

import math
import os

import pytest

from circuitforge_core.vision.backends.base import (
    VisionBackend,
    VisionResult,
    make_vision_backend,
)
from circuitforge_core.vision.backends.mock import MockVisionBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Not a real PNG, but enough for mock


@pytest.fixture()
def mock_backend() -> MockVisionBackend:
    return MockVisionBackend(model_name="test-mock")


# ── Protocol compliance ───────────────────────────────────────────────────────

def test_mock_is_vision_backend(mock_backend: MockVisionBackend) -> None:
    assert isinstance(mock_backend, VisionBackend)


def test_mock_model_name(mock_backend: MockVisionBackend) -> None:
    assert mock_backend.model_name == "test-mock"


def test_mock_vram_mb(mock_backend: MockVisionBackend) -> None:
    assert mock_backend.vram_mb == 0


def test_mock_supports_embed(mock_backend: MockVisionBackend) -> None:
    assert mock_backend.supports_embed is True


def test_mock_supports_caption(mock_backend: MockVisionBackend) -> None:
    assert mock_backend.supports_caption is True


# ── classify() ───────────────────────────────────────────────────────────────

def test_classify_returns_vision_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.classify(FAKE_IMAGE, ["cat", "dog", "bird"])
    assert isinstance(result, VisionResult)


def test_classify_labels_preserved(mock_backend: MockVisionBackend) -> None:
    labels = ["cat", "dog", "bird"]
    result = mock_backend.classify(FAKE_IMAGE, labels)
    assert result.labels == labels


def test_classify_scores_length_matches_labels(mock_backend: MockVisionBackend) -> None:
    labels = ["cat", "dog", "bird"]
    result = mock_backend.classify(FAKE_IMAGE, labels)
    assert len(result.scores) == len(labels)


def test_classify_uniform_scores(mock_backend: MockVisionBackend) -> None:
    labels = ["cat", "dog", "bird"]
    result = mock_backend.classify(FAKE_IMAGE, labels)
    expected = 1.0 / 3
    for score in result.scores:
        assert abs(score - expected) < 1e-9


def test_classify_single_label(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.classify(FAKE_IMAGE, ["document"])
    assert result.labels == ["document"]
    assert abs(result.scores[0] - 1.0) < 1e-9


def test_classify_model_name_in_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.classify(FAKE_IMAGE, ["x"])
    assert result.model == "test-mock"


# ── embed() ──────────────────────────────────────────────────────────────────

def test_embed_returns_vision_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.embed(FAKE_IMAGE)
    assert isinstance(result, VisionResult)


def test_embed_returns_embedding(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.embed(FAKE_IMAGE)
    assert result.embedding is not None
    assert len(result.embedding) == 512


def test_embed_is_unit_vector(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.embed(FAKE_IMAGE)
    magnitude = math.sqrt(sum(v * v for v in result.embedding))
    assert abs(magnitude - 1.0) < 1e-6


def test_embed_labels_empty(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.embed(FAKE_IMAGE)
    assert result.labels == []
    assert result.scores == []


def test_embed_model_name_in_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.embed(FAKE_IMAGE)
    assert result.model == "test-mock"


# ── caption() ────────────────────────────────────────────────────────────────

def test_caption_returns_vision_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.caption(FAKE_IMAGE)
    assert isinstance(result, VisionResult)


def test_caption_returns_string(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.caption(FAKE_IMAGE)
    assert isinstance(result.caption, str)
    assert len(result.caption) > 0


def test_caption_with_prompt(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.caption(FAKE_IMAGE, prompt="What is in this image?")
    assert result.caption is not None


def test_caption_model_name_in_result(mock_backend: MockVisionBackend) -> None:
    result = mock_backend.caption(FAKE_IMAGE)
    assert result.model == "test-mock"


# ── VisionResult helpers ──────────────────────────────────────────────────────

def test_top_returns_sorted_pairs() -> None:
    result = VisionResult(
        labels=["cat", "dog", "bird"],
        scores=[0.3, 0.6, 0.1],
    )
    top = result.top(2)
    assert top[0] == ("dog", 0.6)
    assert top[1] == ("cat", 0.3)


def test_top_default_n1() -> None:
    result = VisionResult(labels=["cat", "dog"], scores=[0.4, 0.9])
    assert result.top() == [("dog", 0.9)]


# ── Factory routing ───────────────────────────────────────────────────────────

def test_factory_mock_flag() -> None:
    backend = make_vision_backend("any-model", mock=True)
    assert isinstance(backend, MockVisionBackend)


def test_factory_mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_VISION_MOCK", "1")
    backend = make_vision_backend("any-model")
    assert isinstance(backend, MockVisionBackend)


def test_factory_mock_model_name() -> None:
    backend = make_vision_backend("google/siglip-so400m-patch14-384", mock=True)
    assert backend.model_name == "google/siglip-so400m-patch14-384"


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown vision backend"):
        make_vision_backend("any-model", backend="nonexistent", mock=False)


def test_factory_vlm_autodetect_moondream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detection should select VLM for moondream model paths."""
    # We mock at the import level to avoid requiring GPU deps
    monkeypatch.setenv("CF_VISION_MOCK", "0")
    # Just verify the ValueError is about vlm backend, not "unknown"
    # (the ImportError from missing torch is expected in CI)
    try:
        make_vision_backend("vikhyatk/moondream2", mock=False)
    except ImportError:
        pass  # Expected in CI without torch
    except ValueError as exc:
        pytest.fail(f"Should not raise ValueError for known backend: {exc}")


def test_factory_siglip_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detection should select siglip for non-VLM model paths (no ValueError)."""
    monkeypatch.setenv("CF_VISION_MOCK", "0")
    try:
        make_vision_backend("google/siglip-so400m-patch14-384", mock=False)
    except ValueError as exc:
        pytest.fail(f"Should not raise ValueError for known backend: {exc}")
    except Exception:
        pass  # ImportError or model-loading errors are expected outside GPU CI


# ── Process singleton ─────────────────────────────────────────────────────────

def test_module_classify_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_VISION_MOCK", "1")
    # Reset the module-level singleton
    import circuitforge_core.vision as vision_mod
    vision_mod._backend = None

    result = vision_mod.classify(FAKE_IMAGE, ["cat", "dog"])
    assert result.labels == ["cat", "dog"]
    assert len(result.scores) == 2


def test_module_embed_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_VISION_MOCK", "1")
    import circuitforge_core.vision as vision_mod
    vision_mod._backend = None

    result = vision_mod.embed(FAKE_IMAGE)
    assert result.embedding is not None


def test_module_caption_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CF_VISION_MOCK", "1")
    import circuitforge_core.vision as vision_mod
    vision_mod._backend = None

    result = vision_mod.caption(FAKE_IMAGE, prompt="Describe")
    assert result.caption is not None


def test_module_make_backend_returns_fresh_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    import circuitforge_core.vision as vision_mod
    b1 = vision_mod.make_backend("m1", mock=True)
    b2 = vision_mod.make_backend("m2", mock=True)
    assert b1 is not b2
    assert b1.model_name != b2.model_name
