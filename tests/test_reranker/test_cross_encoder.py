"""Tests for CrossEncoderTextReranker with mocked sentence-transformers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.reranker.adapters.cross_encoder import CrossEncoderTextReranker


def _make_mock_cross_encoder(scores: list[float]) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = scores
    return m


def test_model_id():
    assert (
        CrossEncoderTextReranker("mixedbread-ai/mxbai-rerank-base-v1").model_id
        == "mixedbread-ai/mxbai-rerank-base-v1"
    )


def test_load_raises_without_sentence_transformers():
    reranker = CrossEncoderTextReranker()
    with patch("circuitforge_core.reranker.adapters.cross_encoder._CrossEncoder", None):
        with pytest.raises(ImportError, match="sentence-transformers"):
            reranker.load()


def test_rerank_scores_and_sorts():
    reranker = CrossEncoderTextReranker("mixedbread-ai/mxbai-rerank-base-v1")
    reranker._model = _make_mock_cross_encoder([0.2, 0.9, 0.5])

    results = reranker.rerank("query", ["a", "b", "c"])
    assert results[0].candidate == "b"
    assert results[0].rank == 0
    assert results[2].candidate == "a"


def test_rerank_top_n():
    reranker = CrossEncoderTextReranker()
    reranker._model = _make_mock_cross_encoder([0.1, 0.8, 0.5])
    results = reranker.rerank("q", ["a", "b", "c"], top_n=2)
    assert len(results) == 2
    assert results[0].candidate == "b"


def test_predict_called_with_pairs():
    reranker = CrossEncoderTextReranker()
    mock_model = _make_mock_cross_encoder([0.7, 0.3])
    reranker._model = mock_model

    reranker.rerank("chicken soup", ["recipe one", "recipe two"])
    pairs = mock_model.predict.call_args[0][0]
    assert pairs == [("chicken soup", "recipe one"), ("chicken soup", "recipe two")]


def test_numpy_scores_coerced_to_float():
    """predict() may return numpy floats — verify they're converted cleanly."""
    try:
        import numpy as np
        numpy_scores = np.array([0.8, 0.2])
    except ImportError:
        pytest.skip("numpy not installed")

    reranker = CrossEncoderTextReranker()
    reranker._model = _make_mock_cross_encoder(numpy_scores)  # type: ignore[arg-type]
    results = reranker.rerank("q", ["a", "b"])
    assert isinstance(results[0].score, float)


def test_unload_clears_model():
    reranker = CrossEncoderTextReranker()
    reranker._model = MagicMock()
    reranker.unload()
    assert reranker._model is None
