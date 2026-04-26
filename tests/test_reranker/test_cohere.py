"""Tests for CohereTextReranker with mocked cohere client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.reranker.adapters.cohere import CohereTextReranker


def _make_cohere_result(index: int, score: float) -> MagicMock:
    r = MagicMock()
    r.index = index
    r.relevance_score = score
    return r


def _make_mock_client(results: list[MagicMock]) -> MagicMock:
    response = MagicMock()
    response.results = results
    client = MagicMock()
    client.rerank.return_value = response
    return client


def test_model_id_includes_model_name():
    r = CohereTextReranker(model="rerank-multilingual-v3.0")
    assert r.model_id == "cohere:rerank-multilingual-v3.0"


def test_raises_without_cohere_package():
    reranker = CohereTextReranker(api_key="co-test")
    with patch("circuitforge_core.reranker.adapters.cohere._cohere", None):
        with pytest.raises(ImportError, match="cohere"):
            reranker._score_pairs("q", ["doc"])


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    reranker = CohereTextReranker()  # no api_key arg
    mock_cohere = MagicMock()
    with patch("circuitforge_core.reranker.adapters.cohere._cohere", mock_cohere):
        with pytest.raises(RuntimeError, match="API key"):
            reranker._get_client()


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "co-fromenv")
    mock_cohere = MagicMock()
    with patch("circuitforge_core.reranker.adapters.cohere._cohere", mock_cohere):
        reranker = CohereTextReranker()
        reranker._get_client()
        mock_cohere.Client.assert_called_once_with(api_key="co-fromenv")


def test_score_pairs_returns_original_order():
    """Cohere returns results sorted by score; we must restore original order."""
    reranker = CohereTextReranker(api_key="co-test")
    # Cohere returns candidates ranked: index 2 (0.9), index 0 (0.6), index 1 (0.1)
    mock_client = _make_mock_client([
        _make_cohere_result(index=2, score=0.9),
        _make_cohere_result(index=0, score=0.6),
        _make_cohere_result(index=1, score=0.1),
    ])
    with patch.object(reranker, "_get_client", return_value=mock_client):
        scores = reranker._score_pairs("query", ["a", "b", "c"])
    # Original order: a=0.6, b=0.1, c=0.9
    assert scores == [pytest.approx(0.6), pytest.approx(0.1), pytest.approx(0.9)]


def test_rerank_sorts_correctly():
    reranker = CohereTextReranker(api_key="co-test")
    mock_client = _make_mock_client([
        _make_cohere_result(index=1, score=0.95),
        _make_cohere_result(index=0, score=0.3),
    ])
    with patch.object(reranker, "_get_client", return_value=mock_client):
        results = reranker.rerank("query", ["less relevant", "more relevant"])
    assert results[0].candidate == "more relevant"
    assert results[0].rank == 0


def test_rerank_top_n():
    reranker = CohereTextReranker(api_key="co-test")
    mock_client = _make_mock_client([
        _make_cohere_result(index=0, score=0.9),
        _make_cohere_result(index=1, score=0.5),
        _make_cohere_result(index=2, score=0.1),
    ])
    with patch.object(reranker, "_get_client", return_value=mock_client):
        results = reranker.rerank("q", ["a", "b", "c"], top_n=2)
    assert len(results) == 2


def test_rerank_calls_cohere_with_correct_args():
    reranker = CohereTextReranker(api_key="co-test", model="rerank-english-v3.0")
    mock_client = _make_mock_client([_make_cohere_result(index=0, score=0.8)])
    with patch.object(reranker, "_get_client", return_value=mock_client):
        reranker.rerank("my query", ["only doc"])
    mock_client.rerank.assert_called_once_with(
        query="my query",
        documents=["only doc"],
        model="rerank-english-v3.0",
        top_n=1,
        max_chunks_per_doc=1,
    )
