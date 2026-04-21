"""Tests for BGETextReranker with mocked FlagEmbedding."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.reranker.adapters.bge import BGETextReranker
from circuitforge_core.reranker.base import RerankResult


def _make_mock_flag_reranker(scores: list[float]) -> MagicMock:
    """Return a mock FlagReranker that yields the given scores."""
    m = MagicMock()
    m.compute_score.return_value = scores
    return m


# ── BGETextReranker unit tests ────────────────────────────────────────────────

def test_bge_rerank_scores_and_sorts():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    mock_fr = _make_mock_flag_reranker([0.2, 0.9, 0.5])
    reranker._reranker = mock_fr

    results = reranker.rerank("query", ["a", "b", "c"])
    assert len(results) == 3
    assert results[0].candidate == "b"  # highest score 0.9
    assert results[0].score == pytest.approx(0.9)
    assert results[0].rank == 0
    assert results[1].candidate == "c"
    assert results[2].candidate == "a"


def test_bge_rerank_top_n():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    reranker._reranker = _make_mock_flag_reranker([0.1, 0.8, 0.5])
    results = reranker.rerank("q", ["a", "b", "c"], top_n=2)
    assert len(results) == 2
    assert results[0].candidate == "b"


def test_bge_rerank_single_candidate_float_return():
    """compute_score returns a float (not list) for a single pair."""
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    mock_fr = MagicMock()
    mock_fr.compute_score.return_value = 0.75  # single float
    reranker._reranker = mock_fr
    results = reranker.rerank("q", ["only candidate"])
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.75)


def test_bge_rerank_batch_flattens_pairs():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    mock_fr = _make_mock_flag_reranker([0.9, 0.1, 0.3, 0.8])
    reranker._reranker = mock_fr

    batch = reranker.rerank_batch(
        ["q1", "q2"],
        [["a1", "a2"], ["b1", "b2"]],
    )
    assert len(batch) == 2
    # q1: scores [0.9, 0.1] → a1 first
    assert batch[0][0].candidate == "a1"
    # q2: scores [0.3, 0.8] → b2 first
    assert batch[1][0].candidate == "b2"

    # All pairs were sent in a single compute_score call
    all_pairs = mock_fr.compute_score.call_args[0][0]
    assert len(all_pairs) == 4


def test_bge_rerank_empty_batch():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    reranker._reranker = MagicMock()
    result = reranker.rerank_batch([], [], top_n=5)
    assert result == []


def test_bge_load_raises_without_flagembedding():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    with patch("circuitforge_core.reranker.adapters.bge._FlagReranker", None):
        with pytest.raises(ImportError, match="FlagEmbedding"):
            reranker.load()


def test_bge_model_id():
    assert BGETextReranker("BAAI/bge-reranker-v2-m3").model_id == "BAAI/bge-reranker-v2-m3"


def test_bge_unload_clears_reranker():
    reranker = BGETextReranker("BAAI/bge-reranker-base")
    reranker._reranker = MagicMock()
    reranker.unload()
    assert reranker._reranker is None
