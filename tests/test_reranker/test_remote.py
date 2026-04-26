"""Tests for RemoteTextReranker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.reranker.adapters.remote import RemoteTextReranker


def _make_mock_response(results: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"results": results}
    resp.raise_for_status = MagicMock()
    return resp


def test_model_id():
    assert RemoteTextReranker("http://10.0.0.1:8011").model_id == "remote"


def test_score_pairs_posts_to_rerank_endpoint():
    reranker = RemoteTextReranker("http://10.0.0.1:8011")
    mock_resp = _make_mock_response([
        {"candidate": "doc a", "score": 0.9, "rank": 0},
        {"candidate": "doc b", "score": 0.3, "rank": 1},
    ])
    with patch("requests.post", return_value=mock_resp) as mock_post:
        scores = reranker._score_pairs("query", ["doc a", "doc b"])

    mock_post.assert_called_once_with(
        "http://10.0.0.1:8011/rerank",
        json={"query": "query", "candidates": ["doc a", "doc b"], "top_n": 0},
        timeout=30,
    )
    assert scores == [pytest.approx(0.9), pytest.approx(0.3)]


def test_score_pairs_restores_original_order():
    """Remote may return results in any order — scores must align with input."""
    reranker = RemoteTextReranker("http://10.0.0.1:8011")
    # Remote returned c first (highest score), then a, then b
    mock_resp = _make_mock_response([
        {"candidate": "c", "score": 0.95, "rank": 0},
        {"candidate": "a", "score": 0.6, "rank": 1},
        {"candidate": "b", "score": 0.1, "rank": 2},
    ])
    with patch("requests.post", return_value=mock_resp):
        scores = reranker._score_pairs("q", ["a", "b", "c"])
    assert scores == [pytest.approx(0.6), pytest.approx(0.1), pytest.approx(0.95)]


def test_score_pairs_raises_on_http_error():
    import requests as req
    reranker = RemoteTextReranker("http://10.0.0.1:8011")
    with patch("requests.post", side_effect=req.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="Remote reranker"):
            reranker._score_pairs("q", ["doc"])


def test_rerank_end_to_end():
    reranker = RemoteTextReranker("http://10.0.0.1:8011")
    mock_resp = _make_mock_response([
        {"candidate": "irrelevant", "score": 0.2, "rank": 0},
        {"candidate": "very relevant", "score": 0.9, "rank": 1},
    ])
    with patch("requests.post", return_value=mock_resp):
        results = reranker.rerank("q", ["irrelevant", "very relevant"])
    assert results[0].candidate == "very relevant"
    assert results[0].rank == 0


# ── make_reranker wiring ──────────────────────────────────────────────────────

def test_make_reranker_qwen3():
    from circuitforge_core.reranker import make_reranker
    from circuitforge_core.reranker.adapters.qwen3 import Qwen3TextReranker
    r = make_reranker("Qwen/Qwen3-Reranker-0.6B", backend="qwen3")
    assert isinstance(r, Qwen3TextReranker)


def test_make_reranker_cross_encoder():
    from circuitforge_core.reranker import make_reranker
    from circuitforge_core.reranker.adapters.cross_encoder import CrossEncoderTextReranker
    r = make_reranker("mixedbread-ai/mxbai-rerank-base-v1", backend="cross-encoder")
    assert isinstance(r, CrossEncoderTextReranker)


def test_make_reranker_cohere():
    from circuitforge_core.reranker import make_reranker
    from circuitforge_core.reranker.adapters.cohere import CohereTextReranker
    r = make_reranker("rerank-english-v3.0", backend="cohere")
    assert isinstance(r, CohereTextReranker)


def test_make_reranker_remote():
    from circuitforge_core.reranker import make_reranker
    r = make_reranker("http://10.0.0.1:8011", backend="remote")
    assert isinstance(r, RemoteTextReranker)


def test_make_reranker_unknown_raises():
    from circuitforge_core.reranker import make_reranker
    with pytest.raises(ValueError, match="cross-encoder"):
        make_reranker(backend="unknown-backend")
