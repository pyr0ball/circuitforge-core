"""Tests for RerankResult, TextReranker base class, and the public rerank() API."""
from __future__ import annotations

import os
import pytest

from circuitforge_core.reranker.base import RerankResult, TextReranker
from circuitforge_core.reranker.adapters.mock import MockTextReranker


# ── RerankResult ──────────────────────────────────────────────────────────────

def test_rerank_result_fields():
    r = RerankResult(candidate="recipe text", score=0.9, rank=0)
    assert r.candidate == "recipe text"
    assert r.score == 0.9
    assert r.rank == 0


def test_rerank_result_is_frozen():
    r = RerankResult(candidate="x", score=0.5, rank=0)
    with pytest.raises(Exception):
        r.score = 0.1  # type: ignore[misc]


# ── MockTextReranker ──────────────────────────────────────────────────────────

def test_mock_rerank_returns_sorted_results():
    reranker = MockTextReranker()
    results = reranker.rerank(
        "chicken soup recipe",
        ["chocolate cake recipe", "chicken noodle soup", "tomato basil pasta"],
    )
    assert len(results) == 3
    # "chicken noodle soup" shares more tokens with query → should rank first
    assert results[0].candidate == "chicken noodle soup"
    assert results[0].rank == 0
    assert results[1].rank == 1
    assert results[2].rank == 2


def test_mock_rerank_top_n():
    reranker = MockTextReranker()
    results = reranker.rerank("chicken", ["a", "b chicken", "c chicken soup"], top_n=2)
    assert len(results) == 2


def test_mock_rerank_empty_candidates():
    reranker = MockTextReranker()
    assert reranker.rerank("query", []) == []


def test_mock_rerank_scores_descending():
    reranker = MockTextReranker()
    results = reranker.rerank("apple pie dessert", ["apple pie", "beef stew", "apple crumble dessert"])
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_mock_rerank_batch():
    reranker = MockTextReranker()
    batch = reranker.rerank_batch(
        ["chicken soup", "chocolate cake"],
        [["chicken noodle", "beef stew"], ["chocolate mousse", "caesar salad"]],
        top_n=1,
    )
    assert len(batch) == 2
    assert batch[0][0].candidate == "chicken noodle"
    assert batch[1][0].candidate == "chocolate mousse"


def test_mock_model_id():
    assert MockTextReranker().model_id == "mock"


# ── Public API singleton ──────────────────────────────────────────────────────

def test_rerank_function_mock_mode(monkeypatch):
    monkeypatch.setenv("CF_RERANKER_MOCK", "1")
    from circuitforge_core.reranker import rerank, reset_reranker
    reset_reranker()
    results = rerank("chicken soup", ["beef stew", "chicken noodle soup", "cake"])
    assert results[0].candidate == "chicken noodle soup"
    reset_reranker()


def test_make_reranker_mock_explicit():
    from circuitforge_core.reranker import make_reranker
    r = make_reranker(mock=True)
    assert isinstance(r, MockTextReranker)


def test_make_reranker_unknown_backend_raises():
    from circuitforge_core.reranker import make_reranker
    with pytest.raises(ValueError, match="Unknown reranker backend"):
        make_reranker(backend="nonexistent")


def test_reranker_protocol_conformance():
    """MockTextReranker satisfies the Reranker Protocol."""
    from circuitforge_core.reranker.base import Reranker
    assert isinstance(MockTextReranker(), Reranker)
