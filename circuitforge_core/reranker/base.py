# circuitforge_core/reranker/base.py — Reranker Protocol + modality branches
#
# MIT licensed. The Protocol and RerankResult are always importable.
# Adapter implementations (BGE, Qwen3, cf-orch remote) require optional extras.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankResult:
    """A single scored candidate returned by a reranker.

    rank is 0-based (0 = highest score).
    candidate preserves the original object — text, Path, or any other type
    passed in by the caller, so products don't need to re-index the input list.
    """
    candidate: Any
    score: float
    rank: int


# ── Trunk: generic Reranker Protocol ─────────────────────────────────────────


@runtime_checkable
class Reranker(Protocol):
    """
    Abstract interface for all reranker adapters.

    Implementations must be safe to construct once and call concurrently;
    internal state (loaded model weights) should be guarded by a lock if
    the backend is not thread-safe.

    query    — the reference item to rank against (typically a text query)
    candidates — ordered collection of items to score; ordering is preserved
                 in the returned list, which is sorted by score descending
    top_n    — return at most this many results; 0 means return all

    Returns a list of RerankResult sorted by score descending (rank 0 first).
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_n: int = 0,
    ) -> list[RerankResult]:
        ...

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[Any]],
        top_n: int = 0,
    ) -> list[list[RerankResult]]:
        """Score multiple (query, candidates) pairs in one call.

        Default implementation loops over rerank(); adapters may override
        with a true batched forward pass for efficiency.
        """
        ...

    @property
    def model_id(self) -> str:
        """Identifier for the loaded model (name, path, or URL)."""
        ...


# ── Branch: text-specific reranker ───────────────────────────────────────────


class TextReranker:
    """
    Base class for text-to-text rerankers.

    Subclasses implement _score_pairs(query, candidates) and get rerank()
    and rerank_batch() for free. The default rerank_batch() loops over
    _score_pairs; override it in adapters that support native batching.

    candidates must be strings. query is always a string.
    """

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    def _score_pairs(
        self,
        query: str,
        candidates: list[str],
    ) -> list[float]:
        """Return a score per candidate (higher = more relevant).

        Called by rerank() and rerank_batch(). Must return a list of the
        same length as candidates, in the same order.
        """
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        top_n: int = 0,
    ) -> list[RerankResult]:
        cands = list(candidates)
        if not cands:
            return []
        scores = self._score_pairs(query, cands)
        results = sorted(
            (RerankResult(candidate=c, score=s, rank=0) for c, s in zip(cands, scores)),
            key=lambda r: r.score,
            reverse=True,
        )
        if top_n > 0:
            results = results[:top_n]
        return [
            RerankResult(candidate=r.candidate, score=r.score, rank=i)
            for i, r in enumerate(results)
        ]

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[str]],
        top_n: int = 0,
    ) -> list[list[RerankResult]]:
        return [
            self.rerank(q, cs, top_n)
            for q, cs in zip(queries, candidates)
        ]
