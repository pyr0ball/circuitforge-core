# circuitforge_core/reranker/adapters/mock.py — deterministic mock reranker
#
# Always importable, no optional deps. Used in tests and CF_RERANKER_MOCK=1 mode.
# Scores by descending overlap of query tokens with candidate tokens so results
# are deterministic and meaningful enough to exercise product code paths.
from __future__ import annotations

from typing import Sequence

from circuitforge_core.reranker.base import RerankResult, TextReranker


class MockTextReranker(TextReranker):
    """Deterministic reranker for tests. No model weights required.

    Scoring: Jaccard similarity between query token set and candidate token set.
    Ties broken by candidate length (shorter wins) then lexicographic order,
    so test assertions can be written against a stable ordering.
    """

    _MODEL_ID = "mock"

    @property
    def model_id(self) -> str:
        return self._MODEL_ID

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        scores: list[float] = []
        for candidate in candidates:
            c_tokens = set(candidate.lower().split())
            union = q_tokens | c_tokens
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(q_tokens & c_tokens) / len(union))
        return scores
