# circuitforge_core/reranker/adapters/bge.py — BGE cross-encoder reranker
#
# Requires: pip install circuitforge-core[reranker-bge]
# Tested with FlagEmbedding>=1.2 (BAAI/bge-reranker-* family).
#
# MIT licensed — local inference only, no tier gate.
from __future__ import annotations

import logging
import threading
from typing import Sequence

from circuitforge_core.reranker.base import TextReranker

logger = logging.getLogger(__name__)

# Lazy import sentinel — FlagEmbedding is an optional dep.
try:
    from FlagEmbedding import FlagReranker as _FlagReranker  # type: ignore[import]
except ImportError:
    _FlagReranker = None  # type: ignore[assignment]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class BGETextReranker(TextReranker):
    """
    Cross-encoder reranker using the BAAI BGE reranker family.

    Scores (query, candidate) pairs via FlagEmbedding.FlagReranker.
    Thread-safe: a lock serialises concurrent _score_pairs calls since
    FlagReranker is not guaranteed to be reentrant.

    Recommended free-tier models:
      BAAI/bge-reranker-base        ~570MB VRAM, fast
      BAAI/bge-reranker-v2-m3       ~570MB VRAM, multilingual
      BAAI/bge-reranker-large       ~1.3GB VRAM, higher quality

    Usage:
        reranker = BGETextReranker("BAAI/bge-reranker-base")
        results = reranker.rerank("chicken soup recipe", ["recipe 1...", "recipe 2..."])
    """

    def __init__(self, model_id: str = "BAAI/bge-reranker-base") -> None:
        self._model_id = model_id
        self._reranker: object | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    def load(self) -> None:
        """Explicitly load model weights. Called automatically on first rerank()."""
        if _FlagReranker is None:
            raise ImportError(
                "FlagEmbedding is not installed. "
                "Run: pip install circuitforge-core[reranker-bge]"
            )
        with self._lock:
            if self._reranker is None:
                logger.info("Loading BGE reranker: %s (fp16=%s)", self._model_id, _cuda_available())
                self._reranker = _FlagReranker(self._model_id, use_fp16=_cuda_available())

    def unload(self) -> None:
        """Release model weights. Useful for VRAM management between tasks."""
        with self._lock:
            self._reranker = None

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        if self._reranker is None:
            self.load()
        pairs = [[query, c] for c in candidates]
        with self._lock:
            scores: list[float] = self._reranker.compute_score(  # type: ignore[union-attr]
                pairs, normalize=True
            )
        # compute_score may return a single float when given one pair.
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[str]],
        top_n: int = 0,
    ) -> list[list[object]]:
        """Batch all pairs into a single compute_score call for efficiency."""
        from circuitforge_core.reranker.base import RerankResult

        if self._reranker is None:
            self.load()

        # Flatten all pairs, recording group boundaries for reconstruction.
        all_pairs: list[list[str]] = []
        group_sizes: list[int] = []
        for q, cs in zip(queries, candidates):
            cands = list(cs)
            group_sizes.append(len(cands))
            all_pairs.extend([q, c] for c in cands)

        if not all_pairs:
            return [[] for _ in queries]

        with self._lock:
            all_scores: list[float] = self._reranker.compute_score(  # type: ignore[union-attr]
                all_pairs, normalize=True
            )
        if isinstance(all_scores, float):
            all_scores = [all_scores]

        # Reconstruct per-query result lists.
        results: list[list[RerankResult]] = []
        offset = 0
        for (q, cs), size in zip(zip(queries, candidates), group_sizes):
            cands = list(cs)
            scores = all_scores[offset : offset + size]
            offset += size
            sorted_results = sorted(
                (RerankResult(candidate=c, score=s, rank=0) for c, s in zip(cands, scores)),
                key=lambda r: r.score,
                reverse=True,
            )
            if top_n > 0:
                sorted_results = sorted_results[:top_n]
            results.append([
                RerankResult(candidate=r.candidate, score=r.score, rank=i)
                for i, r in enumerate(sorted_results)
            ])
        return results
