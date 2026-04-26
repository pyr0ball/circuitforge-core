# circuitforge_core/reranker/adapters/cohere.py — Cohere Rerank API (BYOK cloud)
#
# Requires: pip install circuitforge-core[reranker-cohere]
# API key:  set COHERE_API_KEY env var, or pass api_key= explicitly.
#
# Models (as of 2026):
#   rerank-english-v3.0      English-only, highest quality
#   rerank-multilingual-v3.0 Multilingual
#   rerank-english-v2.0      Legacy, lower cost
#
# BYOK unlock path: free-tier users who supply their own Cohere key get cloud
# reranking without needing a cf-orch node. Same pattern as the Anthropic
# backend in LLMRouter.
#
# MIT licensed.
from __future__ import annotations

import logging
import os
from typing import Sequence

from circuitforge_core.reranker.base import TextReranker

logger = logging.getLogger(__name__)

try:
    import cohere as _cohere  # type: ignore[import]
except ImportError:
    _cohere = None  # type: ignore[assignment]

_DEFAULT_MODEL = "rerank-english-v3.0"


class CohereTextReranker(TextReranker):
    """
    Cloud reranker backed by the Cohere Rerank API.

    BYOK (bring your own key): pass api_key= or set COHERE_API_KEY in the
    environment. No model weights loaded locally.

    Usage:
        reranker = CohereTextReranker()  # reads COHERE_API_KEY from env
        results = reranker.rerank("chicken soup recipe", ["recipe 1...", "recipe 2..."])

    With an explicit key and model:
        reranker = CohereTextReranker(
            api_key="co-...",
            model="rerank-multilingual-v3.0",
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_chunks_per_doc: int = 1,
    ) -> None:
        self._api_key_arg = api_key
        self._model = model
        self._max_chunks_per_doc = max_chunks_per_doc

    @property
    def model_id(self) -> str:
        return f"cohere:{self._model}"

    def _get_client(self) -> object:
        if _cohere is None:
            raise ImportError(
                "cohere is not installed. "
                "Run: pip install circuitforge-core[reranker-cohere]"
            )
        api_key = self._api_key_arg or os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Cohere API key is not set. "
                "Pass api_key= to CohereTextReranker or set COHERE_API_KEY."
            )
        return _cohere.Client(api_key=api_key)

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        client = self._get_client()
        response = client.rerank(  # type: ignore[union-attr]
            query=query,
            documents=candidates,
            model=self._model,
            top_n=len(candidates),
            max_chunks_per_doc=self._max_chunks_per_doc,
        )
        # response.results is sorted by relevance_score desc; rebuild
        # in original candidate order so TextReranker.rerank() re-sorts correctly.
        score_map: dict[int, float] = {
            r.index: r.relevance_score for r in response.results
        }
        return [score_map.get(i, 0.0) for i in range(len(candidates))]
