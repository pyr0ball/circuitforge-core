# circuitforge_core/reranker/adapters/cross_encoder.py — sentence-transformers CrossEncoder
#
# Requires: pip install circuitforge-core[reranker-cross-encoder]
#
# Covers models not in the FlagEmbedding ecosystem:
#   mixedbread-ai/mxbai-rerank-base-v1       ~570MB VRAM, strong general-purpose
#   mixedbread-ai/mxbai-rerank-large-v1      ~1.3GB VRAM, higher quality
#   cross-encoder/ms-marco-MiniLM-L-6-v2    ~90MB,  fast, English-only
#   cross-encoder/ms-marco-MiniLM-L-12-v2   ~130MB, balanced
#   jinaai/jina-reranker-v2-base-multilingual ~280MB, multilingual
#
# MIT licensed.
from __future__ import annotations

import logging
import threading
from typing import Sequence

from circuitforge_core.reranker.base import TextReranker

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder  # type: ignore[import]
except ImportError:
    _CrossEncoder = None  # type: ignore[assignment]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class CrossEncoderTextReranker(TextReranker):
    """
    Cross-encoder reranker using the sentence-transformers CrossEncoder class.

    Broader model compatibility than BGETextReranker — any HuggingFace model
    with a sequence-classification head works here. Particularly well-suited
    for the mxbai-rerank and ms-marco families.

    Usage:
        reranker = CrossEncoderTextReranker("mixedbread-ai/mxbai-rerank-base-v1")
        results = reranker.rerank("chicken soup recipe", ["recipe 1...", "recipe 2..."])
    """

    def __init__(
        self,
        model_id: str = "mixedbread-ai/mxbai-rerank-base-v1",
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._model: object | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    def load(self) -> None:
        """Explicitly load model weights. Called automatically on first rerank()."""
        if _CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install circuitforge-core[reranker-cross-encoder]"
            )
        with self._lock:
            if self._model is not None:
                return
            device = "cuda" if _cuda_available() else "cpu"
            logger.info(
                "Loading CrossEncoder reranker: %s (device=%s)", self._model_id, device
            )
            self._model = _CrossEncoder(
                self._model_id,
                max_length=self._max_length,
                device=device,
            )

    def unload(self) -> None:
        """Release model weights."""
        with self._lock:
            self._model = None

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        if self._model is None:
            self.load()
        pairs = [(query, c) for c in candidates]
        with self._lock:
            raw = self._model.predict(pairs)  # type: ignore[union-attr]
        # predict() returns a numpy array or list; normalise to plain floats.
        return [float(s) for s in raw]
