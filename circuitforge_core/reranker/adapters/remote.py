# circuitforge_core/reranker/adapters/remote.py — HTTP remote reranker adapter
#
# Calls a cf-reranker service endpoint (cf-orch allocated or static URL).
# No model weights loaded locally — all inference runs on the remote node.
#
# MIT licensed.
from __future__ import annotations

import logging
from typing import Sequence

import requests

from circuitforge_core.reranker.base import TextReranker

logger = logging.getLogger(__name__)

# Default timeout for a single /rerank call (seconds).
# Large candidate lists may take longer — callers can pass timeout= explicitly.
_DEFAULT_TIMEOUT = 30


class RemoteTextReranker(TextReranker):
    """
    Reranker that delegates scoring to a remote cf-reranker HTTP service.

    The remote service must implement POST /rerank with the request body::

        {"query": str, "candidates": [str, ...], "top_n": int}

    and return::

        {"results": [{"candidate": str, "score": float, "rank": int}, ...]}

    cf-orch allocation (recommended — starts service on-demand):
        reranker = RemoteTextReranker.from_cf_orch(
            orch_url="http://10.1.10.71:7700",
            service="cf-reranker",
            model_candidates=["qwen3-0.6b"],
        )

    Static URL (e.g. dedicated node already running cf-reranker):
        reranker = RemoteTextReranker("http://10.1.10.10:8011")
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = _DEFAULT_TIMEOUT,
        _model_id: str = "remote",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model_id_str = _model_id

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @classmethod
    def from_cf_orch(
        cls,
        orch_url: str,
        service: str = "cf-reranker",
        model_candidates: list[str] | None = None,
        ttl_s: float = 3600.0,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> "RemoteTextReranker":
        """
        Allocate a cf-reranker service via cf-orch and return a configured adapter.

        Blocks until allocation succeeds or raises on failure. The returned
        adapter is valid for the duration of the TTL; create a new one if the
        lease expires.

        This is a one-shot allocation — the caller owns the lifetime. For
        long-running services, prefer the static URL constructor and let
        cf-orch manage the process independently.
        """
        try:
            from circuitforge_orch.client import CFOrchClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "circuitforge_orch is not installed — cannot allocate via cf-orch."
            ) from exc

        client = CFOrchClient(orch_url)
        ctx = client.allocate(
            service,
            model_candidates=model_candidates or [],
            ttl_s=ttl_s,
            caller="reranker-remote",
        )
        alloc = ctx.__enter__()
        # Note: caller is responsible for ctx.__exit__() when done.
        # We stash it on the instance so callers can call release().
        instance = cls(
            base_url=alloc.url,
            timeout=timeout,
            _model_id=f"remote:{service}",
        )
        instance._orch_ctx = ctx  # type: ignore[attr-defined]
        return instance

    def release(self) -> None:
        """Release the cf-orch allocation if this adapter was created via from_cf_orch()."""
        ctx = getattr(self, "_orch_ctx", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._orch_ctx = None  # type: ignore[attr-defined]

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        url = f"{self._base_url}/rerank"
        payload = {"query": query, "candidates": candidates, "top_n": 0}
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Remote reranker at {url!r} failed: {exc}"
            ) from exc

        data = resp.json()
        # Build a score-per-candidate list in the original order.
        score_map: dict[str, float] = {
            r["candidate"]: r["score"] for r in data["results"]
        }
        return [score_map.get(c, 0.0) for c in candidates]
