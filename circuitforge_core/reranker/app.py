"""
circuitforge_core.reranker.app — cf-reranker FastAPI service.

Managed by cf-orch as a process-type service. cf-orch starts this via:

    python -m circuitforge_core.reranker.app \
        --model BAAI/bge-reranker-base \
        --backend bge \
        --port 8011 \
        --gpu-id 0

Or with Qwen3:

    python -m circuitforge_core.reranker.app \
        --model Qwen/Qwen3-Reranker-0.6B \
        --backend qwen3 \
        --port 8011 \
        --gpu-id 0 \
        --dtype float16

Endpoints:
    GET  /health   → {"status": "ok", "model": "...", "backend": "...", "vram_mb": n}
    POST /rerank   → RerankResponse
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Request / response models ─────────────────────────────────────────────────

class RerankRequest(BaseModel):
    query: str
    candidates: list[str]
    top_n: int = 0


class RerankResultItem(BaseModel):
    candidate: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    results: list[RerankResultItem]
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    vram_mb: int


# ── VRAM estimates by backend/model family ────────────────────────────────────

_VRAM_TABLE: dict[str, int] = {
    "bge-reranker-base":       570,
    "bge-reranker-large":      1300,
    "bge-reranker-v2-m3":      570,
    "mxbai-rerank-base-v1":    570,
    "mxbai-rerank-large-v1":   1300,
    "ms-marco-MiniLM-L-6-v2":  90,
    "ms-marco-MiniLM-L-12-v2": 130,
    "Qwen3-Reranker-0.6B":     1200,
    "Qwen3-Reranker-1.5B":     3000,
    "Qwen3-Reranker-8B":       16000,
}

def _estimate_vram(model_id: str) -> int:
    for key, mb in _VRAM_TABLE.items():
        if key in model_id:
            return mb
    return 1024  # safe default


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(model_id: str, backend: str, dtype: str, mock: bool) -> FastAPI:
    from circuitforge_core.reranker import make_reranker

    app = FastAPI(title="cf-reranker", version="0.1.0")
    _reranker = make_reranker(model_id=model_id, backend=backend, mock=mock)
    _vram_mb = _estimate_vram(model_id)

    logger.info("cf-reranker ready: model=%r backend=%r vram=%dMB", model_id, backend, _vram_mb)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=_reranker.model_id,
            backend=backend,
            vram_mb=_vram_mb,
        )

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(req: RerankRequest) -> RerankResponse:
        if not req.candidates:
            raise HTTPException(status_code=400, detail="candidates must not be empty")
        try:
            results = _reranker.rerank(req.query, req.candidates, top_n=req.top_n)
        except Exception as exc:
            logger.exception("rerank failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return RerankResponse(
            results=[
                RerankResultItem(candidate=r.candidate, score=r.score, rank=r.rank)
                for r in results
            ],
            model=_reranker.model_id,
        )

    return app


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="cf-reranker — CircuitForge reranker service")
    parser.add_argument(
        "--model", default="BAAI/bge-reranker-base",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--backend", default="bge",
        choices=["bge", "qwen3", "cross-encoder", "mock"],
        help="Reranker backend",
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--mock", action="store_true",
                        help="Run with mock backend (no GPU, for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )

    if args.backend != "mock" and not args.mock:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    mock = args.mock or os.environ.get("CF_RERANKER_MOCK", "") == "1"
    app = create_app(
        model_id=args.model,
        backend=args.backend,
        dtype=args.dtype,
        mock=mock,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
