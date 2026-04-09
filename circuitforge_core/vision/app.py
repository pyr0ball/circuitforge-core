"""
circuitforge_core.vision.app — cf-vision FastAPI service.

Managed by cf-orch as a process-type service. cf-orch starts this via:

    python -m circuitforge_core.vision.app \
        --model google/siglip-so400m-patch14-384 \
        --backend siglip \
        --port 8006 \
        --gpu-id 0

For VLM inference (caption/VQA):

    python -m circuitforge_core.vision.app \
        --model vikhyatk/moondream2 \
        --backend vlm \
        --port 8006 \
        --gpu-id 0

Endpoints:
    GET  /health       → {"status": "ok", "model": "...", "vram_mb": n,
                          "supports_embed": bool, "supports_caption": bool}
    POST /classify     → VisionClassifyResponse  (multipart: image + labels)
    POST /embed        → VisionEmbedResponse     (multipart: image)
    POST /caption      → VisionCaptionResponse   (multipart: image + prompt)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from circuitforge_core.vision.backends.base import make_vision_backend

logger = logging.getLogger(__name__)


# ── Response models ───────────────────────────────────────────────────────────

class VisionClassifyResponse(BaseModel):
    labels: list[str]
    scores: list[float]
    model: str


class VisionEmbedResponse(BaseModel):
    embedding: list[float]
    model: str


class VisionCaptionResponse(BaseModel):
    caption: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    vram_mb: int
    backend: str
    supports_embed: bool
    supports_caption: bool


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(
    model_path: str,
    backend: str = "siglip",
    device: str = "cuda",
    dtype: str = "float16",
    mock: bool = False,
) -> FastAPI:
    app = FastAPI(title="cf-vision", version="0.1.0")
    _backend = make_vision_backend(
        model_path, backend=backend, device=device, dtype=dtype, mock=mock
    )
    logger.info(
        "cf-vision ready: model=%r backend=%r vram=%dMB",
        _backend.model_name, backend, _backend.vram_mb,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=_backend.model_name,
            vram_mb=_backend.vram_mb,
            backend=backend,
            supports_embed=_backend.supports_embed,
            supports_caption=_backend.supports_caption,
        )

    @app.post("/classify", response_model=VisionClassifyResponse)
    async def classify(
        image: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, ...)"),
        labels: str = Form(
            ...,
            description=(
                "Candidate labels — either a JSON array "
                '(["cat","dog"]) or comma-separated (cat,dog)'
            ),
        ),
    ) -> VisionClassifyResponse:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        parsed_labels = _parse_labels(labels)
        if not parsed_labels:
            raise HTTPException(status_code=400, detail="At least one label is required")

        try:
            result = _backend.classify(image_bytes, parsed_labels)
        except Exception as exc:
            logger.exception("classify failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return VisionClassifyResponse(
            labels=result.labels, scores=result.scores, model=result.model
        )

    @app.post("/embed", response_model=VisionEmbedResponse)
    async def embed_image(
        image: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, ...)"),
    ) -> VisionEmbedResponse:
        if not _backend.supports_embed:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Backend '{backend}' does not support embedding. "
                    "Use backend=siglip for embed()."
                ),
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        try:
            result = _backend.embed(image_bytes)
        except Exception as exc:
            logger.exception("embed failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return VisionEmbedResponse(embedding=result.embedding or [], model=result.model)

    @app.post("/caption", response_model=VisionCaptionResponse)
    async def caption_image(
        image: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, ...)"),
        prompt: str = Form(
            "",
            description="Optional instruction / question for the VLM",
        ),
    ) -> VisionCaptionResponse:
        if not _backend.supports_caption:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"Backend '{backend}' does not support caption generation. "
                    "Use backend=vlm for caption()."
                ),
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        try:
            result = _backend.caption(image_bytes, prompt=prompt)
        except Exception as exc:
            logger.exception("caption failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return VisionCaptionResponse(caption=result.caption or "", model=result.model)

    return app


# ── Label parsing ─────────────────────────────────────────────────────────────

def _parse_labels(raw: str) -> list[str]:
    """Accept JSON array or comma-separated label string."""
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return [lbl.strip() for lbl in stripped.split(",") if lbl.strip()]


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="cf-vision — CircuitForge vision service")
    parser.add_argument(
        "--model",
        default="google/siglip-so400m-patch14-384",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--backend", default="siglip", choices=["siglip", "vlm"],
        help="Vision backend: siglip (classify+embed) or vlm (caption+classify)",
    )
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mock", action="store_true",
                        help="Run with mock backend (no GPU, for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )

    if args.device == "cuda" and not args.mock:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    mock = args.mock or os.environ.get("CF_VISION_MOCK", "") == "1"
    app = create_app(
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        mock=mock,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
