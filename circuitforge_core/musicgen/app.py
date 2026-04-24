"""
cf-musicgen FastAPI service — managed by cf-orch.

Endpoints:
  GET  /health     -> {"status": "ok", "model": str, "vram_mb": int}
  POST /continue   -> audio bytes (Content-Type: audio/wav or audio/mpeg)

Usage:
    python -m circuitforge_core.musicgen.app \
        --model facebook/musicgen-melody \
        --port 8006 \
        --gpu-id 0

The service streams back raw audio bytes. Headers include:
  X-Duration-S      generated duration in seconds
  X-Prompt-Duration-S   how many seconds of the input were used as prompt
  X-Model           model name
  X-Sample-Rate     output sample rate (32000 for all MusicGen variants)

Model weights are cached at /Library/Assets/LLM/musicgen/.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from circuitforge_core.musicgen.backends.base import (
    MODEL_MELODY,
    MODEL_SMALL,
    AudioFormat,
    MusicGenBackend,
    make_musicgen_backend,
)

_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}

app = FastAPI(title="cf-musicgen", version="0.1.0")
_backend: MusicGenBackend | None = None


@app.get("/health")
def health() -> dict:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    return {
        "status": "ok",
        "model": _backend.model_name,
        "vram_mb": _backend.vram_mb,
    }


@app.post("/continue")
async def continue_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, ...)"),
    description: Annotated[str | None, Form()] = None,
    duration_s: Annotated[float, Form()] = 15.0,
    prompt_duration_s: Annotated[float, Form()] = 10.0,
    format: Annotated[AudioFormat, Form()] = "wav",
) -> Response:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    if duration_s <= 0 or duration_s > 60:
        raise HTTPException(422, detail="duration_s must be between 0 and 60")
    if prompt_duration_s <= 0 or prompt_duration_s > 30:
        raise HTTPException(422, detail="prompt_duration_s must be between 0 and 30")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, detail="Empty audio file")

    try:
        result = _backend.continue_audio(
            audio_bytes,
            description=description or None,
            duration_s=duration_s,
            prompt_duration_s=prompt_duration_s,
            format=format,
        )
    except Exception as exc:
        logging.exception("Music continuation failed")
        raise HTTPException(500, detail=str(exc)) from exc

    return Response(
        content=result.audio_bytes,
        media_type=_CONTENT_TYPES.get(result.format, "audio/wav"),
        headers={
            "X-Duration-S": str(round(result.duration_s, 3)),
            "X-Prompt-Duration-S": str(round(result.prompt_duration_s, 3)),
            "X-Model": result.model,
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cf-musicgen service")
    p.add_argument(
        "--model",
        default=MODEL_MELODY,
        choices=[MODEL_MELODY, MODEL_SMALL, "facebook/musicgen-medium", "facebook/musicgen-large"],
        help="MusicGen model variant",
    )
    p.add_argument("--port", type=int, default=8006)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--gpu-id", type=int, default=0,
                   help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--mock", action="store_true",
                   help="Run with mock backend (no GPU, for testing)")
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )
    args = _parse_args()

    if args.device == "cuda" and not args.mock:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    mock = args.mock or args.model == "mock"
    device = "cpu" if mock else args.device

    _backend = make_musicgen_backend(model_name=args.model, mock=mock, device=device)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
