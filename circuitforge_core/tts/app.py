"""
cf-tts FastAPI service — managed by cf-orch.

Endpoints:
  GET  /health       → {"status": "ok", "model": str, "vram_mb": int}
  POST /synthesize   → audio bytes (Content-Type: audio/ogg or audio/wav or audio/mpeg)

Usage:
    python -m circuitforge_core.tts.app \
        --model /Library/Assets/LLM/chatterbox/hub/models--ResembleAI--chatterbox-turbo/snapshots/<hash> \
        --port 8005 \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import os
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response

from circuitforge_core.tts.backends.base import AudioFormat, TTSBackend, make_tts_backend

_CONTENT_TYPES: dict[str, str] = {
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}

app = FastAPI(title="cf-tts")
_backend = None  # type: TTSBackend | None


@app.get("/health")
def health() -> dict:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    return {"status": "ok", "model": _backend.model_name, "vram_mb": _backend.vram_mb}


@app.post("/synthesize")
async def synthesize(
    text: Annotated[str, Form()],
    format: Annotated[AudioFormat, Form()] = "ogg",
    exaggeration: Annotated[float, Form()] = 0.5,
    cfg_weight: Annotated[float, Form()] = 0.5,
    temperature: Annotated[float, Form()] = 0.8,
    audio_prompt: UploadFile | None = None,
) -> Response:
    if _backend is None:
        raise HTTPException(503, detail="backend not initialised")
    if not text.strip():
        raise HTTPException(422, detail="text must not be empty")

    prompt_bytes: bytes | None = None
    if audio_prompt is not None:
        prompt_bytes = await audio_prompt.read()

    result = _backend.synthesize(
        text,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        audio_prompt=prompt_bytes,
        format=format,
    )
    return Response(
        content=result.audio_bytes,
        media_type=_CONTENT_TYPES.get(result.format, "audio/ogg"),
        headers={
            "X-Duration-S": str(round(result.duration_s, 3)),
            "X-Model": result.model,
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cf-tts service")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8005)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    mock = args.mock or args.model == "mock"
    device = "cpu" if mock else "cuda"

    _backend = make_tts_backend(args.model, mock=mock, device=device)
    print(f"cf-tts backend ready: {_backend.model_name} ({_backend.vram_mb} MB)")

    uvicorn.run(app, host=args.host, port=args.port)
