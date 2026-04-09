"""
circuitforge_core.stt.app — cf-stt FastAPI service.

Managed by cf-orch as a process-type service. cf-orch starts this via:

    python -m circuitforge_core.stt.app \
        --model /Library/Assets/LLM/whisper/models/Whisper/faster-whisper/models--Systran--faster-whisper-medium/snapshots/<hash> \
        --port 8004 \
        --gpu-id 0

Endpoints:
    GET  /health       → {"status": "ok", "model": "<name>", "vram_mb": <n>}
    POST /transcribe   → STTTranscribeResponse (multipart: audio file)

Audio format: any format ffmpeg understands (WAV, MP3, OGG, FLAC).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from circuitforge_core.stt.backends.base import STTResult, make_stt_backend

logger = logging.getLogger(__name__)

# ── Response model (mirrors circuitforge_orch.contracts.stt.STTTranscribeResponse) ──

class TranscribeResponse(BaseModel):
    text: str
    confidence: float
    below_threshold: bool
    language: str | None = None
    duration_s: float | None = None
    segments: list[dict] = []
    model: str = ""


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(
    model_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
    confidence_threshold: float = STTResult.CONFIDENCE_DEFAULT_THRESHOLD,
    mock: bool = False,
) -> FastAPI:
    app = FastAPI(title="cf-stt", version="0.1.0")
    backend = make_stt_backend(
        model_path, device=device, compute_type=compute_type, mock=mock
    )
    logger.info("cf-stt ready: model=%r vram=%dMB", backend.model_name, backend.vram_mb)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": backend.model_name, "vram_mb": backend.vram_mb}

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe(
        audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, ...)"),
        language: str | None = Form(None, description="BCP-47 language code hint, e.g. 'en'"),
        confidence_threshold_override: float | None = Form(
            None,
            description="Override default confidence threshold for this request.",
        ),
    ) -> TranscribeResponse:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        threshold = confidence_threshold_override or confidence_threshold
        try:
            result = backend.transcribe(
                audio_bytes, language=language, confidence_threshold=threshold
            )
        except Exception as exc:
            logger.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return TranscribeResponse(
            text=result.text,
            confidence=result.confidence,
            below_threshold=result.below_threshold,
            language=result.language,
            duration_s=result.duration_s,
            segments=[
                {
                    "start_s": s.start_s,
                    "end_s": s.end_s,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in result.segments
            ],
            model=result.model,
        )

    return app


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="cf-stt — CircuitForge STT service")
    parser.add_argument("--model", required=True,
                        help="Model path or size name (e.g. 'medium', or full local path)")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "int8", "int8_float16", "float32"],
                        help="Quantisation / compute type passed to faster-whisper")
    parser.add_argument("--confidence-threshold", type=float,
                        default=STTResult.CONFIDENCE_DEFAULT_THRESHOLD)
    parser.add_argument("--mock", action="store_true",
                        help="Run with mock backend (no GPU, for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )

    # Let cf-orch pass --gpu-id; map to CUDA_VISIBLE_DEVICES so the process
    # only sees its assigned GPU. This prevents accidental multi-GPU usage.
    if args.device == "cuda" and not args.mock:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    mock = args.mock or os.environ.get("CF_STT_MOCK", "") == "1"
    app = create_app(
        model_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        confidence_threshold=args.confidence_threshold,
        mock=mock,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
