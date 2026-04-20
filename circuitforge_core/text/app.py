"""
cf-text FastAPI service — managed by cf-orch.

Lightweight local text generation. Supports GGUF models via llama.cpp and
HuggingFace transformers. Sits alongside vllm/ollama for products that need
fast, frequent inference from small local models (3B–7B Q4).

Endpoints:
  GET  /health      → {"status": "ok", "model": str, "vram_mb": int, "backend": str}
  POST /generate    → GenerateResponse
  POST /chat        → GenerateResponse

Usage:
    python -m circuitforge_core.text.app \
        --model /Library/Assets/LLM/qwen2.5-3b-instruct-q4_k_m.gguf \
        --port 8006 \
        --gpu-id 0

Mock mode (no model or GPU required):
    CF_TEXT_MOCK=1 python -m circuitforge_core.text.app --port 8006
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from functools import partial

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from circuitforge_core.text.backends.base import ChatMessage as BackendChatMessage
from circuitforge_core.text.backends.base import make_text_backend

logger = logging.getLogger(__name__)

_backend = None


# ── Request / response models ─────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] | None = None


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageModel]
    max_tokens: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int = 0
    model: str = ""


# ── OpenAI-compat request / response (for LLMRouter openai_compat path) ──────


class OAIMessageModel(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = "cf-text"
    messages: list[OAIMessageModel]
    max_tokens: int | None = None
    temperature: float = 0.7
    stream: bool = False


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIMessageModel
    finish_reason: str = "stop"


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    model_path: str,
    gpu_id: int = 0,
    backend: str | None = None,
    mock: bool = False,
) -> FastAPI:
    global _backend

    if not mock and not model_path:
        raise ValueError(
            "cf-text: --model is required (got empty string). "
            "Pass a GGUF path, a HuggingFace model ID, or set CF_TEXT_MOCK=1 for mock mode."
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    _backend = make_text_backend(model_path, backend=backend, mock=mock)
    logger.info("cf-text ready: model=%r vram=%dMB", _backend.model_name, _backend.vram_mb)

    app = FastAPI(title="cf-text", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        return {
            "status": "ok",
            "model": _backend.model_name,
            "vram_mb": _backend.vram_mb,
        }

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> GenerateResponse:
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        result = await _backend.generate_async(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=req.stop,
        )
        return GenerateResponse(
            text=result.text,
            tokens_used=result.tokens_used,
            model=result.model,
        )

    @app.post("/chat")
    async def chat(req: ChatRequest) -> GenerateResponse:
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        messages = [BackendChatMessage(m.role, m.content) for m in req.messages]
        # chat() is sync-only in the Protocol; run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(_backend.chat, messages,
                    max_tokens=req.max_tokens, temperature=req.temperature),
        )
        return GenerateResponse(
            text=result.text,
            tokens_used=result.tokens_used,
            model=result.model,
        )

    @app.post("/v1/chat/completions")
    async def oai_chat_completions(req: OAIChatRequest) -> OAIChatResponse:
        """OpenAI-compatible chat completions endpoint.

        Allows LLMRouter (and any openai_compat client) to use cf-text
        without a custom backend type — just set base_url to this service's
        /v1 prefix.
        """
        if _backend is None:
            raise HTTPException(503, detail="backend not initialised")
        messages = [BackendChatMessage(m.role, m.content) for m in req.messages]
        max_tok = req.max_tokens or 512
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(_backend.chat, messages, max_tokens=max_tok, temperature=req.temperature),
        )
        return OAIChatResponse(
            id=f"cftext-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=result.model or req.model,
            choices=[OAIChoice(message=OAIMessageModel(role="assistant", content=result.text))],
            usage=OAIUsage(completion_tokens=result.tokens_used, total_tokens=result.tokens_used),
        )

    return app


# ── CLI entrypoint ────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cf-text inference server")
    parser.add_argument("--model", default=os.environ.get("CF_TEXT_MODEL", "mock"),
                        help="Path to GGUF file or HF model ID")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index to use")
    parser.add_argument("--backend", choices=["llamacpp", "transformers"], default=None)
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode (no model or GPU needed)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    args = _parse_args()
    mock = args.mock or os.environ.get("CF_TEXT_MOCK", "") == "1" or args.model == "mock"
    app = create_app(
        model_path=args.model,
        gpu_id=args.gpu_id,
        backend=args.backend,
        mock=mock,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
