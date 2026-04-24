# circuitforge_core/text/backends/ollama.py — Ollama proxy backend for cf-text
#
# Routes inference requests to a running Ollama instance via its HTTP API.
# cf-text itself holds no GPU memory; Ollama manages the model and VRAM.
#
# Model path format: "ollama://<model-name>"  e.g. "ollama://llama3.1:8b"
# The "ollama://" prefix is stripped before forwarding to the API.
#
# Environment:
#   CF_TEXT_OLLAMA_URL   Base URL of the Ollama server (default: http://localhost:11434)
#
# MIT licensed.
from __future__ import annotations

import json as _json
import logging
import os
import time
from typing import AsyncIterator, Iterator

import httpx

from circuitforge_core.text.backends.base import GenerateResult

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaBackend:
    """
    cf-text backend that proxies inference to a local Ollama instance.

    This backend holds no GPU memory itself — Ollama owns the model and VRAM.
    vram_mb is therefore reported as 0 so cf-orch does not double-count VRAM
    against the separate ollama service budget.

    Supports /generate, /chat, and /v1/chat/completions (via generate/chat).
    Streaming is implemented for all variants.
    """

    def __init__(self, model_path: str, *, vram_mb: int = 0) -> None:
        # Strip the "ollama://" prefix from catalog paths
        self._model = model_path.removeprefix("ollama://")
        self._url = os.environ.get("CF_TEXT_OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")
        self._vram_mb = vram_mb
        logger.info("OllamaBackend: model=%r url=%r", self._model, self._url)

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def vram_mb(self) -> int:
        # Ollama manages its own VRAM; cf-text holds nothing.
        return self._vram_mb

    # ── Synchronous interface ─────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        t0 = time.monotonic()
        payload: dict = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = stop
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self._url}/api/generate", json=payload)
            resp.raise_for_status()
        data = resp.json()
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return GenerateResult(
            text=data.get("response", ""),
            tokens_used=data.get("eval_count", 0),
            model=self._model,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        payload: dict = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = stop
        with httpx.Client(timeout=180.0) as client:
            with client.stream("POST", f"{self._url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        t0 = time.monotonic()
        payload: dict = {
            "model":    self._model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self._url}/api/chat", json=payload)
            resp.raise_for_status()
        data = resp.json()
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return GenerateResult(
            text=data.get("message", {}).get("content", ""),
            tokens_used=data.get("eval_count", 0),
            model=self._model,
        )

    # ── Async interface ───────────────────────────────────────────────────────

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        t0 = time.monotonic()
        payload: dict = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = stop
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{self._url}/api/generate", json=payload)
            resp.raise_for_status()
        data = resp.json()
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        return GenerateResult(
            text=data.get("response", ""),
            tokens_used=data.get("eval_count", 0),
            model=self._model,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if stop:
            payload["options"]["stop"] = stop
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", f"{self._url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = _json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
