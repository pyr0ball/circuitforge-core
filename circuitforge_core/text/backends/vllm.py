# circuitforge_core/text/backends/vllm.py — vllm proxy backend for cf-text
#
# Routes inference requests to a running vllm instance via its OpenAI-compatible
# HTTP API (/v1/chat/completions, /v1/completions).
# cf-text itself holds no GPU memory; vllm manages the model and VRAM.
#
# Model path format: "vllm://<model-id>"  e.g. "vllm://Qwen/Qwen2.5-7B-Instruct"
# The "vllm://" prefix is stripped; the remainder is the model_id sent to vllm.
#
# Environment:
#   CF_TEXT_VLLM_URL   Base URL of the vllm server (default: http://localhost:8000)
#
# MIT licensed.
from __future__ import annotations

import json as _json
import logging
import os
import time
from typing import AsyncIterator, Iterator

import httpx

from circuitforge_core.text.backends.base import ChatMessage, GenerateResult

logger = logging.getLogger(__name__)

_DEFAULT_VLLM_URL = "http://localhost:8000"


class VllmBackend:
    """
    cf-text backend that proxies inference to a local vllm instance.

    vllm exposes an OpenAI-compatible API (/v1/chat/completions).
    This backend holds no GPU memory — vllm owns the model and VRAM.
    vram_mb is reported as 0 so cf-orch does not double-count VRAM
    against the separate vllm service budget.
    """

    def __init__(self, model_path: str, *, vram_mb: int = 0) -> None:
        # Strip the "vllm://" prefix from catalog paths
        self._model = model_path.removeprefix("vllm://")
        self._url = os.environ.get("CF_TEXT_VLLM_URL", _DEFAULT_VLLM_URL).rstrip("/")
        self._vram_mb = vram_mb
        logger.info("VllmBackend: model=%r url=%r", self._model, self._url)

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def vram_mb(self) -> int:
        # vllm manages its own VRAM; cf-text holds nothing.
        return self._vram_mb

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _chat_payload(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        stream: bool,
    ) -> dict:
        payload: dict = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      stream,
        }
        if stop:
            payload["stop"] = stop
        return payload

    def _prompt_as_messages(self, prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

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
        payload = self._chat_payload(
            self._prompt_as_messages(prompt),
            max_tokens=max_tokens, temperature=temperature, stop=stop, stream=False,
        )
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self._url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        data = resp.json()
        return GenerateResult(
            text=data["choices"][0]["message"]["content"],
            tokens_used=data.get("usage", {}).get("completion_tokens", 0),
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
        payload = self._chat_payload(
            self._prompt_as_messages(prompt),
            max_tokens=max_tokens, temperature=temperature, stop=stop, stream=True,
        )
        with httpx.Client(timeout=180.0) as client:
            with client.stream("POST", f"{self._url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    token = _parse_sse_token(line)
                    if token:
                        yield token

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        dicts = [m.to_dict() if hasattr(m, "to_dict") else m for m in messages]
        payload = self._chat_payload(
            dicts, max_tokens=max_tokens, temperature=temperature, stop=None, stream=False,
        )
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{self._url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        data = resp.json()
        return GenerateResult(
            text=data["choices"][0]["message"]["content"],
            tokens_used=data.get("usage", {}).get("completion_tokens", 0),
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
        payload = self._chat_payload(
            self._prompt_as_messages(prompt),
            max_tokens=max_tokens, temperature=temperature, stop=stop, stream=False,
        )
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{self._url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        data = resp.json()
        return GenerateResult(
            text=data["choices"][0]["message"]["content"],
            tokens_used=data.get("usage", {}).get("completion_tokens", 0),
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
        payload = self._chat_payload(
            self._prompt_as_messages(prompt),
            max_tokens=max_tokens, temperature=temperature, stop=stop, stream=True,
        )
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", f"{self._url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    token = _parse_sse_token(line)
                    if token:
                        yield token


# ── SSE parser (OpenAI/vllm format) ──────────────────────────────────────────

def _parse_sse_token(line: str) -> str:
    """Extract content token from an OpenAI-format SSE line.

    Lines look like:  data: {"choices": [{"delta": {"content": "word"}}]}
    Terminal line:    data: [DONE]
    Returns the token string, or "" for empty/done/non-data lines.
    """
    if not line.startswith("data:"):
        return ""
    payload = line[5:].strip()
    if payload == "[DONE]":
        return ""
    try:
        chunk = _json.loads(payload)
        return chunk["choices"][0]["delta"].get("content", "") or ""
    except (KeyError, IndexError, _json.JSONDecodeError):
        return ""
