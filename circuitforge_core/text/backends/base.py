# circuitforge_core/text/backends/base.py — TextBackend Protocol + factory
#
# MIT licensed. The Protocol and mock backend are always importable.
# Real backends (LlamaCppBackend, TransformersBackend) require optional extras.
from __future__ import annotations

import os
from typing import AsyncIterator, Iterator, Protocol, runtime_checkable


# ── Shared result types ───────────────────────────────────────────────────────


class GenerateResult:
    """Result from a single non-streaming generate() call."""

    def __init__(self, text: str, tokens_used: int = 0, model: str = "") -> None:
        self.text = text
        self.tokens_used = tokens_used
        self.model = model

    def __repr__(self) -> str:
        return f"GenerateResult(text={self.text!r:.40}, tokens={self.tokens_used})"


class ChatMessage:
    """A single message in a chat conversation."""

    def __init__(self, role: str, content: str) -> None:
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role {role!r}. Must be system, user, or assistant.")
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ── TextBackend Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class TextBackend(Protocol):
    """
    Abstract interface for direct text generation backends.

    All generate/chat methods have both sync and async variants.
    Streaming variants yield str tokens rather than a complete result.

    Implementations must be safe to construct once and call concurrently
    (the model is loaded at construction time and reused across calls).
    """

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        """Synchronous generate — blocks until the full response is produced."""
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Synchronous streaming — yields tokens as they are produced."""
        ...

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerateResult:
        """Async generate — runs in thread pool, never blocks the event loop."""
        ...

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Async streaming — yields tokens without blocking the event loop."""
        ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> GenerateResult:
        """Chat completion — formats messages into a prompt and generates."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier for the loaded model (path stem or HF repo ID)."""
        ...

    @property
    def vram_mb(self) -> int:
        """Approximate VRAM footprint in MB. Used by cf-orch service registry."""
        ...


# ── Backend selection ─────────────────────────────────────────────────────────


def _select_backend(model_path: str, backend: str | None) -> str:
    """
    Return "llamacpp" or "transformers" for the given model path.

    Parameters
    ----------
    model_path  Path to the model file or HuggingFace repo ID (e.g. "Qwen/Qwen2.5-3B").
    backend     Explicit override from the caller ("llamacpp" | "transformers" | None).
                When provided, trust it without inspection.

    Return "llamacpp" or "transformers". Raise ValueError for unrecognised values.
    """
    _VALID = ("llamacpp", "transformers")

    # 1. Caller-supplied override — highest trust, no inspection needed.
    resolved = backend or os.environ.get("CF_TEXT_BACKEND")
    if resolved:
        if resolved not in _VALID:
            raise ValueError(
                f"CF_TEXT_BACKEND={resolved!r} is not valid. Choose: {', '.join(_VALID)}"
            )
        return resolved

    # 2. Format detection — GGUF files are unambiguously llama-cpp territory.
    if model_path.lower().endswith(".gguf"):
        return "llamacpp"

    # 3. Safe default — transformers covers HF repo IDs and safetensors dirs.
    return "transformers"


# ── Factory ───────────────────────────────────────────────────────────────────


def make_text_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
) -> "TextBackend":
    """
    Return a TextBackend for the given model.

    mock=True or CF_TEXT_MOCK=1  → MockTextBackend (no GPU, no model file needed)
    Otherwise                    → backend resolved via _select_backend()
    """
    use_mock = mock if mock is not None else os.environ.get("CF_TEXT_MOCK", "") == "1"
    if use_mock:
        from circuitforge_core.text.backends.mock import MockTextBackend
        return MockTextBackend(model_name=model_path)

    resolved = _select_backend(model_path, backend)

    if resolved == "llamacpp":
        from circuitforge_core.text.backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend(model_path=model_path)

    if resolved == "transformers":
        from circuitforge_core.text.backends.transformers import TransformersBackend
        return TransformersBackend(model_path=model_path)

    raise ValueError(f"Unknown backend {resolved!r}. Expected 'llamacpp' or 'transformers'.")
