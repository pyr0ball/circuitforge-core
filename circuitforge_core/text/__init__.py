"""
circuitforge_core.text — direct text generation service module.

Provides lightweight, low-overhead text generation that bypasses ollama/vllm
for products that need fast, frequent inference from small local models.

Quick start (mock mode — no model required):

    import os; os.environ["CF_TEXT_MOCK"] = "1"
    from circuitforge_core.text import generate, chat, ChatMessage

    result = generate("Write a short cover letter intro.")
    print(result.text)

    reply = chat([
        ChatMessage("system", "You are a helpful recipe assistant."),
        ChatMessage("user", "What can I make with eggs, spinach, and feta?"),
    ])
    print(reply.text)

Real inference (GGUF model):

    export CF_TEXT_MODEL=/Library/Assets/LLM/qwen2.5-3b-instruct-q4_k_m.gguf
    from circuitforge_core.text import generate
    result = generate("Summarise this job posting in 2 sentences: ...")

Backend selection (CF_TEXT_BACKEND env or explicit):

    from circuitforge_core.text import make_backend
    backend = make_backend("/path/to/model.gguf", backend="llamacpp")

cf-orch service profile:

    service_type:       cf-text
    max_mb:             per-model (3B Q4 ≈ 2048, 7B Q4 ≈ 4096)
    preferred_compute:  7.5 minimum (INT8 tensor cores)
    max_concurrent:     2
    shared:             true
"""
from __future__ import annotations

import os

from circuitforge_core.text.backends.base import (
    ChatMessage,
    GenerateResult,
    TextBackend,
    make_text_backend,
)
from circuitforge_core.text.backends.mock import MockTextBackend

# ── Process-level singleton backend ──────────────────────────────────────────
# Lazily initialised on first call to generate() or chat().
# Products that need per-user or per-request backends should use make_backend().

_backend: TextBackend | None = None


def _get_backend() -> TextBackend:
    global _backend
    if _backend is None:
        model_path = os.environ.get("CF_TEXT_MODEL", "mock")
        mock = model_path == "mock" or os.environ.get("CF_TEXT_MOCK", "") == "1"
        _backend = make_text_backend(model_path, mock=mock)
    return _backend


def make_backend(
    model_path: str,
    backend: str | None = None,
    mock: bool | None = None,
) -> TextBackend:
    """
    Create a TextBackend for the given model.

    Use this when you need a dedicated backend per request or per user,
    rather than the process-level singleton used by generate() and chat().
    """
    return make_text_backend(model_path, backend=backend, mock=mock)


# ── Convenience functions (singleton path) ────────────────────────────────────


def generate(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
    stop: list[str] | None = None,
):
    """
    Generate text from a prompt using the process-level backend.

    stream=True returns an Iterator[str] of tokens instead of GenerateResult.
    model is accepted for API symmetry with LLMRouter but ignored by the
    singleton path — set CF_TEXT_MODEL to change the loaded model.
    """
    backend = _get_backend()
    if stream:
        return backend.generate_stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)


def chat(
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> GenerateResult:
    """
    Chat completion using the process-level backend.

    messages should be a list of ChatMessage(role, content) objects.
    stream=True is not yet supported on the chat path; pass stream=False.
    """
    if stream:
        raise NotImplementedError(
            "stream=True is not yet supported for chat(). "
            "Use generate_stream() directly on a backend instance."
        )
    return _get_backend().chat(messages, max_tokens=max_tokens, temperature=temperature)


def reset_backend() -> None:
    """Reset the process-level singleton. Test teardown only."""
    global _backend
    _backend = None


__all__ = [
    "ChatMessage",
    "GenerateResult",
    "TextBackend",
    "MockTextBackend",
    "make_backend",
    "generate",
    "chat",
    "reset_backend",
]
