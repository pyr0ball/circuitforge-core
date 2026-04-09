# circuitforge_core/pipeline/multimodal.py — cf-docuvision + cf-text pipeline
#
# MIT — orchestration only; vision and text inference stay in their own modules.
#
# Usage (minimal):
#
#   from circuitforge_core.pipeline.multimodal import MultimodalPipeline, MultimodalConfig
#
#   pipe = MultimodalPipeline(MultimodalConfig())
#   for result in pipe.run(page_bytes_list):
#       print(f"Page {result.page_idx}: {result.generated[:80]}")
#
# Streaming (token-by-token):
#
#   for page_idx, token in pipe.stream(page_bytes_list):
#       ui.append(page_idx, token)
#
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from circuitforge_core.documents.client import DocuvisionClient
from circuitforge_core.documents.models import StructuredDocument

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def _default_prompt(page_idx: int, doc: StructuredDocument) -> str:
    """Build a generation prompt from a StructuredDocument."""
    header = f"[Page {page_idx + 1}]\n" if page_idx > 0 else ""
    return header + doc.raw_text


@dataclass
class MultimodalConfig:
    """Configuration for MultimodalPipeline.

    vision_url:
        Base URL of the cf-docuvision service.
    hint:
        Docuvision extraction hint — ``"auto"`` | ``"document"`` | ``"form"``
        | ``"table"`` | ``"figure"``.
    max_tokens:
        Passed to cf-text generate per page.
    temperature:
        Sampling temperature for text generation.
    vram_serialise:
        When True, ``swap_fn`` is called between the vision and text steps
        on each page.  Use this on 8GB GPUs where Dolphin-v2 and the text
        model cannot be resident simultaneously.
    prompt_fn:
        Callable ``(page_idx, StructuredDocument) -> str`` that builds the
        generation prompt.  Defaults to using ``doc.raw_text`` directly.
        Products override this to add system context, few-shot examples, etc.
    vision_timeout:
        HTTP timeout in seconds for each cf-docuvision request.
    """
    vision_url: str = "http://localhost:8003"
    hint: str = "auto"
    max_tokens: int = 512
    temperature: float = 0.7
    vram_serialise: bool = False
    prompt_fn: Callable[[int, StructuredDocument], str] = field(
        default_factory=lambda: _default_prompt
    )
    vision_timeout: int = 60


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    """Result of processing one page through the vision + text pipeline.

    page_idx:
        Zero-based page index.
    doc:
        StructuredDocument from cf-docuvision.
    generated:
        Full text output from cf-text for this page.
    error:
        Non-None if extraction or generation failed for this page.
    """
    page_idx: int
    doc: StructuredDocument | None
    generated: str
    error: str | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MultimodalPipeline:
    """Chunk a multi-page document through vision extraction + text generation.

    Parameters
    ----------
    config:
        Pipeline configuration.
    swap_fn:
        Optional callable with no arguments, called between the vision and text
        steps on each page when ``config.vram_serialise=True``.  Products using
        cf-orch wire this to the VRAM budget API so Dolphin-v2 can offload
        before the text model loads.  A no-op lambda works for testing.
    generate_fn:
        Text generation callable: ``(prompt, max_tokens, temperature) -> str``.
        Defaults to ``circuitforge_core.text.generate``.  Override in tests or
        when the product manages its own text backend.
    stream_fn:
        Streaming text callable: ``(prompt, max_tokens, temperature) -> Iterator[str]``.
        Defaults to ``circuitforge_core.text.generate`` with ``stream=True``.
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        *,
        swap_fn: Callable[[], None] | None = None,
        generate_fn: Callable[..., str] | None = None,
        stream_fn: Callable[..., Iterator[str]] | None = None,
    ) -> None:
        self._cfg = config or MultimodalConfig()
        self._vision = DocuvisionClient(
            base_url=self._cfg.vision_url,
            timeout=self._cfg.vision_timeout,
        )
        self._swap_fn = swap_fn
        self._generate_fn = generate_fn
        self._stream_fn = stream_fn

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, pages: Iterable[bytes]) -> Iterator[PageResult]:
        """Process each page and yield a PageResult as soon as it is ready.

        Callers receive pages one at a time — the UI can begin rendering
        page 0 while pages 1..N are still being extracted and generated.
        """
        for page_idx, page_bytes in enumerate(pages):
            yield self._process_page(page_idx, page_bytes)

    def stream(self, pages: Iterable[bytes]) -> Iterator[tuple[int, str]]:
        """Yield ``(page_idx, token)`` tuples for token-level progressive rendering.

        Each page is fully extracted before text generation begins, but tokens
        are yielded as the text model produces them rather than waiting for the
        full page output.
        """
        for page_idx, page_bytes in enumerate(pages):
            doc, err = self._extract(page_idx, page_bytes)
            if err:
                yield (page_idx, f"[extraction error: {err}]")
                continue

            self._maybe_swap()

            prompt = self._cfg.prompt_fn(page_idx, doc)
            try:
                for token in self._stream_tokens(prompt):
                    yield (page_idx, token)
            except Exception as exc:
                log.error("page %d text streaming failed: %s", page_idx, exc)
                yield (page_idx, f"[generation error: {exc}]")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_page(self, page_idx: int, page_bytes: bytes) -> PageResult:
        doc, err = self._extract(page_idx, page_bytes)
        if err:
            return PageResult(page_idx=page_idx, doc=None, generated="", error=err)

        self._maybe_swap()

        prompt = self._cfg.prompt_fn(page_idx, doc)
        try:
            text = self._generate(prompt)
        except Exception as exc:
            log.error("page %d generation failed: %s", page_idx, exc)
            return PageResult(page_idx=page_idx, doc=doc, generated="",
                              error=str(exc))

        return PageResult(page_idx=page_idx, doc=doc, generated=text)

    def _extract(
        self, page_idx: int, page_bytes: bytes
    ) -> tuple[StructuredDocument | None, str | None]:
        try:
            doc = self._vision.extract(page_bytes, hint=self._cfg.hint)
            log.debug("page %d extracted: %d chars", page_idx, len(doc.raw_text))
            return doc, None
        except Exception as exc:
            log.error("page %d vision extraction failed: %s", page_idx, exc)
            return None, str(exc)

    def _maybe_swap(self) -> None:
        if self._cfg.vram_serialise and self._swap_fn is not None:
            log.debug("vram_serialise: calling swap_fn")
            self._swap_fn()

    def _generate(self, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(
                prompt,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
            )
        from circuitforge_core.text import generate
        result = generate(
            prompt,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
        )
        return result.text

    def _stream_tokens(self, prompt: str) -> Iterator[str]:
        if self._stream_fn is not None:
            yield from self._stream_fn(
                prompt,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
            )
            return
        from circuitforge_core.text import generate
        tokens = generate(
            prompt,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            stream=True,
        )
        yield from tokens
