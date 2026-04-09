"""Tests for pipeline.MultimodalPipeline — mock vision and text backends."""
import pytest
from unittest.mock import MagicMock, patch
from circuitforge_core.documents.models import Element, StructuredDocument
from circuitforge_core.pipeline.multimodal import (
    MultimodalConfig,
    MultimodalPipeline,
    PageResult,
    _default_prompt,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc(text="extracted text", page=0) -> StructuredDocument:
    return StructuredDocument(
        elements=[Element(type="paragraph", text=text)],
        raw_text=text,
    )


def _vision_ok(text="extracted text"):
    """Mock DocuvisionClient.extract that returns a StructuredDocument."""
    mock = MagicMock()
    mock.extract.return_value = _doc(text)
    return mock


def _vision_fail(exc=None):
    mock = MagicMock()
    mock.extract.side_effect = exc or ConnectionError("service down")
    return mock


def _generate_fn(prompt, max_tokens=512, temperature=0.7):
    return f"generated: {prompt[:20]}"


def _stream_fn(prompt, max_tokens=512, temperature=0.7):
    yield "tok1"
    yield "tok2"
    yield "tok3"


def _pipe(vision_mock=None, generate_fn=None, stream_fn=None,
          vram_serialise=False, swap_fn=None, prompt_fn=None) -> MultimodalPipeline:
    cfg = MultimodalConfig(vram_serialise=vram_serialise)
    if prompt_fn:
        cfg.prompt_fn = prompt_fn
    pipe = MultimodalPipeline(cfg, generate_fn=generate_fn or _generate_fn,
                               stream_fn=stream_fn, swap_fn=swap_fn)
    if vision_mock is not None:
        pipe._vision = vision_mock
    return pipe


# ── DefaultPrompt ─────────────────────────────────────────────────────────────

class TestDefaultPrompt:
    def test_page_zero_no_header(self):
        doc = _doc("hello")
        assert _default_prompt(0, doc) == "hello"

    def test_page_one_has_header(self):
        doc = _doc("content")
        prompt = _default_prompt(1, doc)
        assert "[Page 2]" in prompt
        assert "content" in prompt


# ── run() ─────────────────────────────────────────────────────────────────────

class TestMultimodalPipelineRun:
    def test_single_page_success(self):
        pipe = _pipe(vision_mock=_vision_ok("resume text"))
        results = list(pipe.run([b"page0_bytes"]))
        assert len(results) == 1
        assert results[0].page_idx == 0
        assert results[0].error is None
        assert "generated" in results[0].generated

    def test_multiple_pages_all_yielded(self):
        pipe = _pipe(vision_mock=_vision_ok())
        results = list(pipe.run([b"p0", b"p1", b"p2"]))
        assert len(results) == 3
        assert [r.page_idx for r in results] == [0, 1, 2]

    def test_vision_failure_yields_error_page(self):
        pipe = _pipe(vision_mock=_vision_fail())
        results = list(pipe.run([b"p0"]))
        assert results[0].error is not None
        assert results[0].doc is None
        assert results[0].generated == ""

    def test_partial_failure_does_not_stop_pipeline(self):
        """One bad page should not prevent subsequent pages from processing."""
        mock = MagicMock()
        mock.extract.side_effect = [
            ConnectionError("fail"),
            _doc("good text"),
        ]
        pipe = _pipe(vision_mock=mock)
        results = list(pipe.run([b"p0", b"p1"]))
        assert results[0].error is not None
        assert results[1].error is None

    def test_generation_failure_yields_error_page(self):
        def _bad_gen(prompt, **kw):
            raise RuntimeError("model OOM")

        pipe = _pipe(vision_mock=_vision_ok(), generate_fn=_bad_gen)
        results = list(pipe.run([b"p0"]))
        assert results[0].error is not None
        assert "OOM" in results[0].error

    def test_doc_attached_to_result(self):
        pipe = _pipe(vision_mock=_vision_ok("some text"))
        results = list(pipe.run([b"p0"]))
        assert results[0].doc is not None
        assert results[0].doc.raw_text == "some text"

    def test_empty_pages_yields_nothing(self):
        pipe = _pipe(vision_mock=_vision_ok())
        assert list(pipe.run([])) == []

    def test_custom_prompt_fn_called(self):
        calls = []

        def _prompt_fn(page_idx, doc):
            calls.append((page_idx, doc.raw_text))
            return f"custom:{doc.raw_text}"

        pipe = _pipe(vision_mock=_vision_ok("txt"), prompt_fn=_prompt_fn)
        list(pipe.run([b"p0"]))
        assert calls == [(0, "txt")]

    def test_vram_serialise_calls_swap_fn(self):
        swaps = []
        pipe = _pipe(vision_mock=_vision_ok(), vram_serialise=True,
                     swap_fn=lambda: swaps.append(1))
        list(pipe.run([b"p0", b"p1"]))
        assert len(swaps) == 2  # once per page

    def test_vram_serialise_false_no_swap_called(self):
        swaps = []
        pipe = _pipe(vision_mock=_vision_ok(), vram_serialise=False,
                     swap_fn=lambda: swaps.append(1))
        list(pipe.run([b"p0"]))
        assert swaps == []

    def test_swap_fn_none_does_not_raise(self):
        pipe = _pipe(vision_mock=_vision_ok(), vram_serialise=True, swap_fn=None)
        results = list(pipe.run([b"p0"]))
        assert results[0].error is None


# ── stream() ──────────────────────────────────────────────────────────────────

class TestMultimodalPipelineStream:
    def test_yields_page_idx_token_tuples(self):
        pipe = _pipe(vision_mock=_vision_ok(), stream_fn=_stream_fn)
        tokens = list(pipe.stream([b"p0"]))
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tokens)
        assert tokens[0][0] == 0  # page_idx
        assert tokens[0][1] == "tok1"

    def test_multiple_pages_interleaved_by_page(self):
        pipe = _pipe(vision_mock=_vision_ok(), stream_fn=_stream_fn)
        tokens = list(pipe.stream([b"p0", b"p1"]))
        page_indices = [t[0] for t in tokens]
        # All page-0 tokens come before page-1 tokens (pages are sequential)
        assert page_indices == sorted(page_indices)

    def test_vision_failure_yields_error_token(self):
        pipe = _pipe(vision_mock=_vision_fail(), stream_fn=_stream_fn)
        tokens = list(pipe.stream([b"p0"]))
        assert len(tokens) == 1
        assert "extraction error" in tokens[0][1]

    def test_stream_fn_error_yields_error_token(self):
        def _bad_stream(prompt, **kw):
            raise RuntimeError("GPU gone")
            yield  # make it a generator

        pipe = _pipe(vision_mock=_vision_ok(), stream_fn=_bad_stream)
        tokens = list(pipe.stream([b"p0"]))
        assert any("generation error" in t[1] for t in tokens)

    def test_empty_pages_yields_nothing(self):
        pipe = _pipe(vision_mock=_vision_ok(), stream_fn=_stream_fn)
        assert list(pipe.stream([])) == []


# ── Import check ──────────────────────────────────────────────────────────────

def test_exported_from_pipeline_package():
    from circuitforge_core.pipeline import MultimodalPipeline, MultimodalConfig, PageResult
    assert MultimodalPipeline is not None
