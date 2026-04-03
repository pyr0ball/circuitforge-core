# tests/test_documents/test_ingest.py
"""Unit tests for circuitforge_core.documents.ingest."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.documents import ingest
from circuitforge_core.documents.models import Element, StructuredDocument


def _healthy_client(doc: StructuredDocument) -> MagicMock:
    c = MagicMock()
    c.is_healthy.return_value = True
    c.extract.return_value = doc
    return c


def _unhealthy_client() -> MagicMock:
    c = MagicMock()
    c.is_healthy.return_value = False
    return c


def _doc_with_text(text: str) -> StructuredDocument:
    return StructuredDocument(
        elements=[Element(type="paragraph", text=text)],
        raw_text=text,
        metadata={"source": "docuvision"},
    )


# ── primary path ──────────────────────────────────────────────────────────────

def test_ingest_uses_docuvision_when_healthy():
    expected = _doc_with_text("hello world")
    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=_healthy_client(expected)):
        result = ingest(b"imgbytes", hint="auto")

    assert result.raw_text == "hello world"
    assert result.elements[0].type == "paragraph"


def test_ingest_passes_hint_to_client():
    expected = _doc_with_text("table data")
    with patch("circuitforge_core.documents.ingest.DocuvisionClient") as MockClient:
        mock_instance = _healthy_client(expected)
        MockClient.return_value = mock_instance
        ingest(b"imgbytes", hint="table")

    mock_instance.extract.assert_called_once_with(b"imgbytes", hint="table")


def test_ingest_uses_custom_url():
    expected = _doc_with_text("text")
    with patch("circuitforge_core.documents.ingest.DocuvisionClient") as MockClient:
        mock_instance = _healthy_client(expected)
        MockClient.return_value = mock_instance
        ingest(b"imgbytes", docuvision_url="http://myhost:9000")

    MockClient.assert_called_once_with(base_url="http://myhost:9000")


# ── fallback path ─────────────────────────────────────────────────────────────

def test_ingest_falls_back_to_llm_when_docuvision_unhealthy():
    mock_router = MagicMock()
    mock_router.generate_vision.return_value = json.dumps([
        {"type": "paragraph", "text": "fallback text"},
    ])

    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=_unhealthy_client()):
        result = ingest(b"imgbytes", llm_router=mock_router)

    assert result.raw_text == "fallback text"
    assert result.metadata["source"] == "llm_fallback"


def test_ingest_llm_fallback_plain_text():
    """LLM returns plain text (not JSON) → single paragraph element."""
    mock_router = MagicMock()
    mock_router.generate_vision.return_value = "This is plain text output."

    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=_unhealthy_client()):
        result = ingest(b"imgbytes", llm_router=mock_router)

    assert len(result.elements) == 1
    assert result.elements[0].type == "paragraph"
    assert "plain text" in result.elements[0].text


def test_ingest_returns_empty_doc_when_no_llm():
    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=_unhealthy_client()), \
         patch("circuitforge_core.documents.ingest._build_llm_router", return_value=None):
        result = ingest(b"imgbytes")

    assert isinstance(result, StructuredDocument)
    assert result.elements == []
    assert result.metadata["source"] == "none"


def test_ingest_returns_empty_doc_on_docuvision_exception():
    failing_client = MagicMock()
    failing_client.is_healthy.return_value = True
    failing_client.extract.side_effect = ConnectionError("refused")

    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=failing_client), \
         patch("circuitforge_core.documents.ingest._build_llm_router", return_value=None):
        result = ingest(b"imgbytes")

    assert isinstance(result, StructuredDocument)
    assert result.metadata["source"] == "none"


def test_ingest_returns_empty_doc_on_llm_exception():
    mock_router = MagicMock()
    mock_router.generate_vision.side_effect = RuntimeError("GPU OOM")

    with patch("circuitforge_core.documents.ingest.DocuvisionClient", return_value=_unhealthy_client()):
        result = ingest(b"imgbytes", llm_router=mock_router)

    assert isinstance(result, StructuredDocument)
    assert result.metadata["source"] == "llm_error"
    assert "GPU OOM" in result.metadata["error"]


# ── CF_DOCUVISION_URL env var ─────────────────────────────────────────────────

def test_ingest_reads_url_from_env(monkeypatch):
    monkeypatch.setenv("CF_DOCUVISION_URL", "http://envhost:7777")
    expected = _doc_with_text("env-routed")

    with patch("circuitforge_core.documents.ingest.DocuvisionClient") as MockClient:
        mock_instance = _healthy_client(expected)
        MockClient.return_value = mock_instance
        ingest(b"imgbytes")

    MockClient.assert_called_once_with(base_url="http://envhost:7777")
