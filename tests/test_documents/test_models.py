# tests/test_documents/test_models.py
"""Unit tests for circuitforge_core.documents.models."""
from __future__ import annotations

import pytest
from circuitforge_core.documents.models import Element, ParsedTable, StructuredDocument


def test_element_is_frozen():
    e = Element(type="heading", text="Title")
    with pytest.raises(Exception):
        e.text = "changed"  # type: ignore[misc]


def test_element_bbox_optional():
    e = Element(type="paragraph", text="hello")
    assert e.bbox is None


def test_parsed_table_frozen():
    t = ParsedTable(html="<table/>")
    with pytest.raises(Exception):
        t.html = "changed"  # type: ignore[misc]


def test_structured_document_defaults():
    doc = StructuredDocument()
    assert doc.elements == []
    assert doc.raw_text == ""
    assert doc.tables == []
    assert doc.metadata == {}


def test_structured_document_headings_filter():
    doc = StructuredDocument(elements=[
        Element(type="heading", text="H1"),
        Element(type="paragraph", text="body"),
        Element(type="heading", text="H2"),
    ])
    assert [e.text for e in doc.headings] == ["H1", "H2"]


def test_structured_document_paragraphs_filter():
    doc = StructuredDocument(elements=[
        Element(type="heading", text="H1"),
        Element(type="paragraph", text="body"),
    ])
    assert len(doc.paragraphs) == 1
    assert doc.paragraphs[0].text == "body"
