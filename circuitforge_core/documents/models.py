"""
circuitforge_core.documents.models — shared document data types.

These are the canonical output types from the document ingestion pipeline.
All consumers (kiwi, falcon, peregrine, godwit, …) receive a StructuredDocument
regardless of whether Dolphin-v2 or LLM fallback was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Element:
    """A single logical content unit within a document.

    type: one of heading | paragraph | list | table | figure | formula | code
    text: extracted plain text (for tables: may be row summary or empty)
    bbox: normalised [x0, y0, x1, y1] in 0-1 space, None when unavailable
    """
    type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ParsedTable:
    """An extracted table rendered as HTML."""
    html: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class StructuredDocument:
    """
    The canonical result of document ingestion.

    Produced by ingest() for any input image regardless of which backend
    (cf-docuvision or LLM fallback) processed it.
    """
    elements: list[Element] = field(default_factory=list)
    raw_text: str = ""
    tables: list[ParsedTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def headings(self) -> list[Element]:
        return [e for e in self.elements if e.type == "heading"]

    @property
    def paragraphs(self) -> list[Element]:
        return [e for e in self.elements if e.type == "paragraph"]
