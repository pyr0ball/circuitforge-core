"""
circuitforge_core.documents — shared document ingestion pipeline.

Primary entry point::

    from circuitforge_core.documents import ingest, StructuredDocument

    doc: StructuredDocument = ingest(image_bytes, hint="auto")
"""
from .ingest import ingest
from .models import Element, ParsedTable, StructuredDocument

__all__ = [
    "ingest",
    "Element",
    "ParsedTable",
    "StructuredDocument",
]
