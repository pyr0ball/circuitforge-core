# tests/test_documents/test_pdf.py
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from circuitforge_core.documents.pdf import PDFExtractor, PageChunk


def _mock_page(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _mock_pdf(pages: list[MagicMock]) -> MagicMock:
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    pdf.pages = pages
    return pdf


def test_chunk_pages_single_text_layer_page():
    page = _mock_page("Fireball deals 8d6 fire damage on a failed Dexterity saving throw.")
    with patch("circuitforge_core.documents.pdf.pdfplumber") as mock_pl:
        mock_pl.open.return_value = _mock_pdf([page])
        chunks = PDFExtractor().chunk_pages("/fake/book.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].source == "text_layer"
    assert "Fireball" in chunks[0].text
    assert chunks[0].word_count >= 10


def test_chunk_pages_numbers_from_one():
    pages = [_mock_page(f"Rule text for page {i} " * 10) for i in range(1, 4)]
    with patch("circuitforge_core.documents.pdf.pdfplumber") as mock_pl:
        mock_pl.open.return_value = _mock_pdf(pages)
        chunks = PDFExtractor().chunk_pages("/fake/book.pdf")
    assert [c.page_number for c in chunks] == [1, 2, 3]


def test_page_chunk_is_frozen():
    chunk = PageChunk(page_number=1, text="hello", source="text_layer", word_count=1)
    with pytest.raises(Exception):
        chunk.text = "modified"  # type: ignore[misc]
