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
    page = _mock_page(
        "Fireball deals 8d6 fire damage on a failed Dexterity saving throw."
    )
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


def test_pdfplumber_not_installed():
    """pdfplumber=None guard raises ImportError with install hint."""
    import circuitforge_core.documents.pdf as pdf_mod

    with patch.object(pdf_mod, "pdfplumber", None):
        with pytest.raises(ImportError, match="pdfplumber"):
            PDFExtractor().chunk_pages("/fake/book.pdf")


def test_chunk_pages_triggers_ocr_for_sparse_page():
    """Page with fewer words than ocr_min_words falls back to OCR."""
    sparse_page = _mock_page("few words only")  # 3 words < default 10
    mock_image = MagicMock()
    rendered = MagicMock()
    rendered.original = mock_image

    sparse_page.to_image.return_value = rendered

    with (
        patch("circuitforge_core.documents.pdf.pdfplumber") as mock_pl,
        patch("circuitforge_core.documents.pdf.pytesseract") as mock_tess,
        patch("circuitforge_core.documents.pdf.Image") as mock_pil,
    ):
        mock_pl.open.return_value = _mock_pdf([sparse_page])
        mock_pil.open.return_value = mock_image
        mock_tess.image_to_string.return_value = (
            "Full OCR extracted rulebook text about saving throws."
        )

        chunks = PDFExtractor(ocr_min_words=10).chunk_pages("/fake/scan.pdf")

    assert chunks[0].source == "ocr"
    assert "OCR extracted" in chunks[0].text


def test_chunk_pages_ocr_failure_returns_empty_chunk():
    """OCR render failure results in empty chunk, not an exception."""
    sparse_page = _mock_page("")
    sparse_page.to_image.side_effect = RuntimeError("render failed")

    with patch("circuitforge_core.documents.pdf.pdfplumber") as mock_pl:
        mock_pl.open.return_value = _mock_pdf([sparse_page])
        chunks = PDFExtractor().chunk_pages("/fake/broken.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].source == "ocr"
    assert chunks[0].word_count == 0
