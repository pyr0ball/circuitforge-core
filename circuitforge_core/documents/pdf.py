# circuitforge_core/documents/pdf.py
"""
circuitforge_core.documents.pdf — PDF text extraction and page-level chunking.

Primary path: pdfplumber (selectable text layers).
Fallback: pytesseract OCR (scanned / image-only pages).

Usage::

    from circuitforge_core.documents.pdf import PDFExtractor

    chunks = PDFExtractor().chunk_pages("/path/to/book.pdf")
    for chunk in chunks:
        print(f"[p.{chunk.page_number}] ({chunk.source}) {chunk.text[:80]}")
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PageChunk:
    """Text content extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    source: str  # "text_layer" | "ocr"
    word_count: int


class PDFExtractor:
    """
    Extract page-level text chunks from PDF files.

    Args:
        ocr_min_words: Pages with fewer words from the text layer trigger OCR.
    """

    def __init__(self, ocr_min_words: int = 10) -> None:
        self.ocr_min_words = ocr_min_words

    def chunk_pages(self, pdf_path: str | Path) -> list[PageChunk]:
        """
        Primary entry point. Returns one PageChunk per page.

        Uses text-layer extraction per page; falls back to OCR when text is sparse.
        Empty PDFs return an empty list.
        """
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            )

        path = Path(pdf_path)
        chunks: list[PageChunk] = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                words = text.split()

                if len(words) >= self.ocr_min_words:
                    chunks.append(
                        PageChunk(
                            page_number=i,
                            text=text.strip(),
                            source="text_layer",
                            word_count=len(words),
                        )
                    )
                else:
                    logger.debug(
                        "pdf: page %d sparse (%d words), falling back to OCR",
                        i,
                        len(words),
                    )
                    chunks.append(self._ocr_page(page, i))

        return chunks

    def _ocr_page(self, page: object, page_number: int) -> PageChunk:
        """Render page to image and extract text via tesseract."""
        try:
            rendered = page.to_image(resolution=200).original  # type: ignore[attr-defined]
            rendered = _ensure_pil_image(rendered)
            text = pytesseract.image_to_string(rendered)  # type: ignore[union-attr]
            words = text.split()
            return PageChunk(
                page_number=page_number,
                text=text.strip(),
                source="ocr",
                word_count=len(words),
            )
        except Exception as exc:
            logger.warning("pdf: OCR failed for page %d: %s", page_number, exc)
            return PageChunk(
                page_number=page_number, text="", source="ocr", word_count=0
            )


def _ensure_pil_image(rendered: object) -> object:
    """Return *rendered* as a PIL Image, converting from bytes if needed."""
    if Image is None:
        return rendered
    try:
        if not isinstance(rendered, Image.Image):
            rendered = Image.open(io.BytesIO(rendered))  # type: ignore[arg-type]
    except TypeError:
        # Image may be patched (e.g. in tests); skip the conversion.
        pass
    return rendered
