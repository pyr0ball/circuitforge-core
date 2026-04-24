# documents

Document ingestion pipeline. Converts PDF, DOCX, ODT, and images into a normalized `StructuredDocument` for downstream processing.

```python
from circuitforge_core.documents import ingest, StructuredDocument
```

## Supported formats

| Format | Method | Notes |
|--------|--------|-------|
| PDF | `pdfplumber` | Two-column detection via gutter analysis |
| DOCX | `python-docx` | Paragraph and table extraction |
| ODT | stdlib `zipfile` + `ElementTree` | No external deps required |
| PNG/JPG | cf-docuvision fast-path, local fallback | OCR via vision router |

## `ingest(path: str | Path) -> StructuredDocument`

Main entry point. Detects format by file extension and routes to the appropriate parser.

```python
doc = ingest("/tmp/invoice.pdf")
print(doc.text)       # full extracted text
print(doc.pages)      # list of per-page content
print(doc.metadata)   # title, author, creation date if available
```

## StructuredDocument

```python
@dataclass
class StructuredDocument:
    text: str                        # full plain text
    pages: list[str]                 # per-page text (PDFs)
    sections: dict[str, str]         # named sections if detected
    metadata: dict[str, Any]         # format-specific metadata
    source_path: str
    format: str                      # "pdf" | "docx" | "odt" | "image"
```

## PDF specifics

Two-column PDFs (common in resumes and academic papers) are handled by `_find_column_split()`, which detects the gutter via word x-positions and extracts left and right columns separately before merging.

CID glyph references (`(cid:NNN)`) from ATS-reembedded fonts are stripped automatically. Common bullet CIDs (127, 149, 183) are mapped to `•`.

## OCR path

Image inputs go through the vision router (see the [vision module](vision.md)). In practice this means:

1. cf-docuvision fast-path (if available on the cf-orch coordinator)
2. Local moondream2 fallback

OCR results are treated as unstructured text — no section detection is attempted.

## ATS gotcha

Some ATS-exported PDFs embed fonts in ways that cause `pdfplumber` to extract garbled text. If `doc.text` looks corrupted (common with Oracle Taleo exports), try the image fallback:

```python
doc = ingest(path, force_ocr=True)
```
