"""
circuitforge_core.documents.ingest — public document ingestion entry point.

Primary path: cf-docuvision HTTP service (Dolphin-v2, layout-aware).
Fallback path: LLMRouter vision call (lower fidelity, no layout/bbox).

Usage::

    from circuitforge_core.documents import ingest

    with open("receipt.jpg", "rb") as f:
        doc = ingest(f.read(), hint="table")

    print(doc.raw_text)
    for table in doc.tables:
        print(table.html)
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

from .client import DocuvisionClient
from .models import Element, StructuredDocument

logger = logging.getLogger(__name__)

_DOCUVISION_URL_ENV = "CF_DOCUVISION_URL"
_DOCUVISION_URL_DEFAULT = "http://localhost:8003"

_LLM_FALLBACK_PROMPTS: dict[str, str] = {
    "auto":  "Extract all text from this document. Return a JSON array of {\"type\": ..., \"text\": ...} objects.",
    "table": "Extract all tables from this document as HTML. Return a JSON array including {\"type\": \"table\", \"html\": ..., \"text\": ...} objects.",
    "text":  "Extract all text from this document preserving headings and paragraphs. Return a JSON array of {\"type\": ..., \"text\": ...} objects.",
    "form":  "Extract all form field labels and values from this document. Return a JSON array of {\"type\": ..., \"text\": ...} objects.",
}


def ingest(
    image_bytes: bytes,
    hint: str = "auto",
    *,
    docuvision_url: str | None = None,
    llm_router: Any | None = None,
    llm_config_path: Any | None = None,
) -> StructuredDocument:
    """
    Ingest an image and return a StructuredDocument.

    Tries cf-docuvision first; falls back to LLMRouter vision if the service is
    unavailable or fails. If neither is available, returns an empty document.

    Args:
        image_bytes:      Raw bytes of the image (JPEG, PNG, etc.).
        hint:             Extraction mode: "auto" | "table" | "text" | "form".
        docuvision_url:   Override service URL (defaults to CF_DOCUVISION_URL env or localhost:8003).
        llm_router:       Pre-built LLMRouter instance for fallback (optional).
        llm_config_path:  Path to llm.yaml for lazy-constructing LLMRouter if needed.

    Returns:
        StructuredDocument — always, even on total failure (empty document).
    """
    url = docuvision_url or os.environ.get(_DOCUVISION_URL_ENV, _DOCUVISION_URL_DEFAULT)
    client = DocuvisionClient(base_url=url)

    # ── primary: cf-docuvision ────────────────────────────────────────────────
    try:
        if client.is_healthy():
            doc = client.extract(image_bytes, hint=hint)
            logger.debug("ingest: cf-docuvision succeeded (%d elements)", len(doc.elements))
            return doc
        logger.debug("ingest: cf-docuvision unhealthy, falling back to LLM")
    except Exception as exc:
        logger.warning("ingest: cf-docuvision failed (%s), falling back to LLM", exc)

    # ── fallback: LLMRouter vision ────────────────────────────────────────────
    router = llm_router or _build_llm_router(llm_config_path)
    if router is None:
        logger.warning("ingest: no LLM router available; returning empty document")
        return StructuredDocument(metadata={"source": "none", "hint": hint})

    try:
        return _llm_ingest(router, image_bytes, hint)
    except Exception as exc:
        logger.warning("ingest: LLM fallback failed (%s); returning empty document", exc)
        return StructuredDocument(metadata={"source": "llm_error", "hint": hint, "error": str(exc)})


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_llm_router(config_path: Any | None) -> Any | None:
    """Lazily construct an LLMRouter; return None if unavailable."""
    try:
        from circuitforge_core.llm import LLMRouter
        kwargs: dict[str, Any] = {}
        if config_path is not None:
            kwargs["config_path"] = config_path
        return LLMRouter(**kwargs)
    except Exception as exc:
        logger.debug("ingest: could not build LLMRouter: %s", exc)
        return None


def _llm_ingest(router: Any, image_bytes: bytes, hint: str) -> StructuredDocument:
    """Use LLMRouter's vision capability to extract document text."""
    import json

    prompt = _LLM_FALLBACK_PROMPTS.get(hint, _LLM_FALLBACK_PROMPTS["auto"])
    b64 = base64.b64encode(image_bytes).decode()

    raw = router.generate_vision(
        prompt=prompt,
        image_b64=b64,
    )

    # Try to parse structured output; fall back to single paragraph
    elements: list[Element] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                elements.append(Element(
                    type=item.get("type", "paragraph"),
                    text=item.get("text", ""),
                ))
    except (json.JSONDecodeError, TypeError):
        elements = [Element(type="paragraph", text=raw.strip())]

    raw_text = "\n".join(e.text for e in elements)
    return StructuredDocument(
        elements=elements,
        raw_text=raw_text,
        tables=[],
        metadata={"source": "llm_fallback", "hint": hint},
    )
