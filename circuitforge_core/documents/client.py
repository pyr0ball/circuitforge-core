"""
circuitforge_core.documents.client — HTTP client for the cf-docuvision service.

Thin wrapper around the cf-docuvision FastAPI service's POST /extract endpoint.
Used by ingest() as the primary path; callers should not use this directly.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import requests

from .models import Element, ParsedTable, StructuredDocument

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 60


class DocuvisionClient:
    """Synchronous HTTP client for cf-docuvision.

    Args:
        base_url: Root URL of the cf-docuvision service, e.g. 'http://localhost:8003'
        timeout:  Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8003", timeout: int = _DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_healthy(self) -> bool:
        """Return True if the service responds to GET /health."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def extract(self, image_bytes: bytes, hint: str = "auto") -> StructuredDocument:
        """
        Submit image bytes to cf-docuvision and return a StructuredDocument.

        Raises:
            requests.HTTPError: if the service returns a non-2xx status.
            requests.ConnectionError / requests.Timeout: if the service is unreachable.
        """
        payload = {
            "image_b64": base64.b64encode(image_bytes).decode(),
            "hint": hint,
        }
        resp = requests.post(
            f"{self.base_url}/extract",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _parse_response(resp.json())


def _parse_response(data: dict[str, Any]) -> StructuredDocument:
    elements = [
        Element(
            type=e["type"],
            text=e["text"],
            bbox=tuple(e["bbox"]) if e.get("bbox") else None,
        )
        for e in data.get("elements", [])
    ]
    tables = [
        ParsedTable(
            html=t["html"],
            bbox=tuple(t["bbox"]) if t.get("bbox") else None,
        )
        for t in data.get("tables", [])
    ]
    return StructuredDocument(
        elements=elements,
        raw_text=data.get("raw_text", ""),
        tables=tables,
        metadata=data.get("metadata", {}),
    )
