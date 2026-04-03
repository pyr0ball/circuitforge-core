# tests/test_documents/test_client.py
"""Unit tests for DocuvisionClient."""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.documents.client import DocuvisionClient


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


_EXTRACT_RESPONSE = {
    "elements": [
        {"type": "heading", "text": "Invoice", "bbox": [0.0, 0.0, 1.0, 0.1]},
        {"type": "paragraph", "text": "Due: $100"},
    ],
    "raw_text": "Invoice\nDue: $100",
    "tables": [
        {"html": "<table><tr><td>$100</td></tr></table>", "bbox": None},
    ],
    "metadata": {"hint": "auto", "width": 800, "height": 1200, "model": "/fake/model", "device": "cpu"},
}


def test_is_healthy_true():
    with patch("requests.get", return_value=_mock_response({}, 200)):
        client = DocuvisionClient("http://localhost:8003")
        assert client.is_healthy() is True


def test_is_healthy_false_on_error():
    with patch("requests.get", side_effect=ConnectionError("refused")):
        client = DocuvisionClient("http://localhost:8003")
        assert client.is_healthy() is False


def test_is_healthy_false_on_500():
    with patch("requests.get", return_value=_mock_response({}, 500)):
        client = DocuvisionClient("http://localhost:8003")
        assert client.is_healthy() is False


def test_extract_returns_structured_document():
    with patch("requests.post", return_value=_mock_response(_EXTRACT_RESPONSE)):
        client = DocuvisionClient()
        doc = client.extract(b"fake-image-bytes", hint="auto")

    assert doc.raw_text == "Invoice\nDue: $100"
    assert len(doc.elements) == 2
    assert doc.elements[0].type == "heading"
    assert doc.elements[0].bbox == (0.0, 0.0, 1.0, 0.1)
    assert len(doc.tables) == 1
    assert "$100" in doc.tables[0].html


def test_extract_sends_base64_image():
    with patch("requests.post", return_value=_mock_response(_EXTRACT_RESPONSE)) as mock_post:
        client = DocuvisionClient()
        client.extract(b"pixels", hint="table")

    call_json = mock_post.call_args.kwargs["json"]
    assert call_json["hint"] == "table"
    assert base64.b64decode(call_json["image_b64"]) == b"pixels"


def test_extract_raises_on_http_error():
    import requests as req_lib
    with patch("requests.post", return_value=_mock_response({}, 422)):
        client = DocuvisionClient()
        with pytest.raises(req_lib.HTTPError):
            client.extract(b"bad")


def test_extract_table_bbox_none_when_missing():
    response = dict(_EXTRACT_RESPONSE)
    response["tables"] = [{"html": "<table/>"}]
    with patch("requests.post", return_value=_mock_response(response)):
        client = DocuvisionClient()
        doc = client.extract(b"img")
    assert doc.tables[0].bbox is None
