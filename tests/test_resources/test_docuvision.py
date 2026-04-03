# tests/test_resources/test_docuvision.py
"""
Unit tests for cf-docuvision FastAPI service (circuitforge_core/resources/docuvision/app.py).

Covers:
  - GET /health          → status + model path
  - POST /extract        → image_b64, image_path, hint routing, metadata fields
  - _parse_dolphin_output → JSON list path, table detection, plain-text fallback
  - _image_from_request  → missing both fields → 422; bad image_path → 404
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import circuitforge_core.resources.docuvision.app as docuvision_module
from circuitforge_core.resources.docuvision.app import (
    _parse_dolphin_output,
    app,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_jpeg_b64(width: int = 10, height: int = 10) -> str:
    """Return a base64-encoded 10x10 white JPEG."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level model state between tests."""
    docuvision_module._model = None
    docuvision_module._processor = None
    docuvision_module._model_path = "/fake/model"
    docuvision_module._device = "cpu"
    yield
    docuvision_module._model = None
    docuvision_module._processor = None


@pytest.fixture
def mock_model():
    """
    Inject fake model + processor into the module so _load_model() is skipped.

    The processor returns a dict-like with 'input_ids'; the model generate()
    returns a tensor-like whose decode produces a JSON string.
    """
    fake_ids = MagicMock()
    fake_ids.shape = [1, 5]      # input_len = 5

    fake_inputs = {"input_ids": fake_ids}
    fake_inputs_obj = MagicMock()
    fake_inputs_obj.__getitem__ = lambda self, k: fake_inputs[k]
    fake_inputs_obj.to = lambda device: fake_inputs_obj

    fake_output = MagicMock()
    fake_output.__getitem__ = lambda self, idx: MagicMock()  # output_ids[0]

    fake_model = MagicMock()
    fake_model.generate.return_value = fake_output

    fake_processor = MagicMock()
    fake_processor.return_value = fake_inputs_obj
    fake_processor.decode.return_value = json.dumps([
        {"type": "heading", "text": "Invoice", "bbox": [0.0, 0.0, 1.0, 0.1]},
        {"type": "table", "text": "row1", "html": "<table><tr><td>row1</td></tr></table>",
         "bbox": [0.0, 0.1, 1.0, 0.5]},
    ])

    docuvision_module._model = fake_model
    docuvision_module._processor = fake_processor
    return fake_model, fake_processor


@pytest.fixture
def client():
    return TestClient(app)


# ── health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "/fake/model"


# ── _parse_dolphin_output ────────────────────────────────────────────────────

def test_parse_json_list_elements():
    raw = json.dumps([
        {"type": "heading", "text": "Title"},
        {"type": "paragraph", "text": "Body text"},
    ])
    elements, tables, raw_text = _parse_dolphin_output(raw)
    assert len(elements) == 2
    assert elements[0].type == "heading"
    assert elements[0].text == "Title"
    assert elements[1].type == "paragraph"
    assert raw_text == "Title\nBody text"
    assert tables == []


def test_parse_json_table_extracted():
    raw = json.dumps([
        {"type": "table", "text": "row", "html": "<table><tr><td>A</td></tr></table>",
         "bbox": [0.0, 0.0, 1.0, 0.5]},
    ])
    elements, tables, raw_text = _parse_dolphin_output(raw)
    assert len(tables) == 1
    assert tables[0].html == "<table><tr><td>A</td></tr></table>"
    assert tables[0].bbox == [0.0, 0.0, 1.0, 0.5]
    assert len(elements) == 1
    assert elements[0].type == "table"


def test_parse_plain_text_fallback():
    raw = "This is not JSON at all."
    elements, tables, raw_text = _parse_dolphin_output(raw)
    assert len(elements) == 1
    assert elements[0].type == "paragraph"
    assert elements[0].text == raw
    assert tables == []
    assert raw_text == raw


def test_parse_empty_string_fallback():
    elements, tables, raw_text = _parse_dolphin_output("")
    assert len(elements) == 1
    assert elements[0].type == "paragraph"
    assert elements[0].text == ""


def test_parse_json_missing_type_defaults_to_paragraph():
    raw = json.dumps([{"text": "no type field"}])
    elements, tables, _ = _parse_dolphin_output(raw)
    assert elements[0].type == "paragraph"


# ── POST /extract ─────────────────────────────────────────────────────────────

def test_extract_image_b64(client, mock_model):
    resp = client.post("/extract", json={"image_b64": _make_jpeg_b64(), "hint": "auto"})
    assert resp.status_code == 200
    data = resp.json()
    assert "elements" in data
    assert "raw_text" in data
    assert "tables" in data
    assert data["metadata"]["hint"] == "auto"
    assert data["metadata"]["model"] == "/fake/model"
    assert data["metadata"]["width"] == 10
    assert data["metadata"]["height"] == 10


def test_extract_hint_table_routes_correct_prompt(client, mock_model):
    _, fake_processor = mock_model
    resp = client.post("/extract", json={"image_b64": _make_jpeg_b64(), "hint": "table"})
    assert resp.status_code == 200
    # Verify processor was called with the table-specific prompt
    call_kwargs = fake_processor.call_args
    assert "table" in call_kwargs.kwargs.get("text", "") or \
           "table" in str(call_kwargs)


def test_extract_hint_unknown_falls_back_to_auto(client, mock_model):
    """An unrecognised hint silently falls back to the 'auto' prompt."""
    resp = client.post("/extract", json={"image_b64": _make_jpeg_b64(), "hint": "nonsense"})
    assert resp.status_code == 200


def test_extract_image_path(tmp_path, client, mock_model):
    img_file = tmp_path / "doc.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_file)
    resp = client.post("/extract", json={"image_path": str(img_file)})
    assert resp.status_code == 200
    assert resp.json()["metadata"]["width"] == 8


def test_extract_image_path_not_found(client, mock_model):
    resp = client.post("/extract", json={"image_path": "/nonexistent/path/img.png"})
    assert resp.status_code == 404


def test_extract_no_image_raises_422(client, mock_model):
    resp = client.post("/extract", json={"hint": "auto"})
    assert resp.status_code == 422


def test_extract_response_includes_tables(client, mock_model):
    """Verify table objects surface in response when model returns table elements."""
    resp = client.post("/extract", json={"image_b64": _make_jpeg_b64()})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tables"]) == 1
    assert "<table>" in data["tables"][0]["html"]


def test_extract_device_in_metadata(client, mock_model):
    resp = client.post("/extract", json={"image_b64": _make_jpeg_b64()})
    assert resp.status_code == 200
    assert "device" in resp.json()["metadata"]
