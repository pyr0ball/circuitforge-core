"""
Tests for the cf-vision FastAPI service (mock backend).

All tests use the mock backend — no GPU or model files required.
"""
from __future__ import annotations

import json
import io

import pytest
from fastapi.testclient import TestClient

from circuitforge_core.vision.app import create_app, _parse_labels


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def siglip_client() -> TestClient:
    """Client backed by mock-siglip (supports classify + embed, not caption)."""
    app = create_app(model_path="mock-siglip", backend="siglip", mock=True)
    return TestClient(app)


@pytest.fixture(scope="module")
def vlm_client() -> TestClient:
    """Client backed by mock-vlm (mock supports all; VLM contract tested separately)."""
    app = create_app(model_path="mock-vlm", backend="vlm", mock=True)
    return TestClient(app)


FAKE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def _image_upload(data: bytes = FAKE_IMAGE) -> tuple[str, tuple]:
    return ("image", ("test.png", io.BytesIO(data), "image/png"))


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(siglip_client: TestClient) -> None:
    resp = siglip_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "vram_mb" in body
    assert "backend" in body


def test_health_backend_field(siglip_client: TestClient) -> None:
    resp = siglip_client.get("/health")
    assert resp.json()["backend"] == "siglip"


def test_health_supports_fields(siglip_client: TestClient) -> None:
    body = siglip_client.get("/health").json()
    assert "supports_embed" in body
    assert "supports_caption" in body


# ── /classify ─────────────────────────────────────────────────────────────────

def test_classify_json_labels(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/classify",
        files=[_image_upload()],
        data={"labels": json.dumps(["cat", "dog", "bird"])},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == ["cat", "dog", "bird"]
    assert len(body["scores"]) == 3


def test_classify_csv_labels(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/classify",
        files=[_image_upload()],
        data={"labels": "cat, dog, bird"},
    )
    assert resp.status_code == 200
    assert resp.json()["labels"] == ["cat", "dog", "bird"]


def test_classify_single_label(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/classify",
        files=[_image_upload()],
        data={"labels": "document"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == ["document"]
    assert len(body["scores"]) == 1


def test_classify_empty_labels_4xx(siglip_client: TestClient) -> None:
    # Empty labels should yield a 4xx — either our 400 or FastAPI's 422
    # depending on how the empty string is handled by the form layer.
    resp = siglip_client.post(
        "/classify",
        files=[_image_upload()],
        data={"labels": ""},
    )
    assert resp.status_code in (400, 422)


def test_classify_empty_image_400(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/classify",
        files=[("image", ("empty.png", io.BytesIO(b""), "image/png"))],
        data={"labels": "cat"},
    )
    assert resp.status_code == 400


def test_classify_model_in_response(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/classify",
        files=[_image_upload()],
        data={"labels": "cat"},
    )
    assert "model" in resp.json()


# ── /embed ────────────────────────────────────────────────────────────────────

def test_embed_returns_vector(siglip_client: TestClient) -> None:
    resp = siglip_client.post("/embed", files=[_image_upload()])
    assert resp.status_code == 200
    body = resp.json()
    assert "embedding" in body
    assert isinstance(body["embedding"], list)
    assert len(body["embedding"]) > 0


def test_embed_empty_image_400(siglip_client: TestClient) -> None:
    resp = siglip_client.post(
        "/embed",
        files=[("image", ("empty.png", io.BytesIO(b""), "image/png"))],
    )
    assert resp.status_code == 400


def test_embed_model_in_response(siglip_client: TestClient) -> None:
    resp = siglip_client.post("/embed", files=[_image_upload()])
    assert "model" in resp.json()


# ── /caption ──────────────────────────────────────────────────────────────────

def test_caption_returns_text(vlm_client: TestClient) -> None:
    resp = vlm_client.post(
        "/caption",
        files=[_image_upload()],
        data={"prompt": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "caption" in body
    assert isinstance(body["caption"], str)


def test_caption_with_prompt(vlm_client: TestClient) -> None:
    resp = vlm_client.post(
        "/caption",
        files=[_image_upload()],
        data={"prompt": "What text appears here?"},
    )
    assert resp.status_code == 200


def test_caption_empty_image_400(vlm_client: TestClient) -> None:
    resp = vlm_client.post(
        "/caption",
        files=[("image", ("empty.png", io.BytesIO(b""), "image/png"))],
        data={"prompt": ""},
    )
    assert resp.status_code == 400


# ── Label parser ──────────────────────────────────────────────────────────────

def test_parse_labels_json_array() -> None:
    assert _parse_labels('["cat", "dog"]') == ["cat", "dog"]


def test_parse_labels_csv() -> None:
    assert _parse_labels("cat, dog, bird") == ["cat", "dog", "bird"]


def test_parse_labels_single() -> None:
    assert _parse_labels("document") == ["document"]


def test_parse_labels_empty() -> None:
    assert _parse_labels("") == []


def test_parse_labels_whitespace_trimmed() -> None:
    assert _parse_labels("  cat ,  dog  ") == ["cat", "dog"]
