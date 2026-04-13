# tests/test_text/test_oai_compat.py
"""Tests for the OpenAI-compatible /v1/chat/completions endpoint on cf-text."""
import pytest
from fastapi.testclient import TestClient

from circuitforge_core.text.app import create_app


@pytest.fixture()
def client():
    app = create_app(model_path="mock", mock=True)
    return TestClient(app)


def test_oai_chat_completions_returns_200(client: TestClient) -> None:
    """POST /v1/chat/completions returns 200 with a valid request."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "cf-text",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200


def test_oai_chat_completions_response_shape(client: TestClient) -> None:
    """Response contains the fields LLMRouter expects: choices[0].message.content."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "cf-text",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Write a short greeting."},
            ],
            "max_tokens": 64,
        },
    )
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert len(choice["message"]["content"]) > 0


def test_oai_chat_completions_includes_metadata(client: TestClient) -> None:
    """Response includes id, object, created, model, and usage fields."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "cf-text", "messages": [{"role": "user", "content": "Hi"}]},
    )
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert isinstance(data["id"], str)
    assert data["id"].startswith("cftext-")
    assert isinstance(data["created"], int)
    assert "usage" in data


def test_health_endpoint_still_works(client: TestClient) -> None:
    """Existing /health endpoint is unaffected by the new OAI route."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
