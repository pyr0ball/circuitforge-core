"""
Tests for the cf-musicgen FastAPI app using mock backend.
"""
import io
import os
import wave

import pytest
from fastapi.testclient import TestClient

import circuitforge_core.musicgen.app as musicgen_app
from circuitforge_core.musicgen.backends.mock import MockMusicGenBackend


@pytest.fixture(autouse=True)
def inject_mock_backend():
    """Inject mock backend before each test; restore None after."""
    original = musicgen_app._backend
    musicgen_app._backend = MockMusicGenBackend()
    yield
    musicgen_app._backend = original


@pytest.fixture()
def client():
    return TestClient(musicgen_app.app)


def _silent_wav(duration_s: float = 1.0) -> bytes:
    n = int(duration_s * 16000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "mock"
    assert data["vram_mb"] == 0


def test_health_503_when_no_backend(client):
    musicgen_app._backend = None
    resp = client.get("/health")
    assert resp.status_code == 503


# ── /continue ─────────────────────────────────────────────────────────────────


def test_continue_returns_audio(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "5.0"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_continue_duration_header(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "7.0"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.status_code == 200
    assert float(resp.headers["x-duration-s"]) == pytest.approx(7.0)


def test_continue_model_header(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "5.0"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.headers["x-model"] == "mock"


def test_continue_rejects_zero_duration(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "0"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.status_code == 422


def test_continue_rejects_too_long_duration(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "61"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.status_code == 422


def test_continue_rejects_empty_audio(client):
    resp = client.post(
        "/continue",
        data={"duration_s": "5.0"},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_continue_503_when_no_backend(client):
    musicgen_app._backend = None
    resp = client.post(
        "/continue",
        data={"duration_s": "5.0"},
        files={"audio": ("test.wav", _silent_wav(), "audio/wav")},
    )
    assert resp.status_code == 503
