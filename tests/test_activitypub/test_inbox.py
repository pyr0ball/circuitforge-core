"""Tests for the ActivityPub inbox FastAPI router."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from circuitforge_core.activitypub.actor import generate_rsa_keypair, make_actor
from circuitforge_core.activitypub.inbox import make_inbox_router
from circuitforge_core.activitypub.signing import sign_headers

ACTOR_ID = "https://kiwi.example.com/actors/kiwi"


@pytest.fixture(scope="module")
def actor():
    priv, pub = generate_rsa_keypair(bits=1024)
    return make_actor(ACTOR_ID, "kiwi", "Kiwi", priv, pub)


@pytest.fixture
def app_no_verify():
    """App with inbox router, no signature verification (dev mode)."""
    received = []

    async def on_create(activity, headers):
        received.append(activity)

    router = make_inbox_router(handlers={"Create": on_create})
    app = FastAPI()
    app.include_router(router)
    app._received = received
    return app


@pytest.fixture
def client_no_verify(app_no_verify):
    return TestClient(app_no_verify)


class TestInboxNoVerification:
    def test_202_on_known_activity_type(self, client_no_verify):
        resp = client_no_verify.post(
            "/inbox",
            json={"type": "Create", "actor": ACTOR_ID, "object": {}},
        )
        assert resp.status_code == 202

    def test_202_on_unknown_activity_type(self, client_no_verify):
        resp = client_no_verify.post(
            "/inbox",
            json={"type": "Undo", "actor": ACTOR_ID},
        )
        assert resp.status_code == 202

    def test_response_body_contains_accepted(self, client_no_verify):
        resp = client_no_verify.post(
            "/inbox",
            json={"type": "Create", "actor": ACTOR_ID, "object": {}},
        )
        assert resp.json()["status"] == "accepted"

    def test_400_on_invalid_json(self, client_no_verify):
        resp = client_no_verify.post(
            "/inbox",
            data=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_handler_called_with_activity(self, app_no_verify, client_no_verify):
        app_no_verify._received.clear()
        activity = {"type": "Create", "actor": ACTOR_ID, "object": {"type": "Note"}}
        client_no_verify.post("/inbox", json=activity)
        assert len(app_no_verify._received) == 1
        assert app_no_verify._received[0]["type"] == "Create"

    def test_no_handlers_still_returns_202(self):
        router = make_inbox_router()
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as c:
            resp = c.post("/inbox", json={"type": "Follow"})
        assert resp.status_code == 202


class TestInboxWithSignatureVerification:
    def test_valid_signature_accepted(self, actor):
        async def key_fetcher(key_id: str):
            return actor.public_key_pem

        router = make_inbox_router(handlers={}, verify_key_fetcher=key_fetcher)
        app = FastAPI()
        app.include_router(router)

        activity = {"type": "Create", "actor": ACTOR_ID}
        body = json.dumps(activity).encode()
        headers = sign_headers(
            method="POST",
            url="http://testserver/inbox",
            headers={"Content-Type": "application/activity+json"},
            body=body,
            actor=actor,
        )

        with TestClient(app) as c:
            resp = c.post("/inbox", content=body, headers=headers)
        assert resp.status_code == 202

    def test_missing_signature_returns_401(self, actor):
        async def key_fetcher(key_id: str):
            return actor.public_key_pem

        router = make_inbox_router(handlers={}, verify_key_fetcher=key_fetcher)
        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as c:
            resp = c.post("/inbox", json={"type": "Create"})
        assert resp.status_code == 401

    def test_unknown_key_id_returns_401(self, actor):
        async def key_fetcher(key_id: str):
            return None  # Unknown actor

        router = make_inbox_router(handlers={}, verify_key_fetcher=key_fetcher)
        app = FastAPI()
        app.include_router(router)

        activity = {"type": "Create"}
        body = json.dumps(activity).encode()
        headers = sign_headers("POST", "http://testserver/inbox", {}, body, actor)

        with TestClient(app) as c:
            resp = c.post("/inbox", content=body, headers=headers)
        assert resp.status_code == 401


class TestMakeInboxRouterImportError:
    def test_raises_on_missing_fastapi(self, monkeypatch):
        import circuitforge_core.activitypub.inbox as inbox_mod
        monkeypatch.setattr(inbox_mod, "_FASTAPI_AVAILABLE", False)
        with pytest.raises(ImportError, match="fastapi"):
            make_inbox_router()
