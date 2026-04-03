"""Tests for HeimdallAuthMiddleware — TTL cache and request gating."""
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from circuitforge_core.resources.coordinator.auth import (
    HeimdallAuthMiddleware,
    _ValidationCache,
    CACHE_TTL_S,
)


# ── Cache unit tests ──────────────────────────────────────────────────────────

def test_cache_miss_returns_none():
    cache = _ValidationCache()
    assert cache.get("nonexistent") is None


def test_cache_stores_and_retrieves():
    cache = _ValidationCache()
    cache.set("key1", valid=True, tier="paid", user_id="u1")
    entry = cache.get("key1")
    assert entry is not None
    assert entry.valid is True
    assert entry.tier == "paid"


def test_cache_entry_expires():
    cache = _ValidationCache(ttl_s=0.05)
    cache.set("key1", valid=True, tier="paid", user_id="u1")
    time.sleep(0.1)
    assert cache.get("key1") is None


def test_cache_evict_removes_key():
    cache = _ValidationCache()
    cache.set("key1", valid=True, tier="paid", user_id="u1")
    cache.evict("key1")
    assert cache.get("key1") is None


def test_cache_prune_removes_expired():
    cache = _ValidationCache(ttl_s=0.05)
    cache.set("k1", valid=True, tier="paid", user_id="")
    cache.set("k2", valid=True, tier="paid", user_id="")
    time.sleep(0.1)
    removed = cache.prune()
    assert removed == 2


# ── Middleware integration tests ──────────────────────────────────────────────

def _make_app_with_auth(middleware: HeimdallAuthMiddleware) -> TestClient:
    app = FastAPI()
    app.middleware("http")(middleware)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/services/vllm/allocate")
    def allocate():
        return {"allocation_id": "abc", "url": "http://gpu:8000"}

    return TestClient(app, raise_server_exceptions=False)


def _patched_middleware(valid: bool, tier: str = "paid") -> HeimdallAuthMiddleware:
    """Return a middleware whose Heimdall call is pre-mocked."""
    mw = HeimdallAuthMiddleware(
        heimdall_url="http://heimdall.test",
        min_tier="paid",
    )
    mw._validate_against_heimdall = MagicMock(  # type: ignore[method-assign]
        return_value=(valid, tier, "user-1" if valid else "")
    )
    return mw


def test_health_exempt_no_auth_required():
    mw = _patched_middleware(valid=True)
    client = _make_app_with_auth(mw)
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_missing_auth_header_returns_401():
    mw = _patched_middleware(valid=True)
    client = _make_app_with_auth(mw)
    resp = client.post("/api/services/vllm/allocate")
    assert resp.status_code == 401


def test_invalid_key_returns_403():
    mw = _patched_middleware(valid=False)
    client = _make_app_with_auth(mw)
    resp = client.post(
        "/api/services/vllm/allocate",
        headers={"Authorization": "Bearer BAD-KEY"},
    )
    assert resp.status_code == 403


def test_valid_paid_key_passes():
    mw = _patched_middleware(valid=True, tier="paid")
    client = _make_app_with_auth(mw)
    resp = client.post(
        "/api/services/vllm/allocate",
        headers={"Authorization": "Bearer CFG-KIWI-GOOD-GOOD-GOOD"},
    )
    assert resp.status_code == 200


def test_free_tier_key_rejected_when_min_is_paid():
    mw = _patched_middleware(valid=True, tier="free")
    client = _make_app_with_auth(mw)
    resp = client.post(
        "/api/services/vllm/allocate",
        headers={"Authorization": "Bearer CFG-KIWI-FREE-FREE-FREE"},
    )
    assert resp.status_code == 403
    assert "paid" in resp.json()["detail"]


def test_cache_prevents_second_heimdall_call():
    mw = _patched_middleware(valid=True, tier="paid")
    client = _make_app_with_auth(mw)
    key = "CFG-KIWI-CACHED-KEY-1"
    headers = {"Authorization": f"Bearer {key}"}
    client.post("/api/services/vllm/allocate", headers=headers)
    client.post("/api/services/vllm/allocate", headers=headers)
    # Heimdall should only have been called once — second hit is from cache
    assert mw._validate_against_heimdall.call_count == 1  # type: ignore[attr-defined]


def test_from_env_returns_none_without_heimdall_url(monkeypatch):
    monkeypatch.delenv("HEIMDALL_URL", raising=False)
    assert HeimdallAuthMiddleware.from_env() is None


def test_from_env_returns_middleware_when_set(monkeypatch):
    monkeypatch.setenv("HEIMDALL_URL", "http://heimdall.test")
    mw = HeimdallAuthMiddleware.from_env()
    assert mw is not None
    assert mw._heimdall == "http://heimdall.test"
