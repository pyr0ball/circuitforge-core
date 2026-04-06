"""Tests for circuitforge_core.config.license."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

import circuitforge_core.config.license as license_module
from circuitforge_core.config.license import get_license_tier, validate_license


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level cache before each test."""
    license_module._cache.clear()
    yield
    license_module._cache.clear()


# ---------------------------------------------------------------------------
# 1. validate_license returns _INVALID when CF_LICENSE_KEY not set
# ---------------------------------------------------------------------------
def test_validate_license_no_key_returns_invalid(monkeypatch):
    monkeypatch.delenv("CF_LICENSE_KEY", raising=False)
    result = validate_license("kiwi")
    assert result == {"valid": False, "tier": "free", "user_id": ""}


# ---------------------------------------------------------------------------
# 2. validate_license calls Heimdall and returns valid result when key set
# ---------------------------------------------------------------------------
def test_validate_license_valid_response(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-KIWI-AAAA-BBBB-CCCC")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": True, "tier": "paid", "user_id": "user-42"}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp) as mock_post:
        result = validate_license("kiwi")

    mock_post.assert_called_once()
    assert result == {"valid": True, "tier": "paid", "user_id": "user-42"}


# ---------------------------------------------------------------------------
# 3. validate_license returns invalid when Heimdall returns non-ok status
# ---------------------------------------------------------------------------
def test_validate_license_non_ok_response(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-KIWI-AAAA-BBBB-CCCC")
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 403

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp):
        result = validate_license("kiwi")

    assert result == {"valid": False, "tier": "free", "user_id": ""}


# ---------------------------------------------------------------------------
# 4. validate_license returns invalid when network fails
# ---------------------------------------------------------------------------
def test_validate_license_network_error(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-KIWI-AAAA-BBBB-CCCC")

    with patch(
        "circuitforge_core.config.license.requests.post",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    ):
        result = validate_license("kiwi")

    assert result == {"valid": False, "tier": "free", "user_id": ""}


# ---------------------------------------------------------------------------
# 5. validate_license caches result — second call does NOT make a second request
# ---------------------------------------------------------------------------
def test_validate_license_caches_result(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-KIWI-CACHE-TEST-KEY")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": True, "tier": "paid", "user_id": "user-1"}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp) as mock_post:
        result1 = validate_license("kiwi")
        result2 = validate_license("kiwi")

    assert mock_post.call_count == 1
    assert result1 == result2


# ---------------------------------------------------------------------------
# 6. get_license_tier returns "free" when key absent
# ---------------------------------------------------------------------------
def test_get_license_tier_no_key_returns_free(monkeypatch):
    monkeypatch.delenv("CF_LICENSE_KEY", raising=False)
    assert get_license_tier("snipe") == "free"


# ---------------------------------------------------------------------------
# 7. get_license_tier returns tier string from valid Heimdall response
# ---------------------------------------------------------------------------
def test_get_license_tier_valid_key_returns_tier(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-SNPE-AAAA-BBBB-CCCC")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": True, "tier": "premium", "user_id": "user-7"}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp):
        tier = get_license_tier("snipe")

    assert tier == "premium"


# ---------------------------------------------------------------------------
# 8. get_license_tier returns "free" when Heimdall says valid=False
# ---------------------------------------------------------------------------
def test_get_license_tier_invalid_key_returns_free(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-SNPE-DEAD-DEAD-DEAD")
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": False, "tier": "free", "user_id": ""}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp):
        tier = get_license_tier("snipe")

    assert tier == "free"


# ---------------------------------------------------------------------------
# 9. CF_LICENSE_URL env var overrides the default Heimdall URL
# ---------------------------------------------------------------------------
def test_cf_license_url_override(monkeypatch):
    monkeypatch.setenv("CF_LICENSE_KEY", "CFG-PRNG-AAAA-BBBB-CCCC")
    monkeypatch.setenv("CF_LICENSE_URL", "http://localhost:9000")

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": True, "tier": "paid", "user_id": "u1"}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp) as mock_post:
        validate_license("peregrine")

    call_url = mock_post.call_args[0][0]
    assert call_url.startswith("http://localhost:9000"), (
        f"Expected URL to start with http://localhost:9000, got {call_url!r}"
    )


# ---------------------------------------------------------------------------
# 10. Expired cache entry triggers a fresh Heimdall call
# ---------------------------------------------------------------------------
def test_validate_license_expired_cache_triggers_fresh_call(monkeypatch):
    key = "CFG-KIWI-EXPR-EXPR-EXPR"
    monkeypatch.setenv("CF_LICENSE_KEY", key)

    # Inject an expired cache entry
    expired_result = {"valid": True, "tier": "paid", "user_id": "old-user"}
    license_module._cache[(key, "kiwi")] = (expired_result, time.monotonic() - 1)

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"valid": True, "tier": "premium", "user_id": "new-user"}

    with patch("circuitforge_core.config.license.requests.post", return_value=mock_resp) as mock_post:
        result = validate_license("kiwi")

    mock_post.assert_called_once()
    assert result["tier"] == "premium"
    assert result["user_id"] == "new-user"
