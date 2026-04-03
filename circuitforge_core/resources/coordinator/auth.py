"""
cf-orch coordinator auth middleware.

When HEIMDALL_URL is set, all /api/* requests (except /api/health) must carry:
    Authorization: Bearer <CF license key>

The key is validated against Heimdall and the result cached for
CACHE_TTL_S seconds (default 300 / 5 min). This keeps Heimdall out of the
per-allocation hot path while keeping revocation latency bounded.

When HEIMDALL_URL is not set, auth is disabled — self-hosted deployments work
with no configuration change.

Environment variables
---------------------
HEIMDALL_URL          Heimdall base URL, e.g. https://license.circuitforge.tech
                      When absent, auth is skipped entirely.
HEIMDALL_MIN_TIER     Minimum tier required (default: "paid").
                      Accepted values: free, paid, premium, ultra.
CF_ORCH_AUTH_SECRET   Shared secret sent to Heimdall so it can distinguish
                      coordinator service calls from end-user requests.
                      Must match the COORDINATOR_SECRET env var on Heimdall.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Unauthenticated paths — health check must always be accessible for monitoring.
_EXEMPT_PATHS: frozenset[str] = frozenset({"/api/health", "/", "/openapi.json", "/docs", "/redoc"})

_TIER_ORDER: dict[str, int] = {"free": 0, "paid": 1, "premium": 2, "ultra": 3}

CACHE_TTL_S: float = 300.0  # 5 minutes — matches Kiwi cloud session TTL


@dataclass
class _CacheEntry:
    valid: bool
    tier: str
    user_id: str
    expires_at: float


class _ValidationCache:
    """Thread-safe TTL cache for Heimdall validation results."""

    def __init__(self, ttl_s: float = CACHE_TTL_S) -> None:
        self._ttl = ttl_s
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()

    def get(self, key: str) -> _CacheEntry | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or time.monotonic() > entry.expires_at:
                return None
            return entry

    def set(self, key: str, valid: bool, tier: str, user_id: str) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(
                valid=valid,
                tier=tier,
                user_id=user_id,
                expires_at=time.monotonic() + self._ttl,
            )

    def evict(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def prune(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, e in self._store.items() if now > e.expires_at]
            for k in expired:
                del self._store[k]
        return len(expired)


class HeimdallAuthMiddleware:
    """
    ASGI middleware that validates CF license keys against Heimdall.

    Attach to a FastAPI app via app.middleware("http"):

        middleware = HeimdallAuthMiddleware.from_env()
        if middleware:
            app.middleware("http")(middleware)
    """

    def __init__(
        self,
        heimdall_url: str,
        min_tier: str = "paid",
        auth_secret: str = "",
        cache_ttl_s: float = CACHE_TTL_S,
    ) -> None:
        self._heimdall = heimdall_url.rstrip("/")
        self._min_tier_rank = _TIER_ORDER.get(min_tier, 1)
        self._min_tier = min_tier
        self._auth_secret = auth_secret
        self._cache = _ValidationCache(ttl_s=cache_ttl_s)
        logger.info(
            "[cf-orch auth] Heimdall auth enabled — url=%s min_tier=%s ttl=%ss",
            self._heimdall, min_tier, cache_ttl_s,
        )

    @classmethod
    def from_env(cls) -> "HeimdallAuthMiddleware | None":
        """Return a configured middleware instance, or None if HEIMDALL_URL is not set."""
        url = os.environ.get("HEIMDALL_URL", "")
        if not url:
            logger.info("[cf-orch auth] HEIMDALL_URL not set — auth disabled (self-hosted mode)")
            return None
        return cls(
            heimdall_url=url,
            min_tier=os.environ.get("HEIMDALL_MIN_TIER", "paid"),
            auth_secret=os.environ.get("CF_ORCH_AUTH_SECRET", ""),
        )

    def _validate_against_heimdall(self, license_key: str) -> tuple[bool, str, str]:
        """
        Call Heimdall's /licenses/verify endpoint.

        Returns (valid, tier, user_id).
        On any network or parse error, returns (False, "", "") — fail closed.
        """
        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._auth_secret:
                headers["X-Coordinator-Secret"] = self._auth_secret
            resp = httpx.post(
                f"{self._heimdall}/licenses/verify",
                json={"key": license_key, "min_tier": self._min_tier},
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("valid", False), data.get("tier", ""), data.get("user_id", "")
            # 401/403 from Heimdall = key invalid/insufficient tier
            logger.debug("[cf-orch auth] Heimdall returned %s for key ...%s", resp.status_code, license_key[-6:])
            return False, "", ""
        except Exception as exc:
            logger.warning("[cf-orch auth] Heimdall unreachable — failing closed: %s", exc)
            return False, "", ""

    def _check_key(self, license_key: str) -> tuple[bool, str]:
        """
        Validate key (cache-first). Returns (authorized, reason_if_denied).
        """
        cached = self._cache.get(license_key)
        if cached is not None:
            if not cached.valid:
                return False, "license key invalid or expired"
            if _TIER_ORDER.get(cached.tier, -1) < self._min_tier_rank:
                return False, f"feature requires {self._min_tier} tier (have: {cached.tier})"
            return True, ""

        valid, tier, user_id = self._validate_against_heimdall(license_key)
        self._cache.set(license_key, valid=valid, tier=tier, user_id=user_id)

        if not valid:
            return False, "license key invalid or expired"
        if _TIER_ORDER.get(tier, -1) < self._min_tier_rank:
            return False, f"feature requires {self._min_tier} tier (have: {tier})"
        return True, ""

    async def __call__(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization: Bearer <license_key> required"},
            )

        license_key = auth_header.removeprefix("Bearer ").strip()
        authorized, reason = self._check_key(license_key)
        if not authorized:
            return JSONResponse(status_code=403, content={"detail": reason})

        return await call_next(request)
