"""
ActivityPub inbox router — FastAPI stub for receiving federated activities.

Products mount this router to handle incoming Create, Follow, Like, Announce,
and other ActivityPub activities from the Fediverse.

Requires fastapi (optional dep). ImportError is raised with a clear message
when fastapi is not installed.

NOTE: from __future__ import annotations is intentionally omitted here.
FastAPI resolves route parameter annotations against module globals at
definition time; lazy string annotations break the Request injection.

MIT licensed.
"""

import json as _json
import re
from typing import Awaitable, Callable

# Handler type: receives (activity_dict, request_headers) and returns None
InboxHandler = Callable[[dict, dict], Awaitable[None]]

# FastAPI imports at module level so annotations resolve correctly.
# Products that don't use the inbox router are not affected by this import
# since circuitforge_core.activitypub.__init__ does NOT import inbox.
try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Provide stubs so the module can be imported without fastapi
    APIRouter = None       # type: ignore[assignment,misc]
    HTTPException = None   # type: ignore[assignment]
    Request = None         # type: ignore[assignment]
    JSONResponse = None    # type: ignore[assignment]


def make_inbox_router(
    handlers: dict[str, InboxHandler] | None = None,
    verify_key_fetcher: Callable[[str], Awaitable[str | None]] | None = None,
    path: str = "/inbox",
) -> "APIRouter":  # type: ignore[name-defined]
    """
    Build a FastAPI router that handles ActivityPub inbox POSTs.

    The router:
    1. Parses the JSON body into an activity dict
    2. Optionally verifies the HTTP Signature (when verify_key_fetcher is provided)
    3. Dispatches activity["type"] to the matching handler from *handlers*
    4. Returns 202 Accepted on success, 400 on bad JSON, 401 on bad signature

    Args:
        handlers:            Dict mapping activity type strings (e.g. "Create",
                             "Follow") to async handler callables.
        verify_key_fetcher:  Async callable that takes a keyId URL and returns the
                             actor's public key PEM, or None if not found.
                             When None, signature verification is skipped (dev mode).
        path:                Inbox endpoint path (default "/inbox").

    Returns:
        FastAPI APIRouter.

    Example::

        async def on_create(activity: dict, headers: dict) -> None:
            print("Received Create:", activity)

        router = make_inbox_router(handlers={"Create": on_create})
        app.include_router(router, prefix="/actors/kiwi")
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "circuitforge_core.activitypub.inbox requires fastapi. "
            "Install with: pip install fastapi"
        )

    from circuitforge_core.activitypub.signing import verify_signature

    router = APIRouter()
    _handlers: dict[str, InboxHandler] = handlers or {}

    @router.post(path, status_code=202)
    async def inbox_endpoint(request: Request) -> JSONResponse:
        # Parse body — read bytes first (needed for signature verification),
        # then decode JSON manually to avoid double-read issues.
        try:
            body = await request.body()
            activity = _json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        # Optional signature verification
        if verify_key_fetcher is not None:
            sig_header = request.headers.get("Signature", "")
            key_id = _parse_key_id(sig_header)
            if not key_id:
                raise HTTPException(status_code=401, detail="Missing or malformed Signature header.")
            public_key_pem = await verify_key_fetcher(key_id)
            if public_key_pem is None:
                raise HTTPException(status_code=401, detail=f"Unknown keyId: {key_id}")
            ok = verify_signature(
                headers=dict(request.headers),
                method="POST",
                path=request.url.path,
                body=body,
                public_key_pem=public_key_pem,
            )
            if not ok:
                raise HTTPException(status_code=401, detail="Signature verification failed.")

        activity_type = activity.get("type", "")
        handler = _handlers.get(activity_type)
        if handler is None:
            # Unknown types are silently accepted per AP spec — return 202
            return JSONResponse(status_code=202, content={"status": "accepted", "type": activity_type})

        await handler(activity, dict(request.headers))
        return JSONResponse(status_code=202, content={"status": "accepted"})

    return router


def _parse_key_id(sig_header: str) -> str | None:
    """Extract keyId value from a Signature header string."""
    match = re.search(r'keyId="([^"]+)"', sig_header)
    return match.group(1) if match else None
