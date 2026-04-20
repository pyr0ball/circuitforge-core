"""
ActivityPub HTTP delivery — POST a signed activity to a remote inbox.

Synchronous (uses requests). Async callers can wrap in asyncio.to_thread.

MIT licensed.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import requests

from circuitforge_core.activitypub.signing import sign_headers

if TYPE_CHECKING:
    from circuitforge_core.activitypub.actor import CFActor

ACTIVITY_CONTENT_TYPE = "application/activity+json"


def deliver_activity(
    activity: dict,
    inbox_url: str,
    actor: "CFActor",
    timeout: float = 10.0,
) -> requests.Response:
    """
    POST a signed ActivityPub activity to a remote inbox.

    The activity dict is serialized to JSON, signed with the actor's private
    key (HTTP Signatures, rsa-sha256), and delivered via HTTP POST.

    Args:
        activity:   ActivityPub activity dict (e.g. from make_create()).
        inbox_url:  Target inbox URL (e.g. "https://lemmy.ml/inbox").
        actor:      CFActor whose key signs the request.
        timeout:    Request timeout in seconds.

    Returns:
        The raw requests.Response. Caller decides retry / error policy.

    Raises:
        requests.RequestException: On network-level failure.
    """
    body = json.dumps(activity).encode()
    base_headers = {"Content-Type": ACTIVITY_CONTENT_TYPE}
    signed = sign_headers(
        method="POST",
        url=inbox_url,
        headers=base_headers,
        body=body,
        actor=actor,
    )
    return requests.post(inbox_url, data=body, headers=signed, timeout=timeout)
