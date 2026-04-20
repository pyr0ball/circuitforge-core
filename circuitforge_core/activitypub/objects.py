"""
ActivityStreams 2.0 object constructors.

All functions return plain dicts (no classes) — they are serialized to JSON
for delivery. IDs are minted with UUID4 so callers don't need to track them.

Custom types:
- "Offer"   — AS2 Offer (Rook exchange offers)
- "Request" — custom CF extension (Rook exchange requests); not in core AS2

MIT licensed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuitforge_core.activitypub.actor import CFActor

# AS2 public address (all followers)
PUBLIC = "https://www.w3.org/ns/activitystreams#Public"

# Custom context extension for CF-specific types
_CF_CONTEXT = "https://circuitforge.tech/ns/activitystreams"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _mint_id(actor_id: str, type_slug: str) -> str:
    """Generate a unique ID scoped to the actor's namespace."""
    return f"{actor_id}/{type_slug}/{uuid.uuid4().hex}"


def make_note(
    actor_id: str,
    content: str,
    to: list[str] | None = None,
    cc: list[str] | None = None,
    in_reply_to: str | None = None,
    tag: list[dict] | None = None,
    published: datetime | None = None,
) -> dict:
    """
    Construct an AS2 Note object.

    Args:
        actor_id:    The actor's ID URL (attributedTo).
        content:     HTML or plain-text body.
        to:          Direct recipients (defaults to [PUBLIC]).
        cc:          CC recipients.
        in_reply_to: URL of the parent note when replying.
        tag:         Mention/hashtag tag dicts.
        published:   Post timestamp (defaults to now UTC).
    """
    note: dict = {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": _mint_id(actor_id, "notes"),
        "type": "Note",
        "attributedTo": actor_id,
        "content": content,
        "to": to if to is not None else [PUBLIC],
        "published": published.isoformat().replace("+00:00", "Z") if published else _now_iso(),
    }
    if cc:
        note["cc"] = cc
    if in_reply_to:
        note["inReplyTo"] = in_reply_to
    if tag:
        note["tag"] = tag
    return note


def make_offer(
    actor_id: str,
    summary: str,
    content: str,
    to: list[str] | None = None,
    cc: list[str] | None = None,
) -> dict:
    """
    Construct an AS2 Offer object (Rook exchange offers).

    The Offer type is part of core ActivityStreams 2.0.

    Args:
        actor_id: The actor's ID URL (actor field).
        summary:  Short one-line description (used as title in Lemmy).
        content:  Full HTML/plain-text description.
        to:       Recipients (defaults to [PUBLIC]).
        cc:       CC recipients.
    """
    return {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": _mint_id(actor_id, "offers"),
        "type": "Offer",
        "actor": actor_id,
        "summary": summary,
        "content": content,
        "to": to if to is not None else [PUBLIC],
        "cc": cc or [],
        "published": _now_iso(),
    }


def make_request(
    actor_id: str,
    summary: str,
    content: str,
    to: list[str] | None = None,
    cc: list[str] | None = None,
) -> dict:
    """
    Construct a CF-extension Request object (Rook exchange requests).

    "Request" is not in core AS2 vocabulary — the CF namespace context
    extension is included so federating servers don't reject it.

    Args:
        actor_id: The actor's ID URL.
        summary:  Short one-line description.
        content:  Full HTML/plain-text description.
        to:       Recipients (defaults to [PUBLIC]).
        cc:       CC recipients.
    """
    return {
        "@context": [
            "https://www.w3.org/ns/activitystreams",
            _CF_CONTEXT,
        ],
        "id": _mint_id(actor_id, "requests"),
        "type": "Request",
        "actor": actor_id,
        "summary": summary,
        "content": content,
        "to": to if to is not None else [PUBLIC],
        "cc": cc or [],
        "published": _now_iso(),
    }


def make_create(actor: "CFActor", obj: dict) -> dict:
    """
    Wrap any object dict in an AS2 Create activity.

    The Create activity's id, actor, to, cc, and published fields are
    derived from the wrapped object where available.

    Args:
        actor: The CFActor originating the Create.
        obj:   An object dict (Note, Offer, Request, etc.).
    """
    # Propagate context from inner object if it's a list (custom types)
    ctx = obj.get("@context", "https://www.w3.org/ns/activitystreams")

    return {
        "@context": ctx,
        "id": _mint_id(actor.actor_id, "activities"),
        "type": "Create",
        "actor": actor.actor_id,
        "to": obj.get("to", [PUBLIC]),
        "cc": obj.get("cc", []),
        "published": obj.get("published", _now_iso()),
        "object": obj,
    }
