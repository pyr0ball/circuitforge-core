"""
circuitforge_core.activitypub — ActivityPub actor management, object construction,
HTTP Signature signing, delivery, and Lemmy integration.

MIT licensed.
"""

from circuitforge_core.activitypub.actor import (
    CFActor,
    generate_rsa_keypair,
    load_actor_from_key_file,
    make_actor,
)
from circuitforge_core.activitypub.delivery import deliver_activity
from circuitforge_core.activitypub.lemmy import (
    LemmyAuthError,
    LemmyClient,
    LemmyCommunityNotFound,
    LemmyConfig,
)
from circuitforge_core.activitypub.objects import (
    PUBLIC,
    make_create,
    make_note,
    make_offer,
    make_request,
)
from circuitforge_core.activitypub.signing import sign_headers, verify_signature

__all__ = [
    # Actor
    "CFActor",
    "generate_rsa_keypair",
    "load_actor_from_key_file",
    "make_actor",
    # Objects
    "PUBLIC",
    "make_note",
    "make_offer",
    "make_request",
    "make_create",
    # Signing
    "sign_headers",
    "verify_signature",
    # Delivery
    "deliver_activity",
    # Lemmy
    "LemmyConfig",
    "LemmyClient",
    "LemmyAuthError",
    "LemmyCommunityNotFound",
]

# inbox is optional (requires fastapi) — import it when needed:
#   from circuitforge_core.activitypub.inbox import make_inbox_router
