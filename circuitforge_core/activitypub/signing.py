"""
HTTP Signatures for ActivityPub (draft-cavage-http-signatures-08).

This is the signing convention used by Mastodon, Lemmy, and the broader
ActivityPub ecosystem. It is distinct from the newer RFC 9421.

Signing algorithm: rsa-sha256
Signed headers: (request-target) host date [digest] content-type
Digest header: SHA-256 of request body (when body is present)
keyId: {actor.actor_id}#main-key

MIT licensed.
"""
from __future__ import annotations

import base64
import hashlib
import re
from email.utils import formatdate
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from circuitforge_core.activitypub.actor import CFActor


def _rfc1123_now() -> str:
    """Return current UTC time in RFC 1123 format as required by HTTP Date header."""
    return formatdate(usegmt=True)


def _sha256_digest(body: bytes) -> str:
    """Return 'SHA-256=<base64>' digest string for body."""
    digest = hashlib.sha256(body).digest()
    return f"SHA-256={base64.b64encode(digest).decode()}"


def sign_headers(
    method: str,
    url: str,
    headers: dict,
    body: bytes | None,
    actor: "CFActor",  # type: ignore[name-defined]
) -> dict:
    """
    Return a new headers dict with Date, Digest (if body), and Signature added.

    The input *headers* dict is not mutated.

    Args:
        method:  HTTP method string (e.g. "POST"), case-insensitive.
        url:     Full request URL.
        headers: Existing headers dict (Content-Type, etc.).
        body:    Request body bytes, or None for bodyless requests.
        actor:   CFActor whose private key signs the request.

    Returns:
        New dict with all original headers plus Date, Digest (if body), Signature.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    method_lower = method.lower()
    date = _rfc1123_now()

    out = dict(headers)
    out["Date"] = date
    out["Host"] = host

    signed_header_names = ["(request-target)", "host", "date"]

    if body is not None:
        digest = _sha256_digest(body)
        out["Digest"] = digest
        signed_header_names.append("digest")

    if "Content-Type" in out:
        signed_header_names.append("content-type")

    # Build the signature string — header names in the spec are lowercase,
    # but the dict uses Title-Case HTTP convention, so look up case-insensitively.
    def _ci_get(d: dict, key: str) -> str:
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(key)

    lines = []
    for name in signed_header_names:
        if name == "(request-target)":
            lines.append(f"(request-target): {method_lower} {path}")
        else:
            lines.append(f"{name}: {_ci_get(out, name)}")

    signature_string = "\n".join(lines).encode()

    private_key = load_pem_private_key(actor.private_key_pem.encode(), password=None)
    raw_sig = private_key.sign(signature_string, padding.PKCS1v15(), hashes.SHA256())
    b64_sig = base64.b64encode(raw_sig).decode()

    key_id = f"{actor.actor_id}#main-key"
    headers_param = " ".join(signed_header_names)

    out["Signature"] = (
        f'keyId="{key_id}",'
        f'algorithm="rsa-sha256",'
        f'headers="{headers_param}",'
        f'signature="{b64_sig}"'
    )

    return out


def verify_signature(
    headers: dict,
    method: str,
    path: str,
    body: bytes | None,
    public_key_pem: str,
) -> bool:
    """
    Verify an incoming ActivityPub HTTP Signature.

    Returns False on any parse or verification failure — never raises.

    Args:
        headers:        Request headers dict (case-insensitive lookup attempted).
        method:         HTTP method (e.g. "POST").
        path:           Request path (e.g. "/actors/kiwi/inbox").
        body:           Raw request body bytes, or None.
        public_key_pem: PEM-encoded RSA public key of the signing actor.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        # Case-insensitive header lookup helper
        def _get(name: str) -> str | None:
            name_lower = name.lower()
            for k, v in headers.items():
                if k.lower() == name_lower:
                    return v
            return None

        sig_header = _get("Signature")
        if not sig_header:
            return False

        # Parse Signature header key=value pairs
        params: dict[str, str] = {}
        for match in re.finditer(r'(\w+)="([^"]*)"', sig_header):
            params[match.group(1)] = match.group(2)

        if "signature" not in params or "headers" not in params:
            return False

        signed_header_names = params["headers"].split()
        method_lower = method.lower()

        lines = []
        for name in signed_header_names:
            if name == "(request-target)":
                lines.append(f"(request-target): {method_lower} {path}")
            else:
                val = _get(name)
                if val is None:
                    return False
                lines.append(f"{name}: {val}")

        signature_string = "\n".join(lines).encode()
        raw_sig = base64.b64decode(params["signature"])

        public_key = load_pem_public_key(public_key_pem.encode())
        public_key.verify(raw_sig, signature_string, padding.PKCS1v15(), hashes.SHA256())

        # Also verify the Digest header matches the actual body, if both are present.
        # Signing the Digest header proves it wasn't swapped; re-computing it proves
        # the body wasn't replaced after signing.
        digest_val = _get("Digest")
        if digest_val and body is not None:
            expected = _sha256_digest(body)
            if digest_val != expected:
                return False

        return True

    except (InvalidSignature, Exception):
        return False
