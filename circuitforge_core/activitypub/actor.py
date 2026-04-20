"""
CFActor — ActivityPub actor identity for CircuitForge products.

An actor holds RSA key material and its ActivityPub identity URLs.
The private key is in-memory only; to_ap_dict() never includes it.

MIT licensed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CFActor:
    """ActivityPub actor for a CircuitForge product instance."""

    actor_id: str          # e.g. "https://kiwi.circuitforge.tech/actors/kiwi"
    username: str
    display_name: str
    inbox_url: str
    outbox_url: str
    public_key_pem: str
    private_key_pem: str   # Never included in to_ap_dict()
    icon_url: str | None = None
    summary: str | None = None

    def to_ap_dict(self) -> dict:
        """Return an ActivityPub Person/Application object (public only)."""
        obj: dict = {
            "@context": [
                "https://www.w3.org/ns/activitystreams",
                "https://w3id.org/security/v1",
            ],
            "id": self.actor_id,
            "type": "Application",
            "preferredUsername": self.username,
            "name": self.display_name,
            "inbox": self.inbox_url,
            "outbox": self.outbox_url,
            "publicKey": {
                "id": f"{self.actor_id}#main-key",
                "owner": self.actor_id,
                "publicKeyPem": self.public_key_pem,
            },
        }
        if self.summary:
            obj["summary"] = self.summary
        if self.icon_url:
            obj["icon"] = {
                "type": "Image",
                "mediaType": "image/png",
                "url": self.icon_url,
            }
        return obj


def generate_rsa_keypair(bits: int = 2048) -> tuple[str, str]:
    """
    Generate a new RSA keypair.

    Returns:
        (private_key_pem, public_key_pem) as PEM-encoded strings.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def make_actor(
    actor_id: str,
    username: str,
    display_name: str,
    private_key_pem: str,
    public_key_pem: str,
    icon_url: str | None = None,
    summary: str | None = None,
) -> CFActor:
    """
    Construct a CFActor from an existing keypair.

    Inbox and outbox URLs are derived from actor_id by convention:
      {actor_id}/inbox and {actor_id}/outbox
    """
    return CFActor(
        actor_id=actor_id,
        username=username,
        display_name=display_name,
        inbox_url=f"{actor_id}/inbox",
        outbox_url=f"{actor_id}/outbox",
        public_key_pem=public_key_pem,
        private_key_pem=private_key_pem,
        icon_url=icon_url,
        summary=summary,
    )


def load_actor_from_key_file(
    actor_id: str,
    username: str,
    display_name: str,
    private_key_path: str,
    icon_url: str | None = None,
    summary: str | None = None,
) -> CFActor:
    """
    Load a CFActor from a PEM private key file on disk.

    The public key is derived from the private key — no separate public key
    file is required.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    pem_bytes = Path(private_key_path).read_bytes()
    private_key = load_pem_private_key(pem_bytes, password=None)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return make_actor(
        actor_id=actor_id,
        username=username,
        display_name=display_name,
        private_key_pem=private_pem,
        public_key_pem=public_pem,
        icon_url=icon_url,
        summary=summary,
    )
