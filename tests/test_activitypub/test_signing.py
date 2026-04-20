"""Tests for HTTP Signature signing and verification (draft-cavage-http-signatures-08)."""
import pytest
from circuitforge_core.activitypub.actor import generate_rsa_keypair, make_actor
from circuitforge_core.activitypub.signing import sign_headers, verify_signature

ACTOR_ID = "https://kiwi.example.com/actors/kiwi"
INBOX_URL = "https://lemmy.example.com/inbox"


@pytest.fixture(scope="module")
def actor():
    priv, pub = generate_rsa_keypair(bits=1024)
    return make_actor(ACTOR_ID, "kiwi", "Kiwi", priv, pub)


class TestSignHeaders:
    def test_date_header_added(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body", actor)
        assert "Date" in headers

    def test_host_header_added(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body", actor)
        assert headers["Host"] == "lemmy.example.com"

    def test_digest_header_added_when_body(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body content", actor)
        assert "Digest" in headers
        assert headers["Digest"].startswith("SHA-256=")

    def test_digest_not_added_when_no_body(self, actor):
        headers = sign_headers("GET", INBOX_URL, {}, None, actor)
        assert "Digest" not in headers

    def test_signature_header_present(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body", actor)
        assert "Signature" in headers

    def test_signature_contains_key_id(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body", actor)
        assert f"{ACTOR_ID}#main-key" in headers["Signature"]

    def test_signature_algorithm_rsa_sha256(self, actor):
        headers = sign_headers("POST", INBOX_URL, {}, b"body", actor)
        assert 'algorithm="rsa-sha256"' in headers["Signature"]

    def test_does_not_mutate_input_headers(self, actor):
        original = {"Content-Type": "application/json"}
        sign_headers("POST", INBOX_URL, original, b"body", actor)
        assert "Signature" not in original

    def test_content_type_signed_when_present(self, actor):
        headers = sign_headers("POST", INBOX_URL, {"Content-Type": "application/activity+json"}, b"x", actor)
        assert "content-type" in headers["Signature"]


class TestVerifySignature:
    def test_valid_signature_returns_true(self, actor):
        body = b'{"type": "Create"}'
        headers = sign_headers("POST", INBOX_URL, {"Content-Type": "application/activity+json"}, body, actor)
        result = verify_signature(
            headers=headers,
            method="POST",
            path="/inbox",
            body=body,
            public_key_pem=actor.public_key_pem,
        )
        assert result is True

    def test_tampered_body_returns_false(self, actor):
        body = b'{"type": "Create"}'
        headers = sign_headers("POST", INBOX_URL, {"Content-Type": "application/activity+json"}, body, actor)
        result = verify_signature(
            headers=headers,
            method="POST",
            path="/inbox",
            body=b"tampered body",
            public_key_pem=actor.public_key_pem,
        )
        assert result is False

    def test_wrong_public_key_returns_false(self, actor):
        _, other_pub = generate_rsa_keypair(bits=1024)
        body = b"hello"
        headers = sign_headers("POST", INBOX_URL, {}, body, actor)
        result = verify_signature(
            headers=headers,
            method="POST",
            path="/inbox",
            body=body,
            public_key_pem=other_pub,
        )
        assert result is False

    def test_missing_signature_header_returns_false(self, actor):
        result = verify_signature(
            headers={"Date": "Mon, 20 Apr 2026 12:00:00 GMT"},
            method="POST",
            path="/inbox",
            body=b"body",
            public_key_pem=actor.public_key_pem,
        )
        assert result is False

    def test_bodyless_get_roundtrip(self, actor):
        headers = sign_headers("GET", INBOX_URL, {}, None, actor)
        result = verify_signature(
            headers=headers,
            method="GET",
            path="/inbox",
            body=None,
            public_key_pem=actor.public_key_pem,
        )
        assert result is True

    def test_wrong_method_fails_verification(self, actor):
        body = b"data"
        headers = sign_headers("POST", INBOX_URL, {}, body, actor)
        # Verify with wrong method — (request-target) will differ
        result = verify_signature(
            headers=headers,
            method="PUT",
            path="/inbox",
            body=body,
            public_key_pem=actor.public_key_pem,
        )
        assert result is False
