"""Tests for ActivityStreams 2.0 object constructors."""
import pytest
from circuitforge_core.activitypub.actor import generate_rsa_keypair, make_actor
from circuitforge_core.activitypub.objects import (
    PUBLIC,
    make_create,
    make_note,
    make_offer,
    make_request,
)

ACTOR_ID = "https://rook.example.com/actors/rook"


@pytest.fixture(scope="module")
def actor():
    priv, pub = generate_rsa_keypair(bits=1024)
    return make_actor(ACTOR_ID, "rook", "Rook Exchange", priv, pub)


class TestPublicConstant:
    def test_is_as_public_uri(self):
        assert PUBLIC == "https://www.w3.org/ns/activitystreams#Public"


class TestMakeNote:
    def test_type_is_note(self):
        assert make_note(ACTOR_ID, "Hello")["type"] == "Note"

    def test_attributed_to_actor(self):
        n = make_note(ACTOR_ID, "Hello")
        assert n["attributedTo"] == ACTOR_ID

    def test_content_stored(self):
        n = make_note(ACTOR_ID, "Hello world")
        assert n["content"] == "Hello world"

    def test_default_to_is_public(self):
        n = make_note(ACTOR_ID, "Hello")
        assert PUBLIC in n["to"]

    def test_custom_to(self):
        n = make_note(ACTOR_ID, "Hello", to=["https://other.example/inbox"])
        assert "https://other.example/inbox" in n["to"]

    def test_cc_present_when_set(self):
        n = make_note(ACTOR_ID, "Hello", cc=["https://x.com/followers"])
        assert n["cc"] == ["https://x.com/followers"]

    def test_cc_absent_when_not_set(self):
        n = make_note(ACTOR_ID, "Hello")
        assert "cc" not in n

    def test_in_reply_to_included(self):
        n = make_note(ACTOR_ID, "Reply", in_reply_to="https://mastodon.social/notes/123")
        assert n["inReplyTo"] == "https://mastodon.social/notes/123"

    def test_in_reply_to_absent_by_default(self):
        assert "inReplyTo" not in make_note(ACTOR_ID, "Hello")

    def test_tag_included(self):
        tag = [{"type": "Mention", "href": "https://mastodon.social/users/alice"}]
        n = make_note(ACTOR_ID, "Hi @alice", tag=tag)
        assert n["tag"] == tag

    def test_id_is_unique(self):
        a = make_note(ACTOR_ID, "Hello")
        b = make_note(ACTOR_ID, "Hello")
        assert a["id"] != b["id"]

    def test_id_scoped_to_actor(self):
        n = make_note(ACTOR_ID, "Hello")
        assert n["id"].startswith(ACTOR_ID)

    def test_published_present(self):
        n = make_note(ACTOR_ID, "Hello")
        assert "published" in n
        assert n["published"].endswith("Z")

    def test_custom_published(self):
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        n = make_note(ACTOR_ID, "Hello", published=ts)
        assert "2026-01-15" in n["published"]

    def test_context_is_activitystreams(self):
        n = make_note(ACTOR_ID, "Hello")
        assert "activitystreams" in n["@context"]


class TestMakeOffer:
    def test_type_is_offer(self):
        assert make_offer(ACTOR_ID, "Free apples", "Lots of apples")["type"] == "Offer"

    def test_summary_stored(self):
        o = make_offer(ACTOR_ID, "Free apples", "Lots of apples")
        assert o["summary"] == "Free apples"

    def test_content_stored(self):
        o = make_offer(ACTOR_ID, "Free apples", "Lots of apples available.")
        assert o["content"] == "Lots of apples available."

    def test_actor_field_set(self):
        o = make_offer(ACTOR_ID, "x", "y")
        assert o["actor"] == ACTOR_ID

    def test_default_to_is_public(self):
        o = make_offer(ACTOR_ID, "x", "y")
        assert PUBLIC in o["to"]

    def test_id_is_unique(self):
        assert make_offer(ACTOR_ID, "x", "y")["id"] != make_offer(ACTOR_ID, "x", "y")["id"]


class TestMakeRequest:
    def test_type_is_request(self):
        assert make_request(ACTOR_ID, "Need a ladder", "Borrowing a ladder")["type"] == "Request"

    def test_context_includes_cf_namespace(self):
        r = make_request(ACTOR_ID, "Need", "Need something")
        ctx = r["@context"]
        assert isinstance(ctx, list)
        assert any("circuitforge" in c for c in ctx)

    def test_summary_stored(self):
        r = make_request(ACTOR_ID, "Need a ladder", "...")
        assert r["summary"] == "Need a ladder"

    def test_actor_field_set(self):
        assert make_request(ACTOR_ID, "x", "y")["actor"] == ACTOR_ID

    def test_id_is_unique(self):
        a = make_request(ACTOR_ID, "x", "y")
        b = make_request(ACTOR_ID, "x", "y")
        assert a["id"] != b["id"]


class TestMakeCreate:
    def test_type_is_create(self, actor):
        note = make_note(ACTOR_ID, "Hello")
        c = make_create(actor, note)
        assert c["type"] == "Create"

    def test_actor_field_matches(self, actor):
        note = make_note(ACTOR_ID, "Hello")
        c = make_create(actor, note)
        assert c["actor"] == ACTOR_ID

    def test_object_is_inner_dict(self, actor):
        note = make_note(ACTOR_ID, "Hello")
        c = make_create(actor, note)
        assert c["object"] is note

    def test_to_propagated_from_object(self, actor):
        note = make_note(ACTOR_ID, "Hello", to=["https://inbox.example/"])
        c = make_create(actor, note)
        assert "https://inbox.example/" in c["to"]

    def test_published_propagated(self, actor):
        note = make_note(ACTOR_ID, "Hello")
        c = make_create(actor, note)
        assert c["published"] == note["published"]

    def test_id_is_unique(self, actor):
        note = make_note(ACTOR_ID, "Hello")
        a = make_create(actor, note)
        b = make_create(actor, note)
        assert a["id"] != b["id"]

    def test_wraps_offer(self, actor):
        offer = make_offer(ACTOR_ID, "Free apples", "Take some")
        c = make_create(actor, offer)
        assert c["object"]["type"] == "Offer"

    def test_wraps_request(self, actor):
        req = make_request(ACTOR_ID, "Need ladder", "...")
        c = make_create(actor, req)
        assert c["object"]["type"] == "Request"
