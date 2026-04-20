"""Tests for deliver_activity — mocked at requests layer."""
import json
import pytest
from unittest.mock import MagicMock, patch

from circuitforge_core.activitypub.actor import generate_rsa_keypair, make_actor
from circuitforge_core.activitypub.delivery import deliver_activity
from circuitforge_core.activitypub.objects import make_note, make_create

ACTOR_ID = "https://kiwi.example.com/actors/kiwi"
INBOX_URL = "https://lemmy.example.com/inbox"


@pytest.fixture(scope="module")
def actor():
    priv, pub = generate_rsa_keypair(bits=1024)
    return make_actor(ACTOR_ID, "kiwi", "Kiwi", priv, pub)


@pytest.fixture(scope="module")
def activity(actor):
    note = make_note(ACTOR_ID, "Hello Lemmy!")
    return make_create(actor, note)


class TestDeliverActivity:
    def test_posts_to_inbox_url(self, actor, activity):
        mock_resp = MagicMock(status_code=202)
        with patch("circuitforge_core.activitypub.delivery.requests.post", return_value=mock_resp) as mock_post:
            deliver_activity(activity, INBOX_URL, actor)
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == INBOX_URL

    def test_content_type_is_activity_json(self, actor, activity):
        mock_resp = MagicMock(status_code=202)
        with patch("circuitforge_core.activitypub.delivery.requests.post", return_value=mock_resp) as mock_post:
            deliver_activity(activity, INBOX_URL, actor)
        headers = mock_post.call_args[1]["headers"]
        assert headers.get("Content-Type") == "application/activity+json"

    def test_body_is_json_serialized(self, actor, activity):
        mock_resp = MagicMock(status_code=202)
        with patch("circuitforge_core.activitypub.delivery.requests.post", return_value=mock_resp) as mock_post:
            deliver_activity(activity, INBOX_URL, actor)
        body = mock_post.call_args[1]["data"]
        parsed = json.loads(body)
        assert parsed["type"] == "Create"

    def test_signature_header_present(self, actor, activity):
        mock_resp = MagicMock(status_code=202)
        with patch("circuitforge_core.activitypub.delivery.requests.post", return_value=mock_resp) as mock_post:
            deliver_activity(activity, INBOX_URL, actor)
        headers = mock_post.call_args[1]["headers"]
        assert "Signature" in headers

    def test_returns_response(self, actor, activity):
        mock_resp = MagicMock(status_code=202)
        with patch("circuitforge_core.activitypub.delivery.requests.post", return_value=mock_resp):
            result = deliver_activity(activity, INBOX_URL, actor)
        assert result is mock_resp
