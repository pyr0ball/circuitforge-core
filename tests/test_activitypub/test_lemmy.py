"""Tests for LemmyClient — mocked at the requests layer."""
import pytest
from unittest.mock import MagicMock, patch

from circuitforge_core.activitypub.lemmy import (
    LemmyAuthError,
    LemmyClient,
    LemmyCommunityNotFound,
    LemmyConfig,
)

CONFIG = LemmyConfig(
    instance_url="https://lemmy.example.com",
    username="kiwi_bot",
    password="s3cret",
)


@pytest.fixture
def client():
    return LemmyClient(CONFIG)


def _mock_response(status_code: int, json_data: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    resp.raise_for_status = MagicMock()
    return resp


class TestLemmyConfig:
    def test_fields_stored(self):
        assert CONFIG.instance_url == "https://lemmy.example.com"
        assert CONFIG.username == "kiwi_bot"
        assert CONFIG.password == "s3cret"

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            CONFIG.username = "other"  # type: ignore[misc]


class TestLogin:
    def test_successful_login_stores_jwt(self, client):
        with patch.object(client._session, "post", return_value=_mock_response(200, {"jwt": "token123"})):
            client.login()
        assert client._jwt == "token123"

    def test_401_raises_lemmy_auth_error(self, client):
        with patch.object(client._session, "post", return_value=_mock_response(401, {})):
            with pytest.raises(LemmyAuthError):
                client.login()

    def test_missing_jwt_field_raises(self, client):
        with patch.object(client._session, "post", return_value=_mock_response(200, {})):
            with pytest.raises(LemmyAuthError, match="missing 'jwt'"):
                client.login()


class TestResolveCommunity:
    def _logged_in_client(self):
        c = LemmyClient(CONFIG)
        c._jwt = "fake-jwt"
        return c

    def test_resolves_exact_name_match(self):
        client = self._logged_in_client()
        community_resp = {
            "communities": [
                {"community": {"id": 42, "name": "cooking", "actor_id": "https://lemmy.world/c/cooking"}}
            ]
        }
        with patch.object(client._session, "get", return_value=_mock_response(200, community_resp)):
            cid = client.resolve_community("cooking")
        assert cid == 42

    def test_resolves_fediverse_address(self):
        client = self._logged_in_client()
        community_resp = {
            "communities": [
                {"community": {"id": 99, "name": "cooking", "actor_id": "https://lemmy.world/c/cooking"}}
            ]
        }
        with patch.object(client._session, "get", return_value=_mock_response(200, community_resp)):
            cid = client.resolve_community("!cooking@lemmy.world")
        assert cid == 99

    def test_empty_results_raises_not_found(self):
        client = self._logged_in_client()
        with patch.object(client._session, "get", return_value=_mock_response(200, {"communities": []})):
            with pytest.raises(LemmyCommunityNotFound):
                client.resolve_community("nonexistent")

    def test_search_failure_raises_not_found(self):
        client = self._logged_in_client()
        with patch.object(client._session, "get", return_value=_mock_response(500, {})):
            with pytest.raises(LemmyCommunityNotFound):
                client.resolve_community("cooking")

    def test_not_logged_in_raises_auth_error(self, client):
        with pytest.raises(LemmyAuthError):
            client.resolve_community("cooking")


class TestPostToCommunity:
    def _logged_in_client(self):
        c = LemmyClient(CONFIG)
        c._jwt = "fake-jwt"
        return c

    def test_successful_post_returns_dict(self):
        client = self._logged_in_client()
        post_resp = {"post_view": {"post": {"id": 123, "name": "Recipe post"}}}
        with patch.object(client._session, "post", return_value=_mock_response(200, post_resp)):
            result = client.post_to_community(42, "Recipe post", "Great recipe!")
        assert result == post_resp

    def test_post_includes_title_and_body(self):
        client = self._logged_in_client()
        post_resp = {"post_view": {}}
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _mock_response(200, post_resp)

        with patch.object(client._session, "post", side_effect=fake_post):
            client.post_to_community(42, "My Title", "My body")

        assert captured["json"]["name"] == "My Title"
        assert captured["json"]["body"] == "My body"
        assert captured["json"]["community_id"] == 42

    def test_optional_url_included_when_set(self):
        client = self._logged_in_client()
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _mock_response(200, {})

        with patch.object(client._session, "post", side_effect=fake_post):
            client.post_to_community(42, "Title", "Body", url="https://example.com")

        assert captured["json"]["url"] == "https://example.com"

    def test_url_absent_when_not_set(self):
        client = self._logged_in_client()
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _mock_response(200, {})

        with patch.object(client._session, "post", side_effect=fake_post):
            client.post_to_community(42, "Title", "Body")

        assert "url" not in captured["json"]

    def test_not_logged_in_raises_auth_error(self, client):
        with pytest.raises(LemmyAuthError):
            client.post_to_community(42, "Title", "Body")
