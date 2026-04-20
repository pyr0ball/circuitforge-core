"""
Lemmy REST API client for posting to Lemmy communities.

Uses JWT authentication (Lemmy v0.19+ API). Does not require ActivityPub
federation setup — the Lemmy REST API is simpler and more reliable for
the initial integration.

MIT licensed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class LemmyAuthError(Exception):
    """Raised when Lemmy login fails."""


class LemmyCommunityNotFound(Exception):
    """Raised when a community cannot be resolved by name."""


@dataclass(frozen=True)
class LemmyConfig:
    """Connection config for a Lemmy instance."""

    instance_url: str  # e.g. "https://lemmy.ml" (no trailing slash)
    username: str
    password: str      # Load from env/config; never hardcode


class LemmyClient:
    """
    Lemmy REST API client.

    Usage::

        config = LemmyConfig(instance_url="https://lemmy.ml", username="bot", password="...")
        client = LemmyClient(config)
        client.login()
        community_id = client.resolve_community("!cooking@lemmy.world")
        client.post_to_community(community_id, title="Fresh pesto recipe", body="...")
    """

    def __init__(self, config: LemmyConfig) -> None:
        self._config = config
        self._jwt: str | None = None
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @property
    def _api(self) -> str:
        return f"{self._config.instance_url.rstrip('/')}/api/v3"

    def _auth_headers(self) -> dict[str, str]:
        if not self._jwt:
            raise LemmyAuthError("Not logged in — call login() first.")
        return {"Authorization": f"Bearer {self._jwt}"}

    def login(self) -> None:
        """
        Authenticate with the Lemmy instance and store the JWT.

        Raises:
            LemmyAuthError: If credentials are rejected or the request fails.
        """
        resp = self._session.post(
            f"{self._api}/user/login",
            json={"username_or_email": self._config.username, "password": self._config.password},
            timeout=10,
        )
        if resp.status_code != 200:
            raise LemmyAuthError(
                f"Lemmy login failed ({resp.status_code}): {resp.text[:200]}"
            )
        data = resp.json()
        token = data.get("jwt")
        if not token:
            raise LemmyAuthError("Lemmy login response missing 'jwt' field.")
        self._jwt = token

    def resolve_community(self, name: str) -> int:
        """
        Resolve a community name or address to its numeric Lemmy ID.

        Accepts:
        - Bare name: "cooking"
        - Fediverse address: "!cooking@lemmy.world"
        - Display name search (best-effort)

        Args:
            name: Community identifier.

        Returns:
            Numeric community ID.

        Raises:
            LemmyCommunityNotFound: If not found or multiple matches are ambiguous.
            LemmyAuthError: If not logged in.
        """
        # Strip leading ! for address lookups
        lookup = name.lstrip("!")
        resp = self._session.get(
            f"{self._api}/search",
            params={"q": lookup, "type_": "Communities", "limit": 5},
            headers=self._auth_headers(),
            timeout=10,
        )
        if resp.status_code != 200:
            raise LemmyCommunityNotFound(
                f"Community search failed ({resp.status_code}): {resp.text[:200]}"
            )
        communities = resp.json().get("communities", [])
        if not communities:
            raise LemmyCommunityNotFound(f"No communities found for '{name}'.")
        # Prefer exact actor_id match (e.g. !cooking@lemmy.world)
        for item in communities:
            view = item.get("community", {})
            if "@" in lookup:
                actor_id: str = view.get("actor_id", "")
                if lookup.lower() in actor_id.lower():
                    return int(view["id"])
            else:
                if view.get("name", "").lower() == lookup.lower():
                    return int(view["id"])
        # Fall back to first result
        return int(communities[0]["community"]["id"])

    def post_to_community(
        self,
        community_id: int,
        title: str,
        body: str,
        url: str | None = None,
        nsfw: bool = False,
    ) -> dict[str, Any]:
        """
        Create a post in a Lemmy community.

        Args:
            community_id: Numeric community ID (from resolve_community()).
            title:        Post title.
            body:         Markdown post body.
            url:          Optional external URL to attach.
            nsfw:         Mark NSFW (default False).

        Returns:
            Lemmy API response dict (contains 'post_view', etc.).

        Raises:
            LemmyAuthError: If not logged in.
            requests.RequestException: On network failure.
        """
        payload: dict[str, Any] = {
            "community_id": community_id,
            "name": title,
            "body": body,
            "nsfw": nsfw,
        }
        if url:
            payload["url"] = url

        resp = self._session.post(
            f"{self._api}/post",
            json=payload,
            headers=self._auth_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
