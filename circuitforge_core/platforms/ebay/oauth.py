"""eBay OAuth Authorization Code flow — user-level token manager.

Implements the Authorization Code Grant for eBay's Trading API.
App-level client credentials (Browse API) are handled separately in
the product-level EbayTokenManager (snipe/app/platforms/ebay/auth.py).

Usage (Snipe):
    manager = EbayUserTokenManager(
        client_id=app_id,
        client_secret=cert_id,
        runame=runame,
        redirect_uri=redirect_uri,
        env="production",
    )

    # 1. Send user to eBay
    url = manager.get_authorization_url(state="csrf-token-here")
    redirect(url)

    # 2. Handle callback
    tokens = manager.exchange_code(code)   # returns EbayUserTokens
    # store tokens.access_token, tokens.refresh_token, tokens.expires_at

    # 3. Get a fresh access token for API calls
    access_token = manager.refresh(stored_refresh_token)
"""
from __future__ import annotations

import base64
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests

EBAY_AUTH_URLS = {
    "production": "https://auth.ebay.com/oauth2/authorize",
    "sandbox":    "https://auth.sandbox.ebay.com/oauth2/authorize",
}

EBAY_TOKEN_URLS = {
    "production": "https://api.ebay.com/identity/v1/oauth2/token",
    "sandbox":    "https://api.sandbox.ebay.com/identity/v1/oauth2/token",
}

# Scopes needed for Trading API GetUser (account age + category feedback).
# https://developer.ebay.com/api-docs/static/oauth-scopes.html
DEFAULT_SCOPES = [
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.account.readonly",
]


@dataclass
class EbayUserTokens:
    access_token: str
    refresh_token: str
    expires_at: float       # epoch seconds
    scopes: list[str]


class EbayUserTokenManager:
    """Manages eBay Authorization Code OAuth tokens for a single user.

    One instance per user session. Does NOT persist tokens — callers are
    responsible for storing/loading tokens via the DB migration
    013_ebay_user_tokens.sql.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        runame: str,
        redirect_uri: str,
        env: str = "production",
        scopes: Optional[list[str]] = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._runame = runame
        self._redirect_uri = redirect_uri
        self._auth_url = EBAY_AUTH_URLS[env]
        self._token_url = EBAY_TOKEN_URLS[env]
        self._scopes = scopes or DEFAULT_SCOPES

    # ── Authorization URL ──────────────────────────────────────────────────────

    def get_authorization_url(self, state: str = "") -> str:
        """Build the eBay OAuth authorization URL to redirect the user to.

        Args:
            state: CSRF token or opaque value passed through unchanged.

        Returns:
            Full URL string to redirect the user's browser to.
        """
        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._runame,   # eBay uses RuName, not the raw URI
            "scope": " ".join(self._scopes),
        }
        if state:
            params["state"] = state
        return f"{self._auth_url}?{urllib.parse.urlencode(params)}"

    # ── Code exchange ──────────────────────────────────────────────────────────

    def exchange_code(self, code: str) -> EbayUserTokens:
        """Exchange an authorization code for access + refresh tokens.

        Called from the OAuth callback endpoint after eBay redirects back.

        Raises:
            requests.HTTPError on non-2xx eBay response.
            KeyError if eBay response is missing expected fields.
        """
        resp = requests.post(
            self._token_url,
            headers={
                "Authorization": f"Basic {self._credentials_b64()}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._runame,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return self._parse_token_response(resp.json())

    # ── Token refresh ──────────────────────────────────────────────────────────

    def refresh(self, refresh_token: str) -> EbayUserTokens:
        """Exchange a refresh token for a new access token.

        eBay refresh tokens are valid for 18 months. Access tokens last 2h.
        Call this before making Trading API requests when the stored token
        is within 60 seconds of expiry.

        Raises:
            requests.HTTPError if the refresh token is expired or revoked.
        """
        resp = requests.post(
            self._token_url,
            headers={
                "Authorization": f"Basic {self._credentials_b64()}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": " ".join(self._scopes),
            },
            timeout=15,
        )
        resp.raise_for_status()
        # Refresh responses do NOT include a new refresh_token — the original stays valid
        data = resp.json()
        return EbayUserTokens(
            access_token=data["access_token"],
            refresh_token=refresh_token,    # unchanged
            expires_at=time.time() + data["expires_in"],
            scopes=data.get("scope", "").split(),
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _credentials_b64(self) -> str:
        raw = f"{self._client_id}:{self._client_secret}"
        return base64.b64encode(raw.encode()).decode()

    def _parse_token_response(self, data: dict) -> EbayUserTokens:
        return EbayUserTokens(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"],
            scopes=data.get("scope", "").split(),
        )
