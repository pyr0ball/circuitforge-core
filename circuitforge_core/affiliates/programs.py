"""Affiliate program definitions and URL builders.

Each ``AffiliateProgram`` knows how to append its affiliate parameters to a
plain product URL.  Built-in programs (eBay EPN, Amazon Associates) are
registered at module import time.  Products can register additional programs
with ``register_program()``.

Affiliate IDs are read from environment variables at call time so they pick
up values set after process startup (useful in tests).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


@dataclass(frozen=True)
class AffiliateProgram:
    """One affiliate program and its URL building logic.

    Attributes:
        name:         Human-readable program name.
        retailer_key: Matches the ``retailer=`` argument in ``wrap_url()``.
        env_var:      Environment variable holding CF's affiliate ID.
        build_url:    ``(plain_url, affiliate_id) -> affiliate_url`` callable.
    """

    name: str
    retailer_key: str
    env_var: str
    build_url: Callable[[str, str], str]

    def cf_affiliate_id(self) -> str | None:
        """Return CF's configured affiliate ID, or None if the env var is unset/blank."""
        val = os.environ.get(self.env_var, "").strip()
        return val or None


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------

def _build_ebay_url(url: str, affiliate_id: str) -> str:
    """Append eBay Partner Network parameters to a listing URL."""
    sep = "&" if "?" in url else "?"
    params = urlencode({
        "mkcid": "1",
        "mkrid": "711-53200-19255-0",
        "siteid": "0",
        "campid": affiliate_id,
        "toolid": "10001",
        "mkevt": "1",
    })
    return f"{url}{sep}{params}"


def _build_instacart_url(url: str, affiliate_id: str) -> str:
    """Append Instacart affiliate parameter to a search URL."""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}aff={affiliate_id}"


def _build_amazon_url(url: str, affiliate_id: str) -> str:
    """Merge an Amazon Associates tag into a product URL's query string."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["tag"] = [affiliate_id]
    new_query = urlencode({k: v[0] for k, v in qs.items()})
    return urlunparse(parsed._replace(query=new_query))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, AffiliateProgram] = {}


def register_program(program: AffiliateProgram) -> None:
    """Register an affiliate program (overwrites any existing entry for the same key)."""
    _REGISTRY[program.retailer_key] = program


def get_program(retailer_key: str) -> AffiliateProgram | None:
    """Return the registered program for *retailer_key*, or None."""
    return _REGISTRY.get(retailer_key)


def registered_keys() -> list[str]:
    """Return all currently registered retailer keys."""
    return list(_REGISTRY.keys())


# Register built-ins
register_program(AffiliateProgram(
    name="eBay Partner Network",
    retailer_key="ebay",
    env_var="EBAY_AFFILIATE_CAMPAIGN_ID",
    build_url=_build_ebay_url,
))

register_program(AffiliateProgram(
    name="Amazon Associates",
    retailer_key="amazon",
    env_var="AMAZON_ASSOCIATES_TAG",
    build_url=_build_amazon_url,
))

register_program(AffiliateProgram(
    name="Instacart",
    retailer_key="instacart",
    env_var="INSTACART_AFFILIATE_ID",
    build_url=_build_instacart_url,
))
