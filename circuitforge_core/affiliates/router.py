"""Affiliate URL wrapping — resolution logic.

Resolution order (from affiliate links design doc):

  1. User opted out?                          → return plain URL
  2. User has BYOK ID for this retailer?      → wrap with user's ID
  3. CF has a program with env var set?       → wrap with CF's ID
  4. No program / no ID configured            → return plain URL

The ``get_preference`` callable is optional.  When None (default), steps 1
and 2 are skipped — the module operates in env-var-only mode.  Products
inject their preferences client to enable opt-out and BYOK.

Signature of ``get_preference``::

    def get_preference(user_id: str | None, path: str, default=None) -> Any: ...
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from .programs import get_program

logger = logging.getLogger(__name__)

GetPreferenceFn = Callable[[str | None, str, Any], Any]


def wrap_url(
    url: str,
    retailer: str,
    user_id: str | None = None,
    get_preference: GetPreferenceFn | None = None,
) -> str:
    """Return an affiliate URL for *url*, or the plain URL if no affiliate
    link can be or should be generated.

    Args:
        url:            Plain product URL to wrap.
        retailer:       Retailer key (e.g. ``"ebay"``, ``"amazon"``).
        user_id:        User identifier for preference lookups. None = anonymous.
        get_preference: Optional callable ``(user_id, path, default) -> value``.
                        Injected by products to enable opt-out and BYOK resolution.
                        When None, opt-out and BYOK checks are skipped.

    Returns:
        Affiliate URL, or *url* unchanged if:
        - The user has opted out
        - No program is registered for *retailer*
        - No affiliate ID is configured (env var unset and no BYOK)
    """
    program = get_program(retailer)
    if program is None:
        logger.debug("affiliates: no program registered for retailer=%r", retailer)
        return url

    # Step 1: opt-out check
    if get_preference is not None:
        opted_out = get_preference(user_id, "affiliate.opt_out", False)
        if opted_out:
            logger.debug("affiliates: user %r opted out — returning plain URL", user_id)
            return url

    # Step 2: BYOK — user's own affiliate ID (Premium)
    if get_preference is not None and user_id is not None:
        byok_id = get_preference(user_id, f"affiliate.byok_ids.{retailer}", None)
        if byok_id:
            logger.debug(
                "affiliates: using BYOK id for user=%r retailer=%r", user_id, retailer
            )
            return program.build_url(url, byok_id)

    # Step 3: CF's affiliate ID from env var
    cf_id = program.cf_affiliate_id()
    if cf_id:
        return program.build_url(url, cf_id)

    logger.debug(
        "affiliates: no affiliate ID configured for retailer=%r (env var %r unset)",
        retailer, program.env_var,
    )
    return url
