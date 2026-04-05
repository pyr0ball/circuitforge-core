"""Public API for circuitforge_core.affiliates.

Usage::

    from circuitforge_core.affiliates import wrap_url, get_disclosure_text

    # Wrap a URL — env-var mode (no preferences, no opt-out)
    url = wrap_url("https://www.ebay.com/itm/123", retailer="ebay")

    # Wrap a URL — with preference injection (opt-out + BYOK)
    url = wrap_url(
        "https://www.ebay.com/itm/123",
        retailer="ebay",
        user_id="u123",
        get_preference=my_prefs_client.get,
    )

    # Frontend disclosure tooltip
    text = get_disclosure_text("ebay")

    # Register a product-specific program at startup
    register_program(AffiliateProgram(
        name="My Shop",
        retailer_key="myshop",
        env_var="MYSHOP_AFFILIATE_ID",
        build_url=lambda url, id_: f"{url}?ref={id_}",
    ))
"""
from .disclosure import BANNER_COPY, get_disclosure_text
from .programs import AffiliateProgram, get_program, register_program, registered_keys
from .router import wrap_url

__all__ = [
    "wrap_url",
    "get_disclosure_text",
    "BANNER_COPY",
    "AffiliateProgram",
    "register_program",
    "get_program",
    "registered_keys",
]
