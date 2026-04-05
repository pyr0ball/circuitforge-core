"""Affiliate disclosure copy constants.

Follows the plain-language disclosure design from the affiliate links design
doc. All copy is centralized here so products don't drift out of sync and
legal/copy review has a single file to audit.
"""
from __future__ import annotations

# Per-retailer tooltip copy (shown on hover/tap of affiliate link indicator)
_TOOLTIP: dict[str, str] = {
    "ebay": (
        "Affiliate link — CircuitForge earns a small commission if you purchase "
        "on eBay. No purchase data is shared with us. [Opt out in Settings]"
    ),
    "amazon": (
        "Affiliate link — CircuitForge earns a small commission if you purchase "
        "on Amazon. No purchase data is shared with us. [Opt out in Settings]"
    ),
}

_GENERIC_TOOLTIP = (
    "Affiliate link — CircuitForge may earn a small commission if you purchase. "
    "No purchase data is shared with us. [Opt out in Settings]"
)

# First-encounter banner copy (shown once, then preference saved)
BANNER_COPY: dict[str, str] = {
    "title": "A note on purchase links",
    "body": (
        "Some links in this product go to retailers using our affiliate code. "
        "When you click one, the retailer knows you came from CircuitForge. "
        "We don't see or store what you buy. The retailer may track your "
        "purchase — that's between you and them.\n\n"
        "If you'd rather use plain links with no tracking code, you can opt "
        "out in Settings."
    ),
    "dismiss_label": "Got it",
    "opt_out_label": "Opt out now",
    "learn_more_label": "Learn more",
}


def get_disclosure_text(retailer: str) -> str:
    """Return the tooltip disclosure string for *retailer*.

    Falls back to a generic string for unregistered retailers so callers
    never receive an empty string.
    """
    return _TOOLTIP.get(retailer, _GENERIC_TOOLTIP)
