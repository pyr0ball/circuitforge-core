"""
License validation via Heimdall.

Products call validate_license() or get_license_tier() at startup to check
the CF_LICENSE_KEY environment variable against Heimdall.

Both functions are safe to call when CF_LICENSE_KEY is absent — they return
"free" tier gracefully rather than raising.

Environment variables:
    CF_LICENSE_KEY  — Raw license key (e.g. CFG-PRNG-XXXX-XXXX-XXXX).
                      If absent, product runs as free tier.
    CF_LICENSE_URL  — Heimdall base URL override.
                      Default: https://license.circuitforge.tech
"""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

_DEFAULT_HEIMDALL_URL = "https://license.circuitforge.tech"
_CACHE_TTL_SECONDS = 1800  # 30 minutes

# Cache: (key, product) -> (result_dict, expires_at)
_cache: dict[tuple[str, str], tuple[dict[str, bool | str], float]] = {}

_INVALID: dict[str, bool | str] = {"valid": False, "tier": "free", "user_id": ""}


def _heimdall_url(override: str | None) -> str:
    return override or os.environ.get("CF_LICENSE_URL", _DEFAULT_HEIMDALL_URL)


def validate_license(
    product: str,
    min_tier: str = "free",
    heimdall_url: str | None = None,
) -> dict[str, bool | str]:
    """
    Validate CF_LICENSE_KEY against Heimdall for the given product.

    Returns a dict with keys: valid (bool), tier (str), user_id (str).
    Returns {"valid": False, "tier": "free", "user_id": ""} when:
      - CF_LICENSE_KEY is not set
      - Heimdall is unreachable
      - The key is invalid/expired/revoked

    Results are cached for 30 minutes per (key, product) pair.
    """
    key = os.environ.get("CF_LICENSE_KEY", "").strip()
    if not key:
        return dict(_INVALID)

    cache_key = (key, product)
    now = time.monotonic()
    if cache_key in _cache:
        cached_result, expires_at = _cache[cache_key]
        if now < expires_at:
            return dict(cached_result)

    base = _heimdall_url(heimdall_url)
    try:
        resp = requests.post(
            f"{base}/licenses/verify",
            json={"key": key, "min_tier": min_tier},
            timeout=5,
        )
        if not resp.ok:
            logger.warning("[license] Heimdall returned %s for key validation", resp.status_code)
            result = dict(_INVALID)
        else:
            data = resp.json()
            result = {
                "valid": bool(data.get("valid", False)),
                "tier": data.get("tier", "free") or "free",
                "user_id": data.get("user_id", "") or "",
            }
    except Exception as exc:
        logger.warning("[license] License validation failed: %s", exc)
        result = dict(_INVALID)

    _cache[cache_key] = (result, now + _CACHE_TTL_SECONDS)
    return result


def get_license_tier(
    product: str,
    heimdall_url: str | None = None,
) -> str:
    """
    Return the active tier for CF_LICENSE_KEY, or "free" if absent/invalid.

    Convenience wrapper around validate_license() for the common case
    where only the tier string is needed.
    """
    result = validate_license(product, min_tier="free", heimdall_url=heimdall_url)
    if not result["valid"]:
        return "free"
    return result["tier"]
