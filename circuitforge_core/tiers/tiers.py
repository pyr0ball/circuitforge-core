"""
Tier system for CircuitForge products.

Tiers: free < paid < premium < ultra
Products register their own FEATURES dict and pass it to can_use().

BYOK_UNLOCKABLE: features that unlock when the user has any configured
LLM backend (local or API key). These are gated only because CF would
otherwise provide the compute.

LOCAL_VISION_UNLOCKABLE: features that unlock when the user has a local
vision model configured (e.g. moondream2). Distinct from BYOK — a text
LLM key does NOT unlock vision features.
"""
from __future__ import annotations

TIERS: list[str] = ["free", "paid", "premium", "ultra"]

# Features that unlock when the user has any LLM backend configured.
# Each product extends this frozenset with its own BYOK-unlockable features.
BYOK_UNLOCKABLE: frozenset[str] = frozenset()

# Features that unlock when the user has a local vision model configured.
LOCAL_VISION_UNLOCKABLE: frozenset[str] = frozenset()


def can_use(
    feature: str,
    tier: str,
    has_byok: bool = False,
    has_local_vision: bool = False,
    _features: dict[str, str] | None = None,
    _byok_unlockable: frozenset[str] | None = None,
    _local_vision_unlockable: frozenset[str] | None = None,
) -> bool:
    """
    Return True if the given tier (and optional unlocks) can access feature.

    Args:
        feature:                Feature key string.
        tier:                   User's current tier ("free", "paid", "premium", "ultra").
        has_byok:               True if user has a configured LLM backend.
        has_local_vision:       True if user has a local vision model configured.
        _features:              Feature→min_tier map. Products pass their own dict here.
                                If None, all features are free.
        _byok_unlockable:       Product-specific BYOK-unlockable features.
                                If None, uses module-level BYOK_UNLOCKABLE.
        _local_vision_unlockable: Product-specific local vision unlockable features.
                                  If None, uses module-level LOCAL_VISION_UNLOCKABLE.
    """
    features = _features or {}
    byok_unlockable = _byok_unlockable if _byok_unlockable is not None else BYOK_UNLOCKABLE
    local_vision_unlockable = _local_vision_unlockable if _local_vision_unlockable is not None else LOCAL_VISION_UNLOCKABLE

    if feature not in features:
        return True

    if has_byok and feature in byok_unlockable:
        return True

    if has_local_vision and feature in local_vision_unlockable:
        return True

    min_tier = features[feature]
    try:
        return TIERS.index(tier) >= TIERS.index(min_tier)
    except ValueError:
        return False


def tier_label(
    feature: str,
    has_byok: bool = False,
    has_local_vision: bool = False,
    _features: dict[str, str] | None = None,
    _byok_unlockable: frozenset[str] | None = None,
    _local_vision_unlockable: frozenset[str] | None = None,
) -> str:
    """Return a human-readable label for the minimum tier needed for feature."""
    features = _features or {}
    byok_unlockable = _byok_unlockable if _byok_unlockable is not None else BYOK_UNLOCKABLE
    local_vision_unlockable = _local_vision_unlockable if _local_vision_unlockable is not None else LOCAL_VISION_UNLOCKABLE

    if feature not in features:
        return "free"
    if has_byok and feature in byok_unlockable:
        return "free (BYOK)"
    if has_local_vision and feature in local_vision_unlockable:
        return "free (local vision)"
    return features[feature]
