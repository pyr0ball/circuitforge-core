import pytest
from circuitforge_core.tiers import can_use, TIERS, BYOK_UNLOCKABLE, LOCAL_VISION_UNLOCKABLE


def test_tiers_order():
    assert TIERS == ["free", "paid", "premium", "ultra"]


def test_free_feature_always_accessible():
    # Features not in FEATURES dict are free for everyone
    assert can_use("nonexistent_feature", tier="free") is True


def test_paid_feature_blocked_for_free_tier():
    # Caller must register features — test via can_use with explicit min_tier
    assert can_use("test_paid", tier="free", _features={"test_paid": "paid"}) is False


def test_paid_feature_accessible_for_paid_tier():
    assert can_use("test_paid", tier="paid", _features={"test_paid": "paid"}) is True


def test_premium_feature_accessible_for_ultra_tier():
    assert can_use("test_premium", tier="ultra", _features={"test_premium": "premium"}) is True


def test_byok_unlocks_byok_feature():
    byok_feature = next(iter(BYOK_UNLOCKABLE)) if BYOK_UNLOCKABLE else None
    if byok_feature:
        assert can_use(byok_feature, tier="free", has_byok=True) is True


def test_byok_does_not_unlock_non_byok_feature():
    assert can_use("test_paid", tier="free", has_byok=True,
                   _features={"test_paid": "paid"}) is False


def test_local_vision_unlocks_vision_feature():
    vision_feature = next(iter(LOCAL_VISION_UNLOCKABLE)) if LOCAL_VISION_UNLOCKABLE else None
    if vision_feature:
        assert can_use(vision_feature, tier="free", has_local_vision=True) is True


def test_local_vision_does_not_unlock_non_vision_feature():
    assert can_use("test_paid", tier="free", has_local_vision=True,
                   _features={"test_paid": "paid"}) is False
