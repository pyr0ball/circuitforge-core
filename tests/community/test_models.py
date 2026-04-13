# tests/community/test_models.py
import pytest
from datetime import datetime, timezone
from circuitforge_core.community.models import CommunityPost


def make_post(**kwargs) -> CommunityPost:
    defaults = dict(
        slug="kiwi-plan-test-2026-04-12-pasta-week",
        pseudonym="PastaWitch",
        post_type="plan",
        published=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
        title="Pasta Week",
        description="Seven days of carbs",
        photo_url=None,
        slots=[{"day": 0, "meal_type": "dinner", "recipe_id": 1, "recipe_name": "Spaghetti"}],
        recipe_id=None,
        recipe_name=None,
        level=None,
        outcome_notes=None,
        seasoning_score=0.7,
        richness_score=0.6,
        brightness_score=0.3,
        depth_score=0.5,
        aroma_score=0.4,
        structure_score=0.8,
        texture_profile="chewy",
        dietary_tags=["vegetarian"],
        allergen_flags=["gluten"],
        flavor_molecules=[1234, 5678],
        fat_pct=12.5,
        protein_pct=10.0,
        moisture_pct=55.0,
    )
    defaults.update(kwargs)
    return CommunityPost(**defaults)


def test_community_post_immutable():
    post = make_post()
    with pytest.raises((AttributeError, TypeError)):
        post.title = "changed"  # type: ignore


def test_community_post_slug_uri_compatible():
    post = make_post(slug="kiwi-plan-test-2026-04-12-pasta-week")
    assert " " not in post.slug
    assert post.slug == post.slug.lower()


def test_community_post_type_valid():
    make_post(post_type="plan")
    make_post(post_type="recipe_success")
    make_post(post_type="recipe_blooper")


def test_community_post_type_invalid():
    with pytest.raises(ValueError):
        make_post(post_type="garbage")


def test_community_post_scores_range():
    post = make_post(seasoning_score=1.0, richness_score=0.0)
    assert 0.0 <= post.seasoning_score <= 1.0
    assert 0.0 <= post.richness_score <= 1.0


def test_community_post_scores_out_of_range():
    with pytest.raises(ValueError):
        make_post(seasoning_score=1.5)
    with pytest.raises(ValueError):
        make_post(richness_score=-0.1)


def test_community_post_dietary_tags_immutable():
    post = make_post(dietary_tags=["vegan"])
    assert isinstance(post.dietary_tags, tuple)


def test_community_post_allergen_flags_immutable():
    post = make_post(allergen_flags=["nuts", "dairy"])
    assert isinstance(post.allergen_flags, tuple)


def test_community_post_flavor_molecules_immutable():
    post = make_post(flavor_molecules=[1, 2, 3])
    assert isinstance(post.flavor_molecules, tuple)


def test_community_post_optional_fields_none():
    post = make_post(photo_url=None, recipe_id=None, fat_pct=None)
    assert post.photo_url is None
    assert post.recipe_id is None
    assert post.fat_pct is None
