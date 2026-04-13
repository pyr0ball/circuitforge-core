# tests/community/test_store.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from circuitforge_core.community.store import SharedStore
from circuitforge_core.community.models import CommunityPost


def make_post_row() -> dict:
    return {
        "id": 1,
        "slug": "kiwi-plan-test-pasta-week",
        "pseudonym": "PastaWitch",
        "post_type": "plan",
        "published": datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
        "title": "Pasta Week",
        "description": None,
        "photo_url": None,
        "slots": [{"day": 0, "meal_type": "dinner", "recipe_id": 1, "recipe_name": "Spaghetti"}],
        "recipe_id": None,
        "recipe_name": None,
        "level": None,
        "outcome_notes": None,
        "seasoning_score": 0.7,
        "richness_score": 0.6,
        "brightness_score": 0.3,
        "depth_score": 0.5,
        "aroma_score": 0.4,
        "structure_score": 0.8,
        "texture_profile": "chewy",
        "dietary_tags": ["vegetarian"],
        "allergen_flags": ["gluten"],
        "flavor_molecules": [1234],
        "fat_pct": 12.5,
        "protein_pct": 10.0,
        "moisture_pct": 55.0,
        "source_product": "kiwi",
    }


@pytest.fixture
def mock_db():
    db = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    db.getconn.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cur
    return db, conn, cur


def test_shared_store_get_post_by_slug(mock_db):
    db, conn, cur = mock_db
    cur.fetchone.return_value = make_post_row()
    cur.description = [(col,) for col in make_post_row().keys()]

    store = SharedStore(db)
    post = store.get_post_by_slug("kiwi-plan-test-pasta-week")

    assert post is not None
    assert isinstance(post, CommunityPost)
    assert post.slug == "kiwi-plan-test-pasta-week"
    assert post.pseudonym == "PastaWitch"


def test_shared_store_get_post_by_slug_not_found(mock_db):
    db, conn, cur = mock_db
    cur.fetchone.return_value = None

    store = SharedStore(db)
    post = store.get_post_by_slug("does-not-exist")
    assert post is None


def test_shared_store_list_posts_returns_list(mock_db):
    db, conn, cur = mock_db
    row = make_post_row()
    cur.fetchall.return_value = [row]
    cur.description = [(col,) for col in row.keys()]

    store = SharedStore(db)
    posts = store.list_posts(limit=10, offset=0)

    assert isinstance(posts, list)
    assert len(posts) == 1
    assert posts[0].slug == "kiwi-plan-test-pasta-week"


def test_shared_store_delete_post(mock_db):
    db, conn, cur = mock_db
    cur.rowcount = 1

    store = SharedStore(db)
    deleted = store.delete_post(slug="kiwi-plan-test-pasta-week", pseudonym="PastaWitch")
    assert deleted is True


def test_shared_store_delete_post_wrong_owner(mock_db):
    db, conn, cur = mock_db
    cur.rowcount = 0

    store = SharedStore(db)
    deleted = store.delete_post(slug="kiwi-plan-test-pasta-week", pseudonym="WrongUser")
    assert deleted is False


def test_shared_store_returns_connection_on_error(mock_db):
    db, conn, cur = mock_db
    cur.fetchone.side_effect = Exception("DB error")

    store = SharedStore(db)
    with pytest.raises(Exception, match="DB error"):
        store.get_post_by_slug("any-slug")

    # Connection must be returned to pool even on error
    db.putconn.assert_called_once_with(conn)
