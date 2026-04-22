# circuitforge_core/community/store.py
# MIT License

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import CommunityPost

if TYPE_CHECKING:
    from .db import CommunityDB

logger = logging.getLogger(__name__)


def _row_to_post(row: dict) -> CommunityPost:
    """Convert a psycopg2 row dict to a CommunityPost.

    JSONB columns (slots, dietary_tags, allergen_flags, flavor_molecules) come
    back from psycopg2 as Python lists already — no json.loads() needed.
    """
    return CommunityPost(
        slug=row["slug"],
        pseudonym=row["pseudonym"],
        post_type=row["post_type"],
        published=row["published"],
        title=row["title"],
        description=row.get("description"),
        photo_url=row.get("photo_url"),
        slots=row.get("slots") or [],
        recipe_id=row.get("recipe_id"),
        recipe_name=row.get("recipe_name"),
        level=row.get("level"),
        outcome_notes=row.get("outcome_notes"),
        seasoning_score=row["seasoning_score"] or 0.0,
        richness_score=row["richness_score"] or 0.0,
        brightness_score=row["brightness_score"] or 0.0,
        depth_score=row["depth_score"] or 0.0,
        aroma_score=row["aroma_score"] or 0.0,
        structure_score=row["structure_score"] or 0.0,
        texture_profile=row.get("texture_profile") or "",
        dietary_tags=row.get("dietary_tags") or [],
        allergen_flags=row.get("allergen_flags") or [],
        flavor_molecules=row.get("flavor_molecules") or [],
        fat_pct=row.get("fat_pct"),
        protein_pct=row.get("protein_pct"),
        moisture_pct=row.get("moisture_pct"),
    )


def _cursor_to_dict(cur, row) -> dict:
    """Convert a psycopg2 row tuple to a dict using cursor.description."""
    if isinstance(row, dict):
        return row
    return {desc[0]: val for desc, val in zip(cur.description, row)}


class SharedStore:
    """Base class for product community stores.

    Subclass this in each product:
        class KiwiCommunityStore(SharedStore):
            def list_posts_for_week(self, week_start: str) -> list[CommunityPost]: ...

    All methods return new objects (immutable pattern). Never mutate rows in-place.
    """

    def __init__(self, db: "CommunityDB", source_product: str = "kiwi") -> None:
        self._db = db
        self._source_product = source_product

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_post_by_slug(self, slug: str) -> CommunityPost | None:
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM community_posts WHERE slug = %s LIMIT 1",
                    (slug,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return _row_to_post(_cursor_to_dict(cur, row))
        finally:
            self._db.putconn(conn)

    def list_posts(
        self,
        limit: int = 20,
        offset: int = 0,
        post_type: str | None = None,
        dietary_tags: list[str] | None = None,
        allergen_exclude: list[str] | None = None,
        source_product: str | None = None,
    ) -> list[CommunityPost]:
        """Paginated post list with optional filters.

        dietary_tags: JSONB containment — posts must include ALL listed tags.
        allergen_exclude: JSONB overlap exclusion — posts must NOT include any listed flag.
        """
        conn = self._db.getconn()
        try:
            conditions = []
            params: list = []

            if post_type:
                conditions.append("post_type = %s")
                params.append(post_type)
            if dietary_tags:
                import json
                conditions.append("dietary_tags @> %s::jsonb")
                params.append(json.dumps(dietary_tags))
            if allergen_exclude:
                import json
                conditions.append("NOT (allergen_flags && %s::jsonb)")
                params.append(json.dumps(allergen_exclude))
            if source_product:
                conditions.append("source_product = %s")
                params.append(source_product)

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.extend([limit, offset])

            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM community_posts {where} "
                    "ORDER BY published DESC LIMIT %s OFFSET %s",
                    params,
                )
                rows = cur.fetchall()
                return [_row_to_post(_cursor_to_dict(cur, r)) for r in rows]
        finally:
            self._db.putconn(conn)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_post(self, post: CommunityPost) -> CommunityPost:
        """Insert a new community post. Returns the inserted post (unchanged — slug is the key)."""
        import json

        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO community_posts (
                        slug, pseudonym, post_type, published, title, description, photo_url,
                        slots, recipe_id, recipe_name, level, outcome_notes,
                        seasoning_score, richness_score, brightness_score,
                        depth_score, aroma_score, structure_score, texture_profile,
                        dietary_tags, allergen_flags, flavor_molecules,
                        fat_pct, protein_pct, moisture_pct, source_product
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb, %s::jsonb,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        post.slug, post.pseudonym, post.post_type,
                        post.published, post.title, post.description, post.photo_url,
                        json.dumps(list(post.slots)),
                        post.recipe_id, post.recipe_name, post.level, post.outcome_notes,
                        post.seasoning_score, post.richness_score, post.brightness_score,
                        post.depth_score, post.aroma_score, post.structure_score,
                        post.texture_profile,
                        json.dumps(list(post.dietary_tags)),
                        json.dumps(list(post.allergen_flags)),
                        json.dumps(list(post.flavor_molecules)),
                        post.fat_pct, post.protein_pct, post.moisture_pct,
                        self._source_product,
                    ),
                )
                conn.commit()
            return post
        except Exception:
            conn.rollback()
            raise
        finally:
            self._db.putconn(conn)

    def delete_post(self, slug: str, pseudonym: str) -> bool:
        """Hard-delete a post. Only succeeds if pseudonym matches the author.

        Returns True if a row was deleted, False if no matching row found.
        """
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM community_posts WHERE slug = %s AND pseudonym = %s",
                    (slug, pseudonym),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            conn.rollback()
            raise
        finally:
            self._db.putconn(conn)

    # ── Recipe tags ───────────────────────────────────────────────────────────

    def submit_recipe_tag(
        self,
        recipe_id: int,
        domain: str,
        category: str,
        subcategory: str | None,
        pseudonym: str,
        source_product: str = "kiwi",
    ) -> dict:
        """Submit a new subcategory tag for a corpus recipe.

        Inserts the tag with upvotes=1 and records the submitter's self-vote in
        recipe_tag_votes. Returns the created tag row as a dict.

        Raises psycopg2.errors.UniqueViolation if the same user has already
        tagged this recipe to this location — let the caller handle it.
        """
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recipe_tags
                        (recipe_source, recipe_ref, domain, category, subcategory,
                         pseudonym, upvotes, source_product)
                    VALUES ('corpus', %s, %s, %s, %s, %s, 1, %s)
                    RETURNING id, recipe_ref, domain, category, subcategory,
                              pseudonym, upvotes, created_at
                    """,
                    (str(recipe_id), domain, category, subcategory,
                     pseudonym, source_product),
                )
                row = dict(zip([d[0] for d in cur.description], cur.fetchone()))
                # Record submitter's self-vote
                cur.execute(
                    "INSERT INTO recipe_tag_votes (tag_id, pseudonym) VALUES (%s, %s)",
                    (row["id"], pseudonym),
                )
                conn.commit()
                return row
        except Exception:
            conn.rollback()
            raise
        finally:
            self._db.putconn(conn)

    def upvote_recipe_tag(self, tag_id: int, pseudonym: str) -> int:
        """Add an upvote to a tag from pseudonym. Returns new upvote count.

        Raises psycopg2.errors.UniqueViolation if this pseudonym already voted.
        Raises ValueError if the tag does not exist.
        """
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO recipe_tag_votes (tag_id, pseudonym) VALUES (%s, %s)",
                    (tag_id, pseudonym),
                )
                cur.execute(
                    "UPDATE recipe_tags SET upvotes = upvotes + 1 WHERE id = %s"
                    " RETURNING upvotes",
                    (tag_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"recipe_tag {tag_id} not found")
                conn.commit()
                return row[0]
        except Exception:
            conn.rollback()
            raise
        finally:
            self._db.putconn(conn)

    def get_recipe_tag_by_id(self, tag_id: int) -> dict | None:
        """Return a single recipe_tag row by ID, or None if not found."""
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, recipe_ref, domain, category, subcategory,
                           pseudonym, upvotes, created_at
                    FROM recipe_tags WHERE id = %s
                    """,
                    (tag_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return dict(zip([d[0] for d in cur.description], row))
        finally:
            self._db.putconn(conn)

    def list_tags_for_recipe(
        self,
        recipe_id: int,
        source_product: str = "kiwi",
    ) -> list[dict]:
        """Return all tags for a corpus recipe, accepted or not, newest first."""
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, domain, category, subcategory, pseudonym,
                           upvotes, created_at
                    FROM recipe_tags
                    WHERE recipe_source = 'corpus'
                      AND recipe_ref = %s
                      AND source_product = %s
                    ORDER BY upvotes DESC, created_at DESC
                    """,
                    (str(recipe_id), source_product),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            self._db.putconn(conn)

    def get_accepted_recipe_ids_for_subcategory(
        self,
        domain: str,
        category: str,
        subcategory: str | None,
        source_product: str = "kiwi",
        threshold: int = 2,
    ) -> list[int]:
        """Return corpus recipe IDs with accepted community tags for a subcategory.

        Used by browse_counts_cache refresh and browse_recipes() FTS fallback.
        Only includes tags that have reached the acceptance threshold.
        """
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                if subcategory is None:
                    cur.execute(
                        """
                        SELECT DISTINCT recipe_ref::INTEGER
                        FROM recipe_tags
                        WHERE source_product = %s
                          AND domain = %s AND category = %s
                          AND subcategory IS NULL
                          AND upvotes >= %s
                        """,
                        (source_product, domain, category, threshold),
                    )
                else:
                    cur.execute(
                        """
                        SELECT DISTINCT recipe_ref::INTEGER
                        FROM recipe_tags
                        WHERE source_product = %s
                          AND domain = %s AND category = %s
                          AND subcategory = %s
                          AND upvotes >= %s
                        """,
                        (source_product, domain, category, subcategory, threshold),
                    )
                return [r[0] for r in cur.fetchall()]
        finally:
            self._db.putconn(conn)
