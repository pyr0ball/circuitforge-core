-- 006_community_dedup.sql
-- Adds variation-linking and title search support for community recipe dedup.
-- Applies to: cf_community PostgreSQL database.
-- BSL boundary: MIT (data layer, no inference).

-- Nullable self-referential FK: user-declared "this is a variation of X"
ALTER TABLE community_posts
    ADD COLUMN IF NOT EXISTS similar_to_ref TEXT REFERENCES community_posts(slug) ON DELETE SET NULL;

-- Index for variation lookup (find all variations of a parent post)
CREATE INDEX IF NOT EXISTS idx_community_posts_similar_ref
    ON community_posts (similar_to_ref)
    WHERE similar_to_ref IS NOT NULL;

-- Index to speed up title ILIKE prefix and substring searches
CREATE INDEX IF NOT EXISTS idx_community_posts_title_lower
    ON community_posts (lower(title));

-- Index on recipe_id for exact-recipe duplicate detection
CREATE INDEX IF NOT EXISTS idx_community_posts_recipe_id
    ON community_posts (recipe_id)
    WHERE recipe_id IS NOT NULL;
