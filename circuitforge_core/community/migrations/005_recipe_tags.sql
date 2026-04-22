-- 005_recipe_tags.sql
-- Community-contributed recipe subcategory tags.
--
-- Users can tag corpus recipes (from a product's local recipe dataset) with a
-- domain/category/subcategory from that product's browse taxonomy. Tags are
-- keyed by (recipe_source, recipe_ref) so a single table serves all CF products
-- that have a recipe corpus (currently: kiwi).
--
-- Acceptance threshold: upvotes >= 2 (submitter's implicit vote counts as 1,
-- so one additional voter is enough to publish). Browse counts caches merge
-- accepted tags into subcategory totals on each nightly refresh.

CREATE TABLE IF NOT EXISTS recipe_tags (
    id              BIGSERIAL PRIMARY KEY,
    recipe_source   TEXT NOT NULL CHECK (recipe_source IN ('corpus')),
    recipe_ref      TEXT NOT NULL,      -- corpus integer recipe ID stored as text
    domain          TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT,               -- NULL = category-level tag (no subcategory)
    pseudonym       TEXT NOT NULL,
    upvotes         INTEGER NOT NULL DEFAULT 1,  -- starts at 1 (submitter's own vote)
    source_product  TEXT NOT NULL DEFAULT 'kiwi',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one tag per (recipe, location, user) — prevents submitting the same tag twice
    UNIQUE (recipe_source, recipe_ref, domain, category, subcategory, pseudonym)
);

CREATE INDEX IF NOT EXISTS idx_recipe_tags_lookup
    ON recipe_tags (source_product, domain, category, subcategory)
    WHERE upvotes >= 2;

CREATE INDEX IF NOT EXISTS idx_recipe_tags_recipe
    ON recipe_tags (recipe_source, recipe_ref);

-- Tracks who voted on which tag to prevent double-voting.
-- The submitter's self-vote is inserted here at submission time.
CREATE TABLE IF NOT EXISTS recipe_tag_votes (
    tag_id      BIGINT NOT NULL REFERENCES recipe_tags(id) ON DELETE CASCADE,
    pseudonym   TEXT NOT NULL,
    voted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tag_id, pseudonym)
);
