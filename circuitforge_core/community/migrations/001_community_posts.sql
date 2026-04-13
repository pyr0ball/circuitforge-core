-- 001_community_posts.sql
-- Community posts table: published meal plans, recipe successes, and bloopers.
-- Applies to: cf_community PostgreSQL database (hosted by cf-orch).
-- BSL boundary: this schema is MIT (data layer, no inference).

CREATE TABLE IF NOT EXISTS community_posts (
    id              BIGSERIAL PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    pseudonym       TEXT NOT NULL,
    post_type       TEXT NOT NULL CHECK (post_type IN ('plan', 'recipe_success', 'recipe_blooper')),
    published       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title           TEXT NOT NULL,
    description     TEXT,
    photo_url       TEXT,

    -- Plan slots (JSON array: [{day, meal_type, recipe_id, recipe_name}])
    slots           JSONB NOT NULL DEFAULT '[]',

    -- Recipe result fields
    recipe_id       BIGINT,
    recipe_name     TEXT,
    level           SMALLINT CHECK (level IS NULL OR level BETWEEN 1 AND 4),
    outcome_notes   TEXT,

    -- Element snapshot (denormalized from corpus at publish time)
    seasoning_score REAL,
    richness_score  REAL,
    brightness_score REAL,
    depth_score     REAL,
    aroma_score     REAL,
    structure_score REAL,
    texture_profile TEXT,

    -- Dietary / allergen / flavor
    dietary_tags    JSONB NOT NULL DEFAULT '[]',
    allergen_flags  JSONB NOT NULL DEFAULT '[]',
    flavor_molecules JSONB NOT NULL DEFAULT '[]',

    -- USDA FDC macros
    fat_pct         REAL,
    protein_pct     REAL,
    moisture_pct    REAL,

    -- Source product identifier
    source_product  TEXT NOT NULL DEFAULT 'kiwi'
);

-- Indexes for common filter patterns
CREATE INDEX IF NOT EXISTS idx_community_posts_published ON community_posts (published DESC);
CREATE INDEX IF NOT EXISTS idx_community_posts_post_type ON community_posts (post_type);
CREATE INDEX IF NOT EXISTS idx_community_posts_source ON community_posts (source_product);

-- GIN index for dietary/allergen JSONB array containment queries
CREATE INDEX IF NOT EXISTS idx_community_posts_dietary_tags ON community_posts USING GIN (dietary_tags);
CREATE INDEX IF NOT EXISTS idx_community_posts_allergen_flags ON community_posts USING GIN (allergen_flags);
