-- 004_community_categories.sql
-- MIT License
-- Shared eBay category tree published by credentialed Snipe instances.
-- Credentialless instances pull from this table during refresh().
-- Privacy: only public eBay category metadata (IDs, names, paths) — no user data.

CREATE TABLE IF NOT EXISTS community_categories (
    id              SERIAL PRIMARY KEY,
    platform        TEXT NOT NULL DEFAULT 'ebay',
    category_id     TEXT NOT NULL,
    name            TEXT NOT NULL,
    full_path       TEXT NOT NULL,
    source_product  TEXT NOT NULL DEFAULT 'snipe',
    published_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (platform, category_id)
);

CREATE INDEX IF NOT EXISTS idx_community_cat_name
    ON community_categories (platform, name);
