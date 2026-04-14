-- Seller trust signals: confirmed scammer / confirmed legitimate outcomes from Snipe.
-- Separate table from community_posts (Kiwi-specific) — seller signals are a
-- structurally different domain and should not overload the recipe post schema.
-- Applies to: cf_community PostgreSQL database (hosted by cf-orch).
-- BSL boundary: table schema is MIT; signal ingestion route in cf-orch is BSL 1.1.

CREATE TABLE IF NOT EXISTS seller_trust_signals (
    id              BIGSERIAL PRIMARY KEY,
    platform        TEXT    NOT NULL DEFAULT 'ebay',
    platform_seller_id TEXT NOT NULL,
    confirmed_scam  BOOLEAN NOT NULL,
    signal_source   TEXT    NOT NULL,  -- 'blocklist_add' | 'community_vote' | 'resolved'
    flags           JSONB   NOT NULL DEFAULT '[]',  -- red flag keys at time of signal
    source_product  TEXT    NOT NULL DEFAULT 'snipe',
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- No PII: platform_seller_id is the public eBay username or platform ID only.
CREATE INDEX IF NOT EXISTS idx_seller_trust_platform_id
    ON seller_trust_signals (platform, platform_seller_id);

CREATE INDEX IF NOT EXISTS idx_seller_trust_confirmed
    ON seller_trust_signals (confirmed_scam);

CREATE INDEX IF NOT EXISTS idx_seller_trust_recorded
    ON seller_trust_signals (recorded_at DESC);
