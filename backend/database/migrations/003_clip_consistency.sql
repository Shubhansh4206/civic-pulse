-- CLIP image-text consistency columns (advisory only; never affects scoring or acceptance).
-- Additive only: existing rows, scores, status and counts remain unchanged.
BEGIN;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS clip_similarity DOUBLE PRECISION;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS clip_status VARCHAR(20);
CREATE TABLE IF NOT EXISTS civicpulse_schema_migrations (
    version VARCHAR(100) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO civicpulse_schema_migrations(version) VALUES ('003_clip_consistency')
ON CONFLICT (version) DO NOTHING;
COMMIT;
