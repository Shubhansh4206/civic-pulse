-- Additive only: original evidence, scores, status and legacy counts remain intact.
BEGIN;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS duplicate_group_id INTEGER;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS duplicate_similarity DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS ix_master_tickets_duplicate_group_id ON master_tickets (duplicate_group_id);
CREATE TABLE IF NOT EXISTS civicpulse_schema_migrations (
    version VARCHAR(100) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO civicpulse_schema_migrations(version) VALUES ('002_semantic_duplicates')
ON CONFLICT (version) DO NOTHING;
COMMIT;
