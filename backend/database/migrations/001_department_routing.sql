-- PostgreSQL: additive, transactional, repeatable; never rewrite existing records.
BEGIN;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS predicted_department VARCHAR(100);
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS department_confidence DOUBLE PRECISION;
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS routing_method VARCHAR(100);
ALTER TABLE master_tickets ADD COLUMN IF NOT EXISTS routing_model_version VARCHAR(100);
CREATE TABLE IF NOT EXISTS civicpulse_schema_migrations (
    version VARCHAR(100) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO civicpulse_schema_migrations(version) VALUES ('001_department_routing')
ON CONFLICT (version) DO NOTHING;
COMMIT;
