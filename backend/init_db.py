"""
Database initialization script for CivicPulse.
Connects to PostgreSQL, activates PostGIS spatial extension if available,
and creates all relational schemas and spatial tables defined in MasterTicket.
"""

import os
import sys
import logging

# Ensure project root is present in sys.path so script runs smoothly
# with both `python backend/init_db.py` and `python -m backend.init_db`
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import text
from backend.config import settings
from backend.database.session import engine, Base
from backend.database.models import MasterTicket, USE_POSTGIS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.init_db")


def init_database() -> None:
    """
    Initialize the database:
      1. Verify database connectivity.
      2. Check and enable PostGIS spatial extension (if installed on PostgreSQL host).
      3. Create all tables registered with Base.metadata (MasterTicket, etc.).
      4. Validate column schemas and PostGIS upgrade path.
    """
    logger.info("Initializing CivicPulse database...")
    logger.info("Target connection URL: %s", engine.url)

    try:
        with engine.connect() as connection:
            if engine.dialect.name == "postgresql":
                # Check if postgis extension files are installed in PostgreSQL share/extension
                postgis_available = connection.execute(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'postgis';")
                ).scalar()

                if postgis_available:
                    logger.info("Enabling PostGIS extension...")
                    connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                    connection.commit()
                    logger.info("PostGIS extension confirmed.")
                else:
                    logger.warning(
                        "PostGIS extension is not installed in PostgreSQL 16.\n"
                        "-> Initializing database with coordinate fallback.\n"
                        "-> To enable native PostGIS indexing later, run Stack Builder "
                        "('C:\\Program Files\\PostgreSQL\\16\\bin\\stackbuilder.exe') "
                        "and install 'PostGIS 3.4 Bundle'."
                    )

        # Create all ORM tables
        logger.info("Creating database tables from ORM metadata...")
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully registered tables: %s", list(Base.metadata.tables.keys()))

        # Post-creation column and migration checks
        with engine.connect() as connection:
            if engine.dialect.name == "postgresql":
                # 1. Ensure report_count column exists
                connection.execute(text("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'master_tickets') THEN
                            IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'master_tickets' AND column_name = 'report_count') THEN
                                ALTER TABLE master_tickets ADD COLUMN report_count INTEGER NOT NULL DEFAULT 1;
                            END IF;
                        END IF;
                    END $$;
                """))
                connection.commit()

                # 2. If PostGIS was enabled and location was previously VARCHAR, upgrade to geometry
                connection.execute(text("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name = 'master_tickets' AND column_name = 'location' AND data_type = 'character varying'
                            ) THEN
                                ALTER TABLE master_tickets ALTER COLUMN location TYPE geometry(POINT, 4326) USING location::geometry;
                            END IF;
                        END IF;
                    END $$;
                """))
                connection.commit()

                # 3. Ensure semantic duplicate columns exist (002_semantic_duplicates migration)
                connection.execute(text("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'master_tickets' AND column_name = 'duplicate_group_id'
                        ) THEN
                            ALTER TABLE master_tickets ADD COLUMN duplicate_group_id INTEGER;
                            CREATE INDEX IF NOT EXISTS ix_master_tickets_duplicate_group_id
                                ON master_tickets (duplicate_group_id);
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'master_tickets' AND column_name = 'duplicate_similarity'
                        ) THEN
                            ALTER TABLE master_tickets ADD COLUMN duplicate_similarity DOUBLE PRECISION;
                        END IF;
                    END $$;
                """))
                connection.commit()
                logger.info("Semantic duplicate columns verified.")

        logger.info("Database initialization completed successfully!")


    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    init_database()
