"""
Database initialization script for CivicPulse.
Connects to PostgreSQL, activates PostGIS spatial extension if not present,
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
from backend.database.models import MasterTicket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.init_db")


def init_database() -> None:
    """
    Initialize the database:
      1. Verify database connectivity.
      2. Ensure PostGIS spatial extension is enabled (on PostgreSQL).
      3. Create all tables registered with Base.metadata (MasterTicket, etc.).
    """
    logger.info("Initializing CivicPulse database...")
    logger.info("Target connection URL: %s", engine.url)

    try:
        with engine.connect() as connection:
            # 1. Enable PostGIS extension for PostgreSQL dialects
            if engine.dialect.name == "postgresql":
                logger.info("Ensuring PostGIS extension is active...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                connection.commit()
                logger.info("PostGIS extension confirmed.")

                # Ensure report_count column exists if table was pre-existing
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

        # 2. Create all ORM tables
        logger.info("Creating database tables from ORM metadata...")
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully created all tables: %s", list(Base.metadata.tables.keys()))
        logger.info("Database initialization completed successfully!")

    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    init_database()
