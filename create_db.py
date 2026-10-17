"""
Database creation script for CivicPulse.
Parses credentials from backend.config.settings.DATABASE_URL,
connects to the default PostgreSQL maintenance database ('postgres'),
and creates the target application database ('civicpulse_db') if it does not already exist.
"""

import sys
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql
from sqlalchemy.engine import make_url

from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.create_db")


def create_database() -> None:
    """Connect to default 'postgres' database and create the target database."""
    url = make_url(settings.DATABASE_URL)
    target_db = url.database
    user = url.username or "postgres"
    password = url.password
    host = url.host or "localhost"
    port = url.port or 5432

    logger.info("Parsed DB configuration: host=%s, port=%s, user=%s, target_db=%s", host, port, user, target_db)
    logger.info("Connecting to maintenance database 'postgres' to provision '%s'...", target_db)

    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if target database already exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
        exists = cursor.fetchone()

        if exists:
            logger.info("Database '%s' already exists. Skipping creation.", target_db)
        else:
            logger.info("Creating database '%s'...", target_db)
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
            logger.info("Database '%s' successfully created!", target_db)

        cursor.close()
        conn.close()

    except Exception as exc:
        logger.error("Failed to create database '%s': %s", target_db, exc)
        sys.exit(1)


if __name__ == "__main__":
    create_database()
