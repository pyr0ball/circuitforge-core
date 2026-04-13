# circuitforge_core/community/db.py
# MIT License

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

_MIN_CONN = 1
_MAX_CONN = 10


class CommunityDB:
    """Shared PostgreSQL connection pool + migration runner for the community module.

    Products instantiate one CommunityDB at startup and pass it to SharedStore
    subclasses. The pool is thread-safe (ThreadedConnectionPool).

    Usage:
        db = CommunityDB.from_env()   # reads COMMUNITY_DB_URL
        db.run_migrations()
        store = MyProductStore(db)
        db.close()                    # at shutdown
    """

    def __init__(self, dsn: str | None) -> None:
        if not dsn:
            raise ValueError(
                "CommunityDB requires a DSN. "
                "Set COMMUNITY_DB_URL or pass dsn= explicitly."
            )
        self._pool = ThreadedConnectionPool(_MIN_CONN, _MAX_CONN, dsn=dsn)
        logger.debug("CommunityDB pool created (min=%d, max=%d)", _MIN_CONN, _MAX_CONN)

    @classmethod
    def from_env(cls) -> "CommunityDB":
        """Construct from the COMMUNITY_DB_URL environment variable."""
        import os
        dsn = os.environ.get("COMMUNITY_DB_URL")
        return cls(dsn=dsn)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def getconn(self):
        """Borrow a connection from the pool. Must be returned via putconn()."""
        return self._pool.getconn()

    def putconn(self, conn) -> None:
        """Return a borrowed connection to the pool."""
        self._pool.putconn(conn)

    def close(self) -> None:
        """Close all pool connections. Call at application shutdown."""
        self._pool.closeall()
        logger.debug("CommunityDB pool closed")

    # ------------------------------------------------------------------
    # Migration runner
    # ------------------------------------------------------------------

    def _discover_migrations(self) -> list[Path]:
        """Return sorted list of .sql migration files from the community migrations dir."""
        pkg = importlib.resources.files("circuitforge_core.community.migrations")
        files = sorted(
            [Path(str(p)) for p in pkg.iterdir() if str(p).endswith(".sql")],
            key=lambda p: p.name,
        )
        return files

    def run_migrations(self) -> None:
        """Apply all community migration SQL files in numeric order.

        Uses a simple applied-migrations table to avoid re-running already
        applied migrations. Idempotent.
        """
        conn = self.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS _community_migrations (
                        filename TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                conn.commit()

                for migration_file in self._discover_migrations():
                    name = migration_file.name
                    cur.execute(
                        "SELECT 1 FROM _community_migrations WHERE filename = %s",
                        (name,),
                    )
                    if cur.fetchone():
                        logger.debug("Migration %s already applied, skipping", name)
                        continue

                    sql = migration_file.read_text()
                    logger.info("Applying community migration: %s", name)
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO _community_migrations (filename) VALUES (%s)",
                        (name,),
                    )
                    conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.putconn(conn)
