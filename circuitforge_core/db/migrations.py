"""
Sequential SQL migration runner.
Applies *.sql files from migrations_dir in filename order.
Tracks applied migrations in a _migrations table — safe to call multiple times.
"""
from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

_log = logging.getLogger(__name__)


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply any unapplied *.sql migrations from migrations_dir.

    Resilient to partial-failure recovery: if a migration previously failed
    mid-run (e.g. a crash after some ALTER TABLE statements auto-committed),
    "duplicate column name" errors on re-run are silently skipped so the
    migration can complete and be marked as applied.  All other errors still
    propagate.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations "
        "(name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.commit()

    applied = {row[0] for row in conn.execute("SELECT name FROM _migrations")}
    sql_files = sorted(migrations_dir.glob("*.sql"))

    for sql_file in sql_files:
        if sql_file.name in applied:
            continue

        _run_script(conn, sql_file)

        # OR IGNORE: safe if two Store() calls race on the same DB — second writer
        # just skips the insert rather than raising UNIQUE constraint failed.
        conn.execute("INSERT OR IGNORE INTO _migrations (name) VALUES (?)", (sql_file.name,))
        conn.commit()


def _run_script(conn: sqlite3.Connection, sql_file: Path) -> None:
    """Execute a SQL migration file, statement by statement.

    Splits on ';' so that individual DDL statements can be skipped on
    "duplicate column name" errors (partial-failure recovery) without
    silencing real errors.  Empty statements and pure-comment chunks are
    skipped automatically.
    """
    text = sql_file.read_text()

    # Split into individual statements.  This is a simple heuristic —
    # semicolons inside string literals would confuse it, but migration files
    # should never contain such strings.
    for raw in text.split(";"):
        stmt = raw.strip()
        if not stmt or stmt.startswith("--"):
            continue
        # Strip inline leading comments (block of -- lines before the SQL).
        lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if not stmt:
            continue

        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                _log.warning(
                    "Migration %s: skipping already-present column (%s) — "
                    "partial-failure recovery",
                    sql_file.name,
                    exc,
                )
            else:
                raise
