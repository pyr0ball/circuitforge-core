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

    Resilient to partial-failure recovery: if a migration previously crashed
    mid-run (e.g. a process killed after some ALTER TABLE statements
    auto-committed via executescript), the next startup re-runs that migration.
    Any "duplicate column name" errors are silently skipped so the migration
    can complete and be marked as applied.  All other errors still propagate.
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

        try:
            conn.executescript(sql_file.read_text())
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
            # A previous run partially applied this migration (some ALTER TABLE
            # statements auto-committed before the failure).  Re-run with
            # per-statement recovery to skip already-applied columns.
            _log.warning(
                "Migration %s: partial-failure detected (%s) — "
                "retrying with per-statement recovery",
                sql_file.name,
                exc,
            )
            _run_script_with_recovery(conn, sql_file)

        # OR IGNORE: safe if two Store() calls race on the same DB — second writer
        # just skips the insert rather than raising UNIQUE constraint failed.
        conn.execute("INSERT OR IGNORE INTO _migrations (name) VALUES (?)", (sql_file.name,))
        conn.commit()


def _run_script_with_recovery(conn: sqlite3.Connection, sql_file: Path) -> None:
    """Re-run a migration via executescript, skipping duplicate-column errors.

    Used only when the first executescript() attempt raised a duplicate column
    error (indicating a previous partial run).  Splits the script on the
    double-dash comment prefix pattern to re-issue each logical statement,
    catching only the known-safe "duplicate column name" error class.

    Splitting is done via SQLite's own parser — we feed the script to a
    temporary in-memory connection using executescript (which commits
    auto-matically per DDL statement) and mirror the results on the real
    connection statement by statement.  That's circular, so instead we use
    the simpler approach: executescript handles tokenization; we wrap the
    whole call in a try/except and retry after removing the offending statement.

    Simpler approach: use conn.execute() per statement from the script.
    This avoids the semicolon-in-comment tokenization problem by not splitting
    ourselves — instead we let the DB tell us which statement failed and only
    skip that exact error class.
    """
    # executescript() uses SQLite's real tokenizer, so re-issuing it after a
    # partial failure will hit "duplicate column name" again.  We catch and
    # ignore that specific error class only, re-running until the script
    # completes or a different error is raised.
    #
    # Implementation: issue the whole script again; catch duplicate-column
    # errors; keep trying.  Since executescript auto-commits per statement,
    # each successful statement in successive retries is a no-op (CREATE TABLE
    # IF NOT EXISTS, etc.) or a benign duplicate skip.
    #
    # Limit retries to prevent infinite loops on genuinely broken SQL.
    script = sql_file.read_text()
    for attempt in range(20):
        try:
            conn.executescript(script)
            return  # success
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column name" in msg:
                col = str(exc).split(":")[-1].strip() if ":" in str(exc) else "?"
                _log.warning(
                    "Migration %s (attempt %d): skipping duplicate column '%s'",
                    sql_file.name,
                    attempt + 1,
                    col,
                )
                # Remove the offending ALTER TABLE statement from the script
                # so the next attempt skips it.  This is safe because SQLite
                # already auto-committed that column addition on a prior run.
                script = _remove_column_add(script, col)
            else:
                raise
    raise RuntimeError(
        f"Migration {sql_file.name}: could not complete after 20 recovery attempts"
    )


def _remove_column_add(script: str, column: str) -> str:
    """Remove the ALTER TABLE ADD COLUMN statement for *column* from *script*."""
    import re
    # Match: ALTER TABLE <tbl> ADD COLUMN <column> <rest-of-line>
    pattern = re.compile(
        r"ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+" + re.escape(column) + r"[^\n]*\n?",
        re.IGNORECASE,
    )
    return pattern.sub("", script)
