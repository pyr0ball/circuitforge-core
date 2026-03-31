"""
Sequential SQL migration runner.
Applies *.sql files from migrations_dir in filename order.
Tracks applied migrations in a _migrations table — safe to call multiple times.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply any unapplied *.sql migrations from migrations_dir."""
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
        conn.executescript(sql_file.read_text())
        # OR IGNORE: safe if two Store() calls race on the same DB — second writer
        # just skips the insert rather than raising UNIQUE constraint failed.
        conn.execute("INSERT OR IGNORE INTO _migrations (name) VALUES (?)", (sql_file.name,))
        conn.commit()
