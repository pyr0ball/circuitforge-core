import sqlite3
import tempfile
from pathlib import Path
import pytest
from circuitforge_core.db import get_connection, run_migrations


def test_get_connection_returns_sqlite_connection(tmp_path):
    db = tmp_path / "test.db"
    conn = get_connection(db)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_get_connection_creates_file(tmp_path):
    db = tmp_path / "test.db"
    assert not db.exists()
    conn = get_connection(db)
    conn.close()
    assert db.exists()


def test_run_migrations_applies_sql_files(tmp_path):
    db = tmp_path / "test.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_foo.sql").write_text(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT);"
    )
    conn = get_connection(db)
    run_migrations(conn, migrations_dir)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='foo'")
    assert cursor.fetchone() is not None
    conn.close()


def test_run_migrations_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_foo.sql").write_text(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY, name TEXT);"
    )
    conn = get_connection(db)
    run_migrations(conn, migrations_dir)
    run_migrations(conn, migrations_dir)  # second run must not raise
    conn.close()


def test_run_migrations_applies_in_order(tmp_path):
    db = tmp_path / "test.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_create_foo.sql").write_text(
        "CREATE TABLE foo (id INTEGER PRIMARY KEY);"
    )
    (migrations_dir / "002_add_name.sql").write_text(
        "ALTER TABLE foo ADD COLUMN name TEXT;"
    )
    conn = get_connection(db)
    run_migrations(conn, migrations_dir)
    conn.execute("INSERT INTO foo (name) VALUES ('bar')")
    conn.close()
