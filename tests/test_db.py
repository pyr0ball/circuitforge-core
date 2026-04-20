import os
import sqlite3
import tempfile
from pathlib import Path
import pytest
from circuitforge_core.db import get_connection, run_migrations

sqlcipher_available = pytest.mark.skipif(
    __import__("importlib").util.find_spec("pysqlcipher3") is None,
    reason="pysqlcipher3 not installed",
)


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


# ── SQLCipher PRAGMA key tests (skipped when pysqlcipher3 not installed) ──────


@sqlcipher_available
def test_sqlcipher_key_with_special_chars_does_not_inject(tmp_path, monkeypatch):
    """Key containing a single quote must not cause a SQL syntax error.

    Regression for: conn.execute(f"PRAGMA key='{key}'") — if key = "x'--"
    the f-string form produced a broken PRAGMA statement. Parameterized
    form (PRAGMA key=?) must handle this safely.
    """
    monkeypatch.setenv("CLOUD_MODE", "1")
    db = tmp_path / "enc.db"
    tricky_key = "pass'word\"--inject"
    # Must not raise; if the f-string form were used, this would produce
    # a syntax error or silently set an incorrect key.
    conn = get_connection(db, key=tricky_key)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.close()


@sqlcipher_available
def test_sqlcipher_wrong_key_raises(tmp_path, monkeypatch):
    """Opening an encrypted DB with the wrong key should raise, not silently corrupt."""
    monkeypatch.setenv("CLOUD_MODE", "1")
    db = tmp_path / "enc.db"
    conn = get_connection(db, key="correct-key")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.close()

    with pytest.raises(Exception):
        bad = get_connection(db, key="wrong-key")
        bad.execute("SELECT * FROM t")  # should raise on bad key
