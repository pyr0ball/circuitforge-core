"""
SQLite connection factory for CircuitForge products.
Supports plain SQLite and SQLCipher (AES-256) when CLOUD_MODE is active.
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path


def get_connection(db_path: Path, key: str = "") -> sqlite3.Connection:
    """
    Open a SQLite database connection.

    In cloud mode with a key: uses SQLCipher (API-identical to sqlite3).
    Otherwise: plain sqlite3.

    Args:
        db_path: Path to the database file. Created if absent.
        key:     SQLCipher encryption key. Empty = unencrypted.
    """
    cloud_mode = os.environ.get("CLOUD_MODE", "").lower() in ("1", "true", "yes")
    if cloud_mode and key:
        from pysqlcipher3 import dbapi2 as _sqlcipher  # type: ignore
        conn = _sqlcipher.connect(str(db_path), timeout=30)
        conn.execute(f"PRAGMA key='{key}'")
        return conn
    # timeout=30: retry for up to 30s when another writer holds the lock (WAL mode
    # allows concurrent readers but only one writer at a time).
    return sqlite3.connect(str(db_path), timeout=30)
