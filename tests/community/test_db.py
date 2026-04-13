# tests/community/test_db.py
import os
import pytest
from unittest.mock import MagicMock, patch, call
from circuitforge_core.community.db import CommunityDB


@pytest.fixture
def mock_pool():
    """Patch psycopg2.pool.ThreadedConnectionPool to avoid needing a real PG instance."""
    with patch("circuitforge_core.community.db.ThreadedConnectionPool") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


def test_community_db_requires_url():
    with pytest.raises(ValueError, match="COMMUNITY_DB_URL"):
        CommunityDB(dsn=None)


def test_community_db_init_creates_pool(mock_pool):
    mock_cls, _ = mock_pool
    CommunityDB(dsn="postgresql://user:pass@localhost/cf_community")
    mock_cls.assert_called_once()


def test_community_db_close_puts_pool(mock_pool):
    _, mock_instance = mock_pool
    db = CommunityDB(dsn="postgresql://user:pass@localhost/cf_community")
    db.close()
    mock_instance.closeall.assert_called_once()


def test_community_db_migration_files_discovered():
    """Migration runner must find at least 001 and 002 SQL files."""
    db = CommunityDB.__new__(CommunityDB)
    files = db._discover_migrations()
    names = [f.name for f in files]
    assert any("001" in n for n in names)
    assert any("002" in n for n in names)
    # Must be sorted numerically
    assert files == sorted(files, key=lambda p: p.name)


def test_community_db_run_migrations_executes_sql(mock_pool):
    _, mock_instance = mock_pool
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_instance.getconn.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    db = CommunityDB(dsn="postgresql://user:pass@localhost/cf_community")
    db.run_migrations()

    # At least one execute call must have happened
    assert mock_cur.execute.called


def test_community_db_from_env(monkeypatch, mock_pool):
    monkeypatch.setenv("COMMUNITY_DB_URL", "postgresql://u:p@host/db")
    db = CommunityDB.from_env()
    assert db is not None
