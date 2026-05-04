# circuitforge_core/vector/sqlite_vec.py
"""
circuitforge_core.vector.sqlite_vec -- sqlite-vec backed VectorStore.

Suitable for single-user local deployments. Cloud Paid tier replaces
this with QdrantStore via the same VectorStore ABC.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import sqlite_vec

from .base import VectorMatch, VectorStore

logger = logging.getLogger(__name__)


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


class LocalSQLiteVecStore(VectorStore):
    """
    VectorStore backed by sqlite-vec virtual tables.

    Uses two tables per logical store:
    - ``<table>_vecs``:  vec0 virtual table (rowid-indexed float vectors)
    - ``<table>_meta``:  companion table mapping rowid to string ID + JSON metadata

    Args:
        db_path:    Path to SQLite database file.
        table:      Logical name prefix (default ``"vecs"``).
        dimensions: Vector length; must match the embedding model (default 768).
    """

    def __init__(
        self,
        db_path: str | Path,
        table: str = "vecs",
        dimensions: int = 768,
    ) -> None:
        self.db_path = str(db_path)
        self.table = table
        self.dimensions = dimensions
        self._init_tables()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_tables(self) -> None:
        with self._conn() as conn:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table}_vecs
                USING vec0(embedding float[{self.dimensions}])
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table}_meta (
                    rowid    INTEGER PRIMARY KEY,
                    entry_id TEXT NOT NULL UNIQUE,
                    metadata TEXT NOT NULL DEFAULT '{{}}'
                )
            """)

    def upsert(
        self, entry_id: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT rowid FROM {self.table}_meta WHERE entry_id = ?", [entry_id]
            ).fetchone()

            if row:
                rowid = row["rowid"]
                conn.execute(
                    f"UPDATE {self.table}_vecs SET embedding = ? WHERE rowid = ?",
                    [_serialize(vector), rowid],
                )
                conn.execute(
                    f"UPDATE {self.table}_meta SET metadata = ? WHERE rowid = ?",
                    [json.dumps(metadata), rowid],
                )
            else:
                cursor = conn.execute(
                    f"INSERT INTO {self.table}_meta(entry_id, metadata) VALUES (?, ?)",
                    [entry_id, json.dumps(metadata)],
                )
                rowid = cursor.lastrowid
                conn.execute(
                    f"INSERT INTO {self.table}_vecs(rowid, embedding) VALUES (?, ?)",
                    [rowid, _serialize(vector)],
                )

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT m.entry_id, v.distance, m.metadata
                FROM {self.table}_vecs v
                JOIN {self.table}_meta m ON m.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                [_serialize(vector), top_k],
            ).fetchall()

        results = [
            VectorMatch(
                entry_id=r["entry_id"],
                score=r["distance"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

        if filter_metadata:
            results = [
                r
                for r in results
                if all(r.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        return results

    def delete(self, entry_id: str) -> None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT rowid FROM {self.table}_meta WHERE entry_id = ?", [entry_id]
            ).fetchone()
            if row:
                rowid = row["rowid"]
                conn.execute(f"DELETE FROM {self.table}_vecs WHERE rowid = ?", [rowid])
                conn.execute(f"DELETE FROM {self.table}_meta WHERE rowid = ?", [rowid])

    def delete_where(self, filter_metadata: dict[str, Any]) -> int:
        if not filter_metadata:
            raise ValueError(
                "delete_where requires a non-empty filter; refusing to delete entire store"
            )
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT rowid, metadata FROM {self.table}_meta"
            ).fetchall()
            to_delete = [
                r["rowid"]
                for r in rows
                if all(
                    json.loads(r["metadata"]).get(k) == v
                    for k, v in filter_metadata.items()
                )
            ]
            for rowid in to_delete:
                conn.execute(f"DELETE FROM {self.table}_vecs WHERE rowid = ?", [rowid])
                conn.execute(f"DELETE FROM {self.table}_meta WHERE rowid = ?", [rowid])
            return len(to_delete)
