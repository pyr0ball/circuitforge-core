"""
circuitforge_core.resources.coordinator.node_store — SQLite persistence for known agent nodes.

Gives the coordinator restart-safe memory of which nodes have ever registered.
On startup the coordinator reloads all known nodes and immediately probes them;
nodes that respond come back online within one heartbeat cycle (~10 s) without
any manual intervention on the agent hosts.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "circuitforge" / "cf-orch-nodes.db"
_STALE_AGE_DAYS = 30  # nodes unseen for this long are pruned automatically


class NodeStore:
    """
    Thin SQLite wrapper for persisting known agent nodes across coordinator restarts.

    Thread-safe for single-writer use (coordinator runs in one asyncio thread).
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()
        logger.debug("NodeStore initialised at %s", db_path)

    def _migrate(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS known_nodes (
                node_id    TEXT PRIMARY KEY,
                agent_url  TEXT NOT NULL,
                last_seen  REAL NOT NULL
            );
        """)
        self._conn.commit()

    def upsert(self, node_id: str, agent_url: str) -> None:
        """Record or update a node. Called on every successful registration."""
        self._conn.execute(
            """
            INSERT INTO known_nodes (node_id, agent_url, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                agent_url = excluded.agent_url,
                last_seen = excluded.last_seen
            """,
            (node_id, agent_url, time.time()),
        )
        self._conn.commit()

    def all(self) -> list[tuple[str, str]]:
        """Return all known (node_id, agent_url) pairs."""
        rows = self._conn.execute(
            "SELECT node_id, agent_url FROM known_nodes ORDER BY last_seen DESC"
        ).fetchall()
        return [(r["node_id"], r["agent_url"]) for r in rows]

    def remove(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM known_nodes WHERE node_id = ?", (node_id,))
        self._conn.commit()

    def prune_stale(self, max_age_days: int = _STALE_AGE_DAYS) -> int:
        """Delete nodes not seen within max_age_days. Returns count removed."""
        cutoff = time.time() - max_age_days * 86400
        cur = self._conn.execute(
            "DELETE FROM known_nodes WHERE last_seen < ?", (cutoff,)
        )
        self._conn.commit()
        removed = cur.rowcount
        if removed:
            logger.info("NodeStore: pruned %d stale node(s) (>%d days old)", removed, max_age_days)
        return removed

    def close(self) -> None:
        self._conn.close()
