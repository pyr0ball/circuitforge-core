# circuitforge_core/community/snipe_store.py
# MIT License
"""Snipe community store — publishes seller trust signals to the shared community DB.

Snipe products subclass SharedStore here to write seller trust signals
(confirmed scammer / confirmed legitimate) to the cf_community PostgreSQL.
These signals aggregate across all Snipe users to power the cross-user
seller trust classifier fine-tuning corpus.

Privacy: only platform_seller_id (public eBay username/ID) and flag keys
are written. No PII is stored.

Usage:
    from circuitforge_core.community import CommunityDB
    from circuitforge_core.community.snipe_store import SnipeCommunityStore

    db = CommunityDB.from_env()
    store = SnipeCommunityStore(db, source_product="snipe")
    store.publish_seller_signal(
        platform_seller_id="ebay-username",
        confirmed_scam=True,
        signal_source="blocklist_add",
        flags=["new_account", "suspicious_price"],
    )
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .store import SharedStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SellerTrustSignal:
    """Immutable snapshot of a recorded seller trust signal."""
    id: int
    platform: str
    platform_seller_id: str
    confirmed_scam: bool
    signal_source: str
    flags: tuple
    source_product: str
    recorded_at: datetime


class SnipeCommunityStore(SharedStore):
    """Community store for Snipe — seller trust signal publishing and querying."""

    def __init__(self, db, source_product: str = "snipe") -> None:
        super().__init__(db, source_product=source_product)

    def publish_seller_signal(
        self,
        platform_seller_id: str,
        confirmed_scam: bool,
        signal_source: str,
        flags: list[str] | None = None,
        platform: str = "ebay",
    ) -> SellerTrustSignal:
        """Record a seller trust outcome in the shared community DB.

        Args:
            platform_seller_id: Public eBay username or platform ID (no PII).
            confirmed_scam: True = confirmed bad actor; False = confirmed legitimate.
            signal_source: Origin of the signal.
                'blocklist_add'   — user explicitly added to local blocklist
                'community_vote'  — consensus threshold reached from multiple reports
                'resolved'        — seller resolved as legitimate over time
            flags: List of red-flag keys active at signal time (e.g. ["new_account"]).
            platform: Source auction platform (default "ebay").

        Returns the inserted SellerTrustSignal.
        """
        flags = flags or []
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO seller_trust_signals
                        (platform, platform_seller_id, confirmed_scam,
                         signal_source, flags, source_product)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id, recorded_at
                    """,
                    (
                        platform,
                        platform_seller_id,
                        confirmed_scam,
                        signal_source,
                        json.dumps(flags),
                        self._source_product,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
            return SellerTrustSignal(
                id=row[0],
                platform=platform,
                platform_seller_id=platform_seller_id,
                confirmed_scam=confirmed_scam,
                signal_source=signal_source,
                flags=tuple(flags),
                source_product=self._source_product,
                recorded_at=row[1],
            )
        except Exception:
            conn.rollback()
            log.warning(
                "Failed to publish seller signal for %s (%s)",
                platform_seller_id, signal_source, exc_info=True,
            )
            raise
        finally:
            self._db.putconn(conn)

    def list_signals_for_seller(
        self,
        platform_seller_id: str,
        platform: str = "ebay",
        limit: int = 50,
    ) -> list[SellerTrustSignal]:
        """Return recent trust signals for a specific seller."""
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, platform, platform_seller_id, confirmed_scam,
                           signal_source, flags, source_product, recorded_at
                    FROM seller_trust_signals
                    WHERE platform = %s AND platform_seller_id = %s
                    ORDER BY recorded_at DESC
                    LIMIT %s
                    """,
                    (platform, platform_seller_id, limit),
                )
                rows = cur.fetchall()
            return [
                SellerTrustSignal(
                    id=r[0], platform=r[1], platform_seller_id=r[2],
                    confirmed_scam=r[3], signal_source=r[4],
                    flags=tuple(json.loads(r[5]) if isinstance(r[5], str) else r[5] or []),
                    source_product=r[6], recorded_at=r[7],
                )
                for r in rows
            ]
        finally:
            self._db.putconn(conn)

    def scam_signal_count(self, platform_seller_id: str, platform: str = "ebay") -> int:
        """Return the number of confirmed_scam=True signals for a seller.

        Used to determine if a seller has crossed the community consensus threshold
        for appearing in the shared blocklist.
        """
        conn = self._db.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM seller_trust_signals
                    WHERE platform = %s AND platform_seller_id = %s AND confirmed_scam = TRUE
                    """,
                    (platform, platform_seller_id),
                )
                return cur.fetchone()[0]
        finally:
            self._db.putconn(conn)
