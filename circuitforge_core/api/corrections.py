"""
Shared corrections router — stores user corrections to LLM output for SFT training.

Products include this with make_corrections_router(get_db=..., product=...).
Corrections are stored locally in each product's SQLite DB and exported as JSONL
for the Avocet SFT pipeline. Separate from the bug-feedback→Forgejo-issue path.

Required DB migration (add to product migrations dir):
    -- From circuitforge_core.api.corrections import CORRECTIONS_MIGRATION_SQL
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Drop this SQL into a product's migrations directory (e.g. 020_corrections.sql).
CORRECTIONS_MIGRATION_SQL = """\
CREATE TABLE IF NOT EXISTS corrections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id          TEXT    NOT NULL DEFAULT '',
    product          TEXT    NOT NULL,
    correction_type  TEXT    NOT NULL,
    input_text       TEXT    NOT NULL,
    original_output  TEXT    NOT NULL,
    corrected_output TEXT    NOT NULL DEFAULT '',
    rating           TEXT    NOT NULL DEFAULT 'down',
    context          TEXT    NOT NULL DEFAULT '{}',
    opted_in         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_corrections_product
    ON corrections (product);

CREATE INDEX IF NOT EXISTS idx_corrections_opted_in
    ON corrections (opted_in);
"""


class CorrectionRequest(BaseModel):
    item_id: str = ""
    product: str
    correction_type: str
    input_text: str
    original_output: str
    corrected_output: str = ""
    rating: Literal["up", "down"] = "down"
    context: dict = Field(default_factory=dict)
    opted_in: bool = False


class CorrectionResponse(BaseModel):
    id: int
    saved: bool


class CorrectionRecord(BaseModel):
    id: int
    item_id: str
    product: str
    correction_type: str
    input_text: str
    original_output: str
    corrected_output: str
    rating: str
    context: dict
    opted_in: bool
    created_at: str


def make_corrections_router(
    get_db: Callable[[], Iterator[sqlite3.Connection]],
    product: str,
) -> APIRouter:
    """Return a configured corrections APIRouter.

    Args:
        get_db: FastAPI dependency that yields a sqlite3.Connection.
        product: Product slug injected into every correction row (e.g. "linnet").
    """
    router = APIRouter()

    @router.post("", response_model=CorrectionResponse)
    def submit_correction(
        payload: CorrectionRequest,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> CorrectionResponse:
        """Store a user correction to an LLM output."""
        # Thumbs-up with no corrected text is a valid positive signal.
        if payload.rating == "down" and not payload.corrected_output.strip():
            raise HTTPException(
                status_code=422,
                detail="corrected_output is required when rating is 'down'.",
            )

        row_id = conn.execute(
            """
            INSERT INTO corrections
                (item_id, product, correction_type, input_text, original_output,
                 corrected_output, rating, context, opted_in)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.item_id,
                product,
                payload.correction_type,
                payload.input_text,
                payload.original_output,
                payload.corrected_output,
                payload.rating,
                json.dumps(payload.context),
                int(payload.opted_in),
            ),
        ).lastrowid
        conn.commit()
        return CorrectionResponse(id=row_id, saved=True)

    @router.get("", response_model=list[CorrectionRecord])
    def list_corrections(
        opted_in_only: bool = False,
        limit: int = 200,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> list[CorrectionRecord]:
        """List stored corrections, optionally filtered to opted-in rows only."""
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM corrections"
        params: list = []
        if opted_in_only:
            query += " WHERE opted_in = 1"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        rows = conn.execute(query, params).fetchall()
        return [
            CorrectionRecord(
                id=r["id"],
                item_id=r["item_id"],
                product=r["product"],
                correction_type=r["correction_type"],
                input_text=r["input_text"],
                original_output=r["original_output"],
                corrected_output=r["corrected_output"],
                rating=r["rating"],
                context=json.loads(r["context"] or "{}"),
                opted_in=bool(r["opted_in"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    @router.get("/export")
    def export_corrections(
        opted_in_only: bool = True,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> StreamingResponse:
        """Stream corrections as JSONL for the Avocet SFT pipeline.

        Each line is a JSON object with the fields expected by avocet's
        SFT candidate importer. opted_in_only=True (default) — only rows
        where the user consented to share are exported.
        """
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM corrections"
        if opted_in_only:
            query += " WHERE opted_in = 1"
        query += " ORDER BY created_at ASC"
        rows = conn.execute(query).fetchall()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"corrections_{product}_{timestamp}.jsonl"

        def generate() -> Iterator[str]:
            for r in rows:
                record = {
                    "input": r["input_text"],
                    "output": r["original_output"],
                    "correction": r["corrected_output"],
                    "rating": r["rating"],
                    "correction_type": r["correction_type"],
                    "product": r["product"],
                    "item_id": r["item_id"],
                    "context": json.loads(r["context"] or "{}"),
                    "created_at": r["created_at"],
                }
                yield json.dumps(record, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
