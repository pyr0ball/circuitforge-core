# pipeline

Staging queue base class. **Stub — partially implemented.**

```python
from circuitforge_core.pipeline import StagingDB  # base class
```

## Purpose

`StagingDB` is the base class for the staging layer that sits between discovery/ingestion and the main product workflow. Products subclass it to add their concrete schema.

The pattern:
```
Source (scraper / scan / upload)
    → StagingDB (raw, unreviewed records)
    → Human review / approval
    → Main product DB (approved records)
```

This is explicit in Peregrine (jobs go from `pending` → `approved` → `applied`) and analogous in Kiwi (receipts go from `uploaded` → `parsed` → `pantry`).

## Crystallization engine

The pipeline module also contains the crystallization engine: a system for promoting AI-generated drafts through a series of structured human-approval checkpoints before the output "crystallizes" into a permanent record.

Each stage in the pipeline has:
- An **AI step** that produces a draft
- A **human approval gate** that must be explicitly cleared
- A **rollback path** back to the previous stage if rejected

This is the architectural embodiment of the "LLMs as drafts, never decisions" principle.

## Current status

`StagingDB` base class exists and is used by Peregrine's job pipeline. The crystallization engine design is documented in `circuitforge-plans/shared/superpowers/specs/` and is being extracted into this module as it stabilizes across products.

## `StagingDB` base class

```python
class StagingDB:
    def __init__(self, db: Connection):
        self.db = db

    def stage(self, record: dict) -> str:
        """Insert a record into staging. Returns record ID."""
        raise NotImplementedError

    def approve(self, record_id: str, reviewer_id: str | None = None):
        """Promote a record past the approval gate."""
        raise NotImplementedError

    def reject(self, record_id: str, reason: str | None = None):
        """Mark a record as rejected."""
        raise NotImplementedError

    def pending(self) -> list[dict]:
        """Return all records awaiting review."""
        raise NotImplementedError
```
