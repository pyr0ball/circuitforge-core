# db

SQLite connection factory and migration runner. Every CircuitForge product uses this for all persistent storage.

```python
from circuitforge_core.db import get_db, run_migrations
```

## Why SQLite

SQLite is local-first by nature — no server process, no network dependency, trivially backed up, and fast enough for single-user workloads. circuitforge-core's `db` module adds migration management and connection pooling on top.

## API

### `get_db(path: str | Path) -> Connection`

Returns a SQLite connection to the database at `path`. Creates the file if it doesn't exist. Enables WAL mode, foreign keys, and sets a sensible busy timeout by default.

```python
db = get_db("/devl/kiwi-data/kiwi.db")
```

In cloud mode, the path comes from the per-user session resolver — never hardcode `DB_PATH` directly in endpoints. Use `_request_db.get() or DB_PATH` or a product shim.

### `run_migrations(db: Connection, migrations_dir: str | Path)`

Discovers and applies all `.sql` files in `migrations_dir` that haven't yet been applied, in filename order. Migration state is tracked in a `_migrations` table created on first run.

```python
run_migrations(db, "app/db/migrations/")
```

**Migration file naming:** `001_initial.sql`, `002_add_column.sql`, etc. Always prefix with zero-padded integers. Never renumber or delete applied migrations.

### `RETURNING *` gotcha

SQLite added `RETURNING *` in version 3.35 (2021). When using it:

```python
cursor = db.execute("INSERT INTO items (...) VALUES (?) RETURNING *", (...,))
row = cursor.fetchone()   # fetch BEFORE commit — row disappears after commit
db.commit()
```

This is a known SQLite behavior that differs from PostgreSQL. cf-core does not paper over it; fetch before committing.

## Migration conventions

- Files go in `app/db/migrations/` inside each product repo
- One concern per file — don't combine unrelated schema changes
- Never use `ALTER TABLE` to rename columns (not supported in SQLite < 3.25); add a new column and migrate data instead
- `IF NOT EXISTS` and `IF EXISTS` guards make migrations idempotent

## Cloud mode

In cloud mode, each user gets their own SQLite file under `CLOUD_DATA_ROOT`. The `db` module is unaware of this; the product's `cloud_session.py` resolves the per-user path before calling `get_db()`.
