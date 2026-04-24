# Using cf-core in a Product

After [installation](installation.md), import modules directly from the package. Each module is independent — import only what you need.

## Minimal wiring example

```python
from circuitforge_core.config import Settings
from circuitforge_core.db import get_db
from circuitforge_core.tiers import require_tier
from circuitforge_core.llm import LLMRouter

settings = Settings()
db = get_db(settings.db_path)
router = LLMRouter(settings)
```

## Module shim pattern

Products that need to extend or override cf-core behavior use a shim module. This is the recommended pattern — it keeps product-specific config resolution separate from the shared implementation.

```python
# myproduct/app/llm_router.py  — shim
from circuitforge_core.llm.router import LLMRouter as _BaseLLMRouter
from .config import get_settings

class LLMRouter(_BaseLLMRouter):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            config_path=settings.llm_config_path,
            cloud_mode=settings.cloud_mode,
        )
```

Product code then imports from the shim, never directly from cf-core. This means tri-level config resolution (env → config file → defaults) and cloud mode wiring stay in one place.

!!! warning "Never import cf-core modules directly in scripts"
    Always import from the product shim. Bypassing the shim silently breaks cloud mode and config resolution. See [Peregrine's llm_router shim](https://git.opensourcesolarpunk.com/Circuit-Forge/peregrine) for the reference implementation.

## Per-user isolation (cloud mode)

When `CLOUD_MODE=true`, products use per-user SQLite trees rather than a shared database. cf-core's `db` module provides the factory; products implement their own `cloud_session.py` to resolve the per-user path from the `X-CF-Session` JWT header.

```python
# In a FastAPI endpoint with cloud mode
from .cloud_session import get_user_db_path
from circuitforge_core.db import get_db

@router.get("/items")
async def list_items(request: Request):
    db_path = get_user_db_path(request)
    db = get_db(db_path)
    ...
```

## Tier gates

Apply the `@require_tier` decorator to any endpoint or function that should be restricted:

```python
from circuitforge_core.tiers import require_tier

@router.post("/suggest")
@require_tier("paid")
async def suggest_recipe(request: Request):
    ...
```

The decorator reads the user's tier from the request context (via Heimdall JWT validation) and raises `403` if the tier is insufficient.

## Background tasks with VRAM awareness

Use `TaskScheduler` for any LLM inference that should be queued rather than run inline:

```python
from circuitforge_core.tasks import TaskScheduler

scheduler = TaskScheduler(service_name="myproduct", coordinator_url=settings.coordinator_url)

async def enqueue_generation(item_id: str):
    await scheduler.submit(
        task_type="generate",
        payload={"item_id": item_id},
        vram_gb=4.0,
    )
```

See the [tasks module reference](../modules/tasks.md) for the full API.
