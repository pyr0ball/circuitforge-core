# config

Env-driven settings with `.env` file loading. Provides a base `Settings` class that products subclass to add their own fields.

```python
from circuitforge_core.config import Settings
```

## Design

Configuration follows a strict priority order: **environment variables > `.env` file > defaults**. This means Docker compose `environment:` overrides always win, which is essential for cloud vs local deployment switching without image rebuilds.

## Base Settings

```python
class Settings(BaseSettings):
    # Database
    db_path: str = "data/app.db"

    # LLM
    llm_config_path: str = "config/llm.yaml"

    # Tier system
    license_key: str | None = None
    cloud_mode: bool = False

    # Cloud
    cloud_data_root: str = "/devl/app-cloud-data"
    cloud_auth_bypass_ips: list[str] = []
    coordinator_url: str = "http://10.1.10.71:7700"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Extending in a product

```python
# myproduct/app/core/config.py
from circuitforge_core.config import Settings as _BaseSettings

class Settings(_BaseSettings):
    # Product-specific settings
    max_pantry_items: int = 500
    barcode_timeout_ms: int = 5000
    recipe_corpus_path: str = "data/recipes.db"

    class Config(_BaseSettings.Config):
        env_prefix = "MYPRODUCT_"
```

## `.env` file

Each product ships a `.env.example` (committed) and a `.env` (gitignored). The `.env` file is loaded automatically by the `Settings` class.

```bash
# .env.example
DB_PATH=data/app.db
CLOUD_MODE=false
LICENSE_KEY=
```

!!! tip "Never commit `.env`"
    `.env` files contain secrets and environment-specific paths. Always commit `.env.example` instead.

## Singleton pattern

Products typically expose a cached `get_settings()` function:

```python
from functools import lru_cache
from .config import Settings

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

This ensures the `.env` file is only read once at startup, and all modules share the same settings instance.
