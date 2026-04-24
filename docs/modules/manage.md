# manage

`manage.sh` scaffolding for cross-platform product process management. Every CircuitForge product ships a `manage.sh` generated from this module.

```python
from circuitforge_core.manage import generate_manage_sh, ProcessManager
```

## Purpose

`manage.sh` is the single entry point for starting, stopping, restarting, and checking the status of a product. It abstracts over Docker Compose (production) and native Python processes (development without Docker).

## Commands

Every product's `manage.sh` supports:

```bash
bash manage.sh start          # Start all services
bash manage.sh stop           # Stop all services
bash manage.sh restart        # Stop then start
bash manage.sh status         # Print running state
bash manage.sh logs           # Tail logs
bash manage.sh open           # Open the product UI in a browser
bash manage.sh update         # Pull latest and restart
```

Products add their own subcommands by extending the base script.

## Docker mode (production)

In Docker mode, `manage.sh` delegates to `docker compose`. The script auto-detects whether Docker is available and falls back to native mode if not.

```bash
# manage.sh internals (Docker mode)
docker compose -f compose.yml up -d
docker compose -f compose.yml logs -f
```

For cloud deployments, products have a `compose.cloud.yml` that's overlaid:

```bash
docker compose -f compose.yml -f compose.cloud.yml up -d
```

## Preflight

`manage.sh start` calls `preflight.py` before launching containers. Preflight:
1. Enumerates GPUs and writes a Docker Compose profile recommendation
2. Checks for port conflicts and auto-increments if needed
3. Detects external services (Ollama, vLLM, SearXNG) already running and adopts them via `compose.override.yml`
4. Writes the final `.env` for the current session

## Extending manage.sh

Products add subcommands by checking `$1` before the default case:

```bash
case "$1" in
  backfill)
    conda run -n cf python scripts/backfill_keywords.py
    ;;
  *)
    # Default manage.sh handling
    ...
    ;;
esac
```
