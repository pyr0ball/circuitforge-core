# Editable Install Pattern

CircuitForge products depend on cf-core via `pip install -e` (editable install) from a local clone, not from a package registry. This is a deliberate architectural choice that makes the development loop fast and the dependency relationship explicit.

## How it works

`pip install -e /path/to/circuitforge-core` installs the package in "editable" mode: instead of copying files into `site-packages`, pip creates a `.pth` file pointing at the source directory. Python imports resolve directly from the cloned repo.

This means:
- Changes to cf-core source take effect immediately in all products — no reinstall needed
- Restarting the product process (or Docker container) is sufficient to pick up changes
- `git pull` in the cf-core repo automatically affects all products using it

## Docker considerations

In Docker, editable install requires the cf-core source to be present inside the container at build time. Two patterns:

**Pattern A: COPY at build time (production)**

```dockerfile
COPY circuitforge-core/ /circuitforge-core/
RUN pip install -e /circuitforge-core
```

The build context must include the cf-core directory. `compose.yml` sets the build context to the parent directory:

```yaml
services:
  api:
    build:
      context: ..          # parent of both product and cf-core
      dockerfile: myproduct/Dockerfile
```

**Pattern B: Bind-mount for dev**

```yaml
# compose.override.yml (dev only, gitignored)
services:
  api:
    volumes:
      - ../circuitforge-core:/circuitforge-core:ro
```

This lets you edit cf-core and restart the container without rebuilding the image.

## Python `.pyc` cache gotcha

Python caches compiled bytecode in `__pycache__/` directories and `.pyc` files. When cf-core source is updated but the product hasn't been restarted, the old `.pyc` files can serve stale code even with the bind-mount in place.

Fix: delete `.pyc` files and restart:

```bash
find /path/to/circuitforge-core -name "*.pyc" -delete
docker compose restart api
```

This is especially common when fixing an import error — the old `ImportError` may persist even after the fix if the bytecode cache isn't cleared.

## When to reinstall

A full `pip install -e .` reinstall is needed when:
- `pyproject.toml` changes (new dependencies, entry points, package metadata)
- A new subpackage directory is added (pip needs to discover it)
- The `.egg-info` directory gets corrupted (delete it and reinstall)

```bash
# Reinstall in the cf env
conda run -n cf pip install -e /Library/Development/CircuitForge/circuitforge-core
```

## Future: Forgejo Packages

When cf-core reaches a stable enough interface (currently targeting "third product shipped"), it will be published to the Circuit-Forge Forgejo private PyPI registry. Products will then depend on it via version pin, and the editable install will be for development only. The shim pattern is designed to make this transition smooth — product code stays the same, only the import source changes.
