# Adding a Module to cf-core

This guide walks through extracting a pattern from a product into a shared cf-core module. The goal is to move battle-tested implementations here once they've stabilized in at least two products.

## When to add a module

Add a module when:
- The same pattern exists in two or more products with minor variations
- The interface is stable enough that changing it would require coordinated updates across products
- The code has no product-specific business logic baked in

Do not add a module for:
- One-off utilities that only one product needs
- Anything still in active design flux
- Product-specific configuration or policy decisions

## Module structure

```
circuitforge_core/
└── mymodule/
    ├── __init__.py        # Public API — what products import
    ├── base.py            # Core implementation
    └── backends/          # Optional: pluggable backends
        ├── __init__.py
        ├── local.py
        └── cloud.py
```

Keep the public API in `__init__.py` clean. Products should import from `circuitforge_core.mymodule`, not from internal submodules.

## Step 1: Define the interface

Write the public interface first — the classes and functions products will call. Get this right before implementing, because changing it requires updating every product shim.

```python
# circuitforge_core/mymodule/__init__.py

from .base import MyThing, get_my_thing

__all__ = ["MyThing", "get_my_thing"]
```

## Step 2: Implement with a stub

Start with a minimal working implementation. Stub out anything uncertain:

```python
# circuitforge_core/mymodule/base.py

class MyThing:
    def __init__(self, config: dict):
        self._config = config

    def do_thing(self, input: str) -> str:
        raise NotImplementedError("Override in product or backend")
```

## Step 3: Write tests

Tests go in `circuitforge_core/tests/test_mymodule.py`. Use `pytest`. The cf env has pytest installed.

```bash
conda run -n cf python -m pytest tests/test_mymodule.py -v
```

Cover:
- Happy path with realistic input
- Missing config / bad input (fail loudly, not silently)
- Cloud vs local mode if applicable

## Step 4: Update `pyproject.toml`

Add any new dependencies:

```toml
[project.optional-dependencies]
mymodule = ["some-dep>=1.0"]
```

Use optional dependency groups so products that don't use the module don't pay the install cost.

## Step 5: Write the docs page

Add `docs/modules/mymodule.md` following the pattern of the existing module docs. Include:
- Import path
- Why this module exists / design rationale
- Full public API with examples
- Any gotchas or non-obvious behavior
- Status (Stable / Stub)

Update `docs/modules/index.md` and `mkdocs.yml` to include the new page.

## Step 6: Update products

In each product that uses the pattern:
1. Add a shim if the product needs to override behavior
2. Replace the inline implementation with imports from cf-core
3. Run the product's tests

The shim pattern:

```python
# myproduct/app/mything.py
from circuitforge_core.mymodule import get_my_thing as _base_get_my_thing
from .config import get_settings

def get_my_thing():
    settings = get_settings()
    return _base_get_my_thing(config=settings.mything_config)
```

## Licensing boundary

The module's license depends on what it does:

| Code | License |
|------|---------|
| Discovery, pipeline, data access | **MIT** |
| LLM inference, AI features, fine-tuned model access | **BSL 1.1** |
| Anything that would give SaaS competitors a free AI product | **BSL 1.1** |

When in doubt, BSL 1.1. See the [licensing guide](licensing.md) for the full decision tree.

## Versioning

cf-core uses semantic versioning. Adding a new module with a stable API is a **minor** version bump. Breaking an existing interface is a **major** bump and requires coordinated updates to all products.

Update `pyproject.toml` and `CHANGELOG.md` before merging.
