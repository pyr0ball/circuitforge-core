# circuitforge-core

Shared scaffold for CircuitForge products.

## Modules

- `circuitforge_core.db` — SQLite connection factory and migration runner
- `circuitforge_core.llm` — LLM router with fallback chain
- `circuitforge_core.tiers` — Tier system with BYOK and local vision unlocks
- `circuitforge_core.config` — Env validation and .env loader
- `circuitforge_core.vision` — Vision router stub (v0.2+)
- `circuitforge_core.wizard` — First-run wizard base class stub
- `circuitforge_core.pipeline` — Staging queue stub (v0.2+)

## Install

```bash
pip install -e .
```

## License

BSL 1.1 — see LICENSE
