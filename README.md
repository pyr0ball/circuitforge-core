# circuitforge-core

Shared scaffold for CircuitForge products.

**Current version: 0.7.0**

## Modules

### Implemented

- `circuitforge_core.db` — SQLite connection factory and migration runner
- `circuitforge_core.llm` — LLM router with fallback chain (Ollama, vLLM, Anthropic, OpenAI-compatible)
- `circuitforge_core.tiers` — Tier system with BYOK and local vision unlocks
- `circuitforge_core.config` — Env validation and .env loader
- `circuitforge_core.hardware` — Hardware detection and LLM backend profile generation (VRAM tiers, GPU/CPU auto-select)
- `circuitforge_core.documents` — Document ingestion pipeline: PDF, DOCX, and image OCR → `StructuredDocument`
- `circuitforge_core.affiliates` — Affiliate URL wrapping with opt-out, BYOK user IDs, and CF env-var fallback (`wrap_url`)
- `circuitforge_core.preferences` — User preference store (local YAML file, pluggable backend); dot-path get/set API
- `circuitforge_core.tasks` — VRAM-aware LLM task scheduler; shared slot manager across services (`TaskScheduler`)
- `circuitforge_core.manage` — Cross-platform product process manager (Docker and native modes)
- `circuitforge_core.resources` — Resource coordinator and agent: VRAM allocation, eviction engine, GPU profile registry

### Stubs (in-tree, not yet implemented)

- `circuitforge_core.vision` — Vision router base class (planned: moondream2 / Claude vision dispatch)
- `circuitforge_core.wizard` — First-run wizard base class (products subclass `BaseWizard`)
- `circuitforge_core.pipeline` — Staging queue base (`StagingDB`; products provide concrete schema)

## Install

```bash
pip install -e .
```

## License

BSL 1.1 — see LICENSE
