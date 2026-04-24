# circuitforge-core

Shared scaffold for all CircuitForge products. Every product in the menagerie depends on it via editable install.

```bash
pip install -e ../circuitforge-core
# or inside conda:
conda run -n cf pip install -e ../circuitforge-core
```

---

## What it provides

circuitforge-core gives every product the same foundation so patterns proven in one product propagate to all others automatically. The 17 modules cover the full stack from database access to LLM routing to tier gates.

```
circuitforge_core/
├── db/          SQLite factory + migration runner
├── llm/         LLM router with fallback chain
├── tiers/       Tier gates — free / paid / premium / ultra
├── config/      Env-driven settings + .env loader
├── hardware/    GPU/CPU detection + VRAM profile generation
├── documents/   PDF, DOCX, image OCR → StructuredDocument
├── affiliates/  URL wrapping with opt-out + BYOK user IDs
├── preferences/ Per-user YAML preference store (dot-path API)
├── tasks/       VRAM-aware background task scheduler
├── manage/      Cross-platform process manager (Docker + native)
├── resources/   VRAM allocation + eviction engine
├── text/        Text processing utilities
├── stt/         Speech-to-text router (stub)
├── tts/         Text-to-speech router (stub)
├── pipeline/    Staging queue base — StagingDB (stub)
├── vision/      Vision router base class (stub)
└── wizard/      First-run wizard base class (stub)
```

---

## Module status

| Module | Status | Purpose |
|--------|--------|---------|
| `db` | Stable | SQLite connection factory, migration runner |
| `llm` | Stable | LLM fallback router (Ollama, vLLM, Anthropic, OpenAI-compatible) |
| `tiers` | Stable | `@require_tier()` decorator, BYOK unlock logic |
| `config` | Stable | Env-driven settings, `.env` loader |
| `hardware` | Stable | GPU enumeration, VRAM tier profiling |
| `documents` | Stable | PDF/DOCX/image ingestion → `StructuredDocument` |
| `affiliates` | Stable | `wrap_url()` with opt-out and BYOK user IDs |
| `preferences` | Stable | Dot-path `get()`/`set()` over local YAML; pluggable backend |
| `tasks` | Stable | `TaskScheduler` — VRAM-aware slot management |
| `manage` | Stable | `manage.sh` scaffolding for Docker and native processes |
| `resources` | Stable | VRAM allocation, eviction engine, GPU profile registry |
| `text` | Stable | Text normalization, truncation, chunking utilities |
| `stt` | Stub | Speech-to-text router (planned: whisper.cpp / faster-whisper) |
| `tts` | Stub | Text-to-speech router (planned: piper / espeak) |
| `pipeline` | Stub | `StagingDB` base — products provide concrete schema |
| `vision` | Stub | Vision router base class (moondream2 / Claude dispatch) |
| `wizard` | Stub | `BaseWizard` — products subclass for first-run setup |

---

## Version

**v0.9.0** — MIT licensed for discovery/pipeline layers, BSL 1.1 for AI features.

See the [developer guide](developer/adding-module.md) to add a new module.
