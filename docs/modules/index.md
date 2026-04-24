# Module Reference

All circuitforge-core modules live under the `circuitforge_core` package. Each is independently importable.

| Module | Import | Status | One-line summary |
|--------|--------|--------|-----------------|
| [db](db.md) | `circuitforge_core.db` | Stable | SQLite connection factory + migration runner |
| [llm](llm.md) | `circuitforge_core.llm` | Stable | LLM router with fallback chain |
| [tiers](tiers.md) | `circuitforge_core.tiers` | Stable | `@require_tier()` decorator, BYOK unlock |
| [config](config.md) | `circuitforge_core.config` | Stable | Env-driven settings, `.env` loader |
| [hardware](hardware.md) | `circuitforge_core.hardware` | Stable | GPU/CPU detection, VRAM profile generation |
| [documents](documents.md) | `circuitforge_core.documents` | Stable | Document ingestion → `StructuredDocument` |
| [affiliates](affiliates.md) | `circuitforge_core.affiliates` | Stable | `wrap_url()` with opt-out + BYOK user IDs |
| [preferences](preferences.md) | `circuitforge_core.preferences` | Stable | Dot-path preference store over local YAML |
| [tasks](tasks.md) | `circuitforge_core.tasks` | Stable | VRAM-aware background task scheduler |
| [manage](manage.md) | `circuitforge_core.manage` | Stable | `manage.sh` scaffolding, Docker + native |
| [resources](resources.md) | `circuitforge_core.resources` | Stable | VRAM allocation + eviction engine |
| [text](text.md) | `circuitforge_core.text` | Stable | Text normalization, chunking utilities |
| [stt](stt.md) | `circuitforge_core.stt` | Stub | Speech-to-text router |
| [tts](tts.md) | `circuitforge_core.tts` | Stub | Text-to-speech router |
| [pipeline](pipeline.md) | `circuitforge_core.pipeline` | Stub | `StagingDB` base class |
| [vision](vision.md) | `circuitforge_core.vision` | Stub | Vision router base class |
| [wizard](wizard.md) | `circuitforge_core.wizard` | Stub | First-run wizard base class |
