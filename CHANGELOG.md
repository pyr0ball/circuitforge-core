# Changelog

All notable changes to `circuitforge-core` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.13.0] — 2026-04-20

### Added

**`circuitforge_core.preferences.currency`** — per-user currency code preference + formatting utility (MIT, closes #52)

- `PREF_CURRENCY_CODE = "currency.code"` — shared store key; all products read from the same path
- `get_currency_code(user_id, store)` — priority fallback: store → `CURRENCY_DEFAULT` env var → `"USD"`
- `set_currency_code(currency_code, user_id, store)` — persists ISO 4217 code, uppercased
- `format_currency(amount, currency_code, locale="en_US")` — uses `babel.numbers.format_currency` when available; falls back to a built-in 30-currency symbol table (no hard babel dependency)
- Symbol table covers: USD, CAD, AUD, NZD, GBP, EUR, CHF, SEK/NOK/DKK, JPY, CNY, KRW, INR, BRL, MXN, ZAR, SGD, HKD, THB, PLN, CZK, HUF, RUB, TRY, ILS, AED, SAR, CLP, COP, ARS, VND, IDR, MYR, PHP
- JPY/KRW/HUF/CLP/COP/VND/IDR format with 0 decimal places per ISO 4217 minor-unit convention
- Exported from `circuitforge_core.preferences` as `currency` submodule
- 30 tests (preference store, env var fallback, format dispatch, symbol table, edge cases)

---

## [0.12.0] — 2026-04-20

### Added

**`circuitforge_core.job_quality`** — deterministic trust scorer for job listings (MIT, closes #48)

Pure signal processing module. No LLM calls, no network calls, no file I/O. Fully auditable and independently unit-testable per signal.

- `models.py` — `JobListing`, `JobEnrichment`, `SignalResult`, `JobQualityScore` (Pydantic)
- `signals.py` — 12 signal functions with weights: `listing_age` (0.25), `repost_detected` (0.25), `no_salary_transparency` (0.20), `always_open_pattern` (0.20), `staffing_agency` (0.15), `requirement_overload` (0.12), `layoff_news` (0.12), `jd_vagueness` (0.10), `ats_blackhole` (0.10), `high_applicant_count` (0.08), `poor_response_history` (0.08), `weekend_posted` (0.04)
- `scorer.py` — `score_job(listing, enrichment=None) -> JobQualityScore`; trust_score = 1 − clamp(sum(triggered weights), 0, 1); confidence = fraction of signals with available evidence
- Salary transparency enforcement for CO, CA, NY, WA, IL, MA; ATS blackhole detection (Lever, Greenhouse, Workday, iCIMS, Taleo)
- `ALL_SIGNALS` registry for iteration and extension
- 83 tests across models, signals (all 12 individually), and scorer — 100% pass

---

## [0.11.0] — 2026-04-20

### Added

**`circuitforge_core.audio`** — shared PCM and audio signal utilities (MIT, numpy-only, closes #50)

Pure signal processing module. No model weights, no HuggingFace, no torch dependency.

- `convert.py` — `pcm_to_float32`, `float32_to_pcm`, `bytes_to_float32` (int16 ↔ float32 with correct int16 asymmetry handling)
- `gate.py` — `is_silent`, `rms` (RMS energy gate; default 0.005 threshold extracted from cf-voice)
- `resample.py` — `resample` (scipy `resample_poly` when available; numpy linear interpolation fallback)
- `buffer.py` — `ChunkAccumulator` (window-based chunk collector with `flush`, `reset`, bounds enforcement)
- Replaces hand-rolled equivalents in cf-voice `stt.py` + `context.py`. Also consumed by Sparrow and Linnet.

**`circuitforge_core.musicgen` tests** — 21 tests covering mock backend, factory, and FastAPI app endpoints (closes #49). Module was already implemented; tests were the missing deliverable.

### Fixed

**SQLCipher PRAGMA injection** (closes #45) — `db/base.py` now uses `PRAGMA key=?` parameterized form instead of f-string interpolation. Regression tests added (skipped gracefully when `pysqlcipher3` is not installed).

**`circuitforge_core.text.app`** — early validation on empty `--model` argument: raises `ValueError` with a clear message before reaching the HuggingFace loader. Prevents the cryptic `HFValidationError` surfaced by cf-orch #46 when no model candidates were provided.

---

## [0.10.0] — 2026-04-12

### Added

**`circuitforge_core.community`** — shared community signal module (BSL 1.1, closes #44)

Provides the PostgreSQL-backed infrastructure for the cross-product community fine-tuning signal pipeline. Products write signals; the training pipeline reads them.

- `CommunityDB` — psycopg2 connection pool with `run_migrations()`. Picks up all `.sql` files from `circuitforge_core/community/migrations/` in filename order. Safe to call on every startup (idempotent `CREATE TABLE IF NOT EXISTS`).
- `CommunityPost` — frozen dataclass capturing a user-authored community post with a snapshot of the originating product item (`element_snapshot` as a tuple of key-value pairs for immutability).
- `SharedStore` — base class for product-specific community stores. Provides typed `pg_read()` and `pg_write()` helpers that products subclass without re-implementing connection management.
- Migration 001: `community_posts` schema (id, product, item_id, pseudonym, title, body, element_snapshot JSONB, created_at).
- Migration 002: `community_reactions` stub (post_id FK, pseudonym, reaction_type, created_at).
- `psycopg2-binary` added to `[community]` optional extras in `pyproject.toml`.
- All community classes exported from `circuitforge_core.community`.

---

## [0.9.0] — 2026-04-10

### Added

**`circuitforge_core.text`** — OpenAI-compatible `/v1/chat/completions` endpoint and pipeline crystallization engine.

**`circuitforge_core.pipeline`** — multimodal pipeline with staged output crystallization. Products queue draft outputs for human review before committing.

**`circuitforge_core.stt`** — speech-to-text module. `FasterWhisperBackend` for local transcription via `faster-whisper`. Managed FastAPI app mountable in any product.

**`circuitforge_core.tts`** — text-to-speech module. `ChatterboxTurbo` backend for local synthesis. Managed FastAPI app.

**Accessibility preferences** — `preferences` module extended with structured accessibility fields (motion reduction, high contrast, font size, focus highlight) under `accessibility.*` key path.

**LLM output corrections router** — `make_corrections_router()` for collecting LLM output corrections in any product. Stores corrections in product SQLite for future fine-tuning.

---

## [0.8.0] — 2026-04-08

### Added

**`circuitforge_core.vision`** — cf-vision managed service shim. Routes vision inference requests to a local cf-vision worker (moondream2 / SigLIP). Closes #43.

**`circuitforge_core.api.feedback`** — `make_feedback_router()` shared Forgejo issue-filing router. Products mount it under `/api/feedback`; requires `FORGEJO_API_TOKEN`. Closes #30.

**License validation** — `CF_LICENSE_KEY` validation via Heimdall REST API. Products call `validate_license(key, product)` to gate premium features. Closes #26.

---

## [0.7.0] — 2026-04-04

### Added

**`circuitforge_core.affiliates`** — affiliate link wrapping module (closes #21)
- `wrap_url(url, retailer, user_id, get_preference)` — resolution order: opt-out → BYOK → CF env var → plain URL
- `AffiliateProgram` frozen dataclass + `register_program()` / `get_program()` registry
- Built-in programs: eBay Partner Network (`EBAY_AFFILIATE_CAMPAIGN_ID`), Amazon Associates (`AMAZON_ASSOCIATES_TAG`)
- `get_disclosure_text(retailer)` — per-retailer tooltip copy + `BANNER_COPY` first-encounter constants
- `get_preference` callable injection for opt-out + BYOK without hard-wiring a storage backend

**`circuitforge_core.preferences`** — preference persistence helpers (closes #22 self-hosted path)
- `LocalFileStore` — YAML-backed single-user preference store (`~/.config/circuitforge/preferences.yaml`)
- `get_user_preference(user_id, path, default, store)` + `set_user_preference(user_id, path, value, store)`
- `PreferenceStore` protocol — Heimdall cloud backend to follow once Heimdall#5 lands
- Dot-path utilities `get_path` / `set_path` (immutable nested dict read/write)

---

## [0.5.0] — 2026-04-02

### Added

**`circuitforge_core.manage` — cross-platform product manager** (closes #6)

Replaces bash-only `manage.sh` across all products. Works on Linux, macOS, and Windows natively — no WSL2 or Docker required.

- **`ManageConfig`**: reads `manage.toml` from the product root (TOML via stdlib `tomllib`). Falls back to directory name when no config file is present — Docker-only products need zero configuration.
- **Docker mode** (`DockerManager`): wraps `docker compose` (v2 plugin) or `docker-compose` (v1). Auto-detected when Docker is available and a compose file exists. Commands: `start`, `stop`, `restart`, `status`, `logs`, `build`.
- **Native mode** (`NativeManager`): PID-file process management with `platformdirs`-based paths (`AppData` on Windows, `~/.local/share` on Linux/macOS). Cross-platform kill (SIGTERM→SIGKILL on Unix, `taskkill /F` on Windows). Log tailing via polling — no `tail -f`, works everywhere.
- **CLI** (`typer`): `start`, `stop`, `restart`, `status`, `logs`, `build`, `open`, `install-shims`. `--mode auto|docker|native` override.
- **`install-shims`**: writes `manage.sh` (bash, +x) and `manage.ps1` (PowerShell) into the product directory, plus `manage.toml.example`.
- **Entry points**: `python -m circuitforge_core.manage` and `cf-manage` console script.
- **`pyproject.toml`**: `[manage]` optional extras group (`platformdirs`, `typer`).

---

## [0.4.0] — 2026-04-02

### Added

**Agent watchdog — coordinator-restart reconnect** (closes #15)
- `NodeStore`: SQLite persistence for known agent nodes (`~/.local/share/circuitforge/cf-orch-nodes.db`); `upsert` on every registration, `prune_stale` removes nodes unseen for 30+ days
- `AgentSupervisor.restore_from_store()`: reloads all previously-known nodes on coordinator startup; nodes start `offline=False` and come online within one heartbeat cycle (~10 s) without touching the agent processes
- `AgentSupervisor.register()` now persists to `NodeStore` on every call
- Agent CLI: one-shot registration replaced with a persistent 30 s reconnect loop (daemon thread); coordinator restart → remote nodes (Navi, Strahl, etc.) reappear automatically with no manual intervention

**Ollama adopt-if-running + configurable health path** (closes #16)
- `ProcessSpec.adopt` (`bool`, default `False`): when `True`, `ServiceManager.start()` probes the health endpoint first and claims the already-running process rather than spawning a new one — designed for system daemons like Ollama
- `ProcessSpec.health_path` (`str`, default `"/health"`): configurable health probe path; Ollama uses `/api/tags`
- `ServiceManager._probe_health()`: shared urllib health check used by both `start()` and `is_running()` for adopt services
- Agent `/services/{service}/start` response includes `adopted: true` when the service was claimed rather than started; coordinator sets instance state to `running` immediately (skips probe loop wait)
- `ServiceInstance.health_path` field; `upsert_instance(health_path=)` kwarg
- Coordinator probe loop uses `inst.health_path` instead of hardcoded `/health`
- `_get_health_path()` helper looks up the ProcessSpec health path from the profile registry
- All GPU profiles (2/4/6/8/16/24 GB + cpu-16/32 GB): `ollama` service now has a `managed:` block with `adopt: true`, `health_path: /api/tags`, port 11434

---

## [0.3.0] — 2026-04-02

### Added

**Hardware module** (`circuitforge_core.hardware`) — closes #5
- `detect_hardware()`: probes nvidia-smi / rocm-smi / Apple system_profiler / CPU fallback → `HardwareSpec`
- `select_tier(vram_mb)`: maps physical VRAM to a named `VramTier` (CPU / 2 / 4 / 6 / 8 / 16 / 24 GB)
- `generate_profile(spec)`: converts a `HardwareSpec` + service URLs → `LLMConfig` (llm.yaml-compatible)
- `HardwareSpec`, `LLMBackendConfig`, `LLMConfig` dataclasses

**cf-docuvision service** (`circuitforge_core.resources.docuvision`) — closes #8
- FastAPI HTTP service wrapping ByteDance/Dolphin-v2 (Qwen2.5-VL backbone, ~8 GB VRAM)
- `POST /extract`: accepts `image_b64` or `image_path` + `hint` (auto / table / text / form) → `ExtractResponse`
- Lazy model loading — model stays unloaded until first request
- JSON-structured output with 21 element types; plain-text fallback when model returns unstructured output
- `ProcessSpec` managed blocks wired into all four GPU profiles (6 / 8 / 16 / 24 GB)
- `--gpu-id` flag respected via `CUDA_VISIBLE_DEVICES`

**Documents module** (`circuitforge_core.documents`) — closes #7
- `ingest(image_bytes, hint) → StructuredDocument` — single call for all consumers
- Primary path: cf-docuvision HTTP service; automatic fallback to `LLMRouter` vision; graceful empty doc on total failure
- `StructuredDocument`, `Element`, `ParsedTable` frozen dataclasses with `.headings` / `.paragraphs` convenience properties
- `CF_DOCUVISION_URL` env var for service URL override
- `DocuvisionClient`: reusable HTTP client for cf-docuvision with `is_healthy()` probe

**Coordinator probe loop tests** — closes #13
- 4 async tests for `_run_instance_probe_loop`: healthy transition, timeout eviction, state cleanup, no-URL guard

---

## [0.2.0] — 2026-04-02

### Added

**Orchestrator — auto service lifecycle**
- `ServiceRegistry`: in-memory allocation tracker with state machine (`starting → running → idle → stopped`)
- `NodeSelector`: warm-first GPU scoring — prefers nodes already running the requested model, falls back to highest free VRAM
- `/api/services/{service}/allocate` coordinator endpoint: auto-selects best node, starts the llm_server process via agent, returns URL
- `CFOrchClient`: sync + async context managers for coordinator allocation/release
- Idle sweep in `AgentSupervisor`: stops instances that have been idle longer than `idle_stop_after_s` (default 600 s for vllm slot)
- Background health probe loop: coordinator polls all `starting` instances every 5 s via `GET /health`; promotes to `running` on 200, marks `stopped` after 300 s timeout (closes #10)
- Services table in coordinator dashboard HTML
- `idle_stop_after_s` field in service profiles

**LLM Router**
- cf-orch allocation support in `LLMRouter` backends
- VRAM lease acquisition/release wired through scheduler batch workers

**Scheduler**
- cf-orch VRAM lease per batch worker — prevents over-subscription
- `join()` on batch worker threads during shutdown

**HF inference server** (`llm_server.py`)
- Generic HuggingFace `transformers` inference server replacing Ouro/vllm-Docker-specific code
- `ProcessSpec` wiring in agent `service_manager.py`
- Handles transformers 5.x `BatchEncoding` return type from `apply_chat_template`
- Uses `dtype=` kwarg (replaces deprecated `torch_dtype=`)

### Fixed

- VRAM pre-flight threshold tightened: coordinator and `NodeSelector` now require full `service_max_mb` free (was `max_mb // 2`), preventing instances from starting on GPUs with insufficient headroom (closes #11 / related)
- `ServiceInstance` now seeded correctly on first `/allocate` call
- TTL sweep, immutability, and service-scoped release correctness in allocation path
- Coordinator logger added for allocation path visibility

### Changed

- Removed Ouro/vllm-Docker specifics from llm_server — now a generic HF inference endpoint

---

## [0.1.0] — 2026-03-01

### Added

- Package scaffold (`circuitforge_core`)
- DB base connection and migration runner
- Generalised tier system with BYOK (bring your own key) and local-vision unlocks
- LLM router extracted from Peregrine (fallback chain, vision-aware, BYOK support)
- Config module and vision router stub
- cf-orch orchestrator: coordinator (port 7700) + agent (port 7701)
- Agent registration + VRAM lease wiring
- Coordinator dashboard (HTML)
