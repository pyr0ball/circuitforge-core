# Changelog

All notable changes to `circuitforge-core` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
