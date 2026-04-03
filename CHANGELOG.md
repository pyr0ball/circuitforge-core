# Changelog

All notable changes to `circuitforge-core` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
