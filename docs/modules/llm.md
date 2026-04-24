# llm

LLM router with a configurable fallback chain. Abstracts over Ollama, vLLM, Anthropic, and any OpenAI-compatible backend. Products never talk to a specific LLM backend directly.

```python
from circuitforge_core.llm import LLMRouter
```

## Design principle

The router implements "local inference first." Cloud backends sit at the end of the fallback chain. A product configured with only Ollama will never silently fall through to a paid API.

## Configuration

The router reads `config/llm.yaml` from the product's working directory (or the path passed to the constructor). Each product maintains its own `llm.yaml`; cf-core provides the router, not the config.

```yaml
# config/llm.yaml example
fallback_order:
  - ollama
  - vllm
  - anthropic

ollama:
  enabled: true
  base_url: http://localhost:11434
  model: llama3.2:3b

vllm:
  enabled: false
  base_url: http://localhost:8000

anthropic:
  enabled: false
  api_key_env: ANTHROPIC_API_KEY
```

## API

### `LLMRouter(config_path=None, cloud_mode=False)`

Instantiate the router. In most products, instantiation happens inside a shim that injects product-specific config resolution.

### `router.complete(prompt, system=None, images=None, fallback_order=None) -> str`

Send a completion request. Tries backends in order; falls through on error or unavailability.

```python
router = LLMRouter()
response = router.complete(
    prompt="Summarize this recipe in one sentence.",
    system="You are a cooking assistant.",
)
```

Pass `images: list[str]` (base64-encoded) for vision requests — non-vision backends are automatically skipped when images are present.

Pass `fallback_order=["vllm", "anthropic"]` to override the config chain for a specific call (useful for task-specific routing).

### `router.stream(prompt, system=None) -> Iterator[str]`

Streaming variant. Yields token chunks as they arrive. Not all backends support streaming; the router logs a warning and falls back to a non-streaming backend if needed.

## Shim requirement

!!! warning "Always use the product shim"
    Scripts and endpoints must import `LLMRouter` from the product shim (`scripts/llm_router.py` or `app/llm_router.py`), never directly from `circuitforge_core.llm.router`. The shim handles tri-level config resolution (env vars override config file overrides defaults) and cloud mode wiring. Bypassing it breaks cloud deployments silently.

## Backends

| Backend | Type | Notes |
|---------|------|-------|
| `ollama` | Local | Preferred default; model names from `config/llm.yaml` |
| `vllm` | Local GPU | For high-throughput or large models |
| `anthropic` | Cloud | Requires `ANTHROPIC_API_KEY` env var |
| `openai` | Cloud | Any OpenAI-compatible endpoint |
| `claude_code` | Local wrapper | claude-bridge OpenAI-compatible wrapper on :3009 |

## Vision routing

When images are included in a `complete()` call, the router checks each backend's vision capability before trying it. Configure vision priority separately:

```yaml
vision_fallback_order:
  - vision_service   # local moondream2 via FastAPI on :8002
  - claude_code
  - anthropic
```
