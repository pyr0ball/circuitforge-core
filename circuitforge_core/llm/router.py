"""
LLM abstraction layer with priority fallback chain.
Reads config from ~/.config/circuitforge/llm.yaml.
Tries backends in order; falls back on any error.
"""
import logging
import os
import yaml
import requests
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".config" / "circuitforge" / "llm.yaml"


class LLMRouter:
    def __init__(self, config_path: Path = CONFIG_PATH):
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} not found. "
                "Copy the llm.yaml.example to ~/.config/circuitforge/llm.yaml and configure your LLM backends."
            )
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _is_reachable(self, base_url: str) -> bool:
        """Quick health-check ping. Returns True if backend is up."""
        health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
        try:
            resp = requests.get(health_url, timeout=2)
            return resp.status_code < 500
        except Exception:
            return False

    def _resolve_model(self, client: OpenAI, model: str) -> str:
        """Resolve __auto__ to the first model served by vLLM."""
        if model != "__auto__":
            return model
        models = client.models.list()
        return models.data[0].id

    def _try_cf_orch_alloc(self, backend: dict) -> "tuple | None":
        """
        If backend config has a cf_orch block and CF_ORCH_URL is set (env takes
        precedence over yaml url), allocate via cf-orch and return (ctx, alloc).
        Returns None if not configured or allocation fails.
        Caller MUST call ctx.__exit__(None, None, None) in a finally block.
        """
        import os
        orch_cfg = backend.get("cf_orch")
        if not orch_cfg:
            return None
        orch_url = os.environ.get("CF_ORCH_URL", orch_cfg.get("url", ""))
        if not orch_url:
            return None
        try:
            from circuitforge_core.resources.client import CFOrchClient
            client = CFOrchClient(orch_url)
            service = orch_cfg.get("service", "vllm")
            candidates = orch_cfg.get("model_candidates", [])
            ttl_s = float(orch_cfg.get("ttl_s", 3600.0))
            ctx = client.allocate(service, model_candidates=candidates, ttl_s=ttl_s, caller="llm-router")
            alloc = ctx.__enter__()
            return (ctx, alloc)
        except Exception as exc:
            logger.warning("[LLMRouter] cf_orch allocation failed, using base_url directly: %s", exc)
            return None

    def complete(self, prompt: str, system: str | None = None,
                 model_override: str | None = None,
                 fallback_order: list[str] | None = None,
                 images: list[str] | None = None,
                 max_tokens: int | None = None) -> str:
        """
        Generate a completion. Tries each backend in fallback_order.

        model_override: when set, replaces the configured model for
        openai_compat backends (e.g. pass a research-specific ollama model).
        fallback_order: when set, overrides config fallback_order for this
        call (e.g. pass config["research_fallback_order"] for research tasks).
        images: optional list of base64-encoded PNG/JPG strings. When provided,
        backends without supports_images=true are skipped. vision_service backends
        are only tried when images is provided.
        Raises RuntimeError if all backends are exhausted.
        """
        if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "AI inference is disabled in the public demo. "
                "Run your own instance to use AI features."
            )
        order = fallback_order if fallback_order is not None else self.config["fallback_order"]
        for name in order:
            backend = self.config["backends"][name]

            if not backend.get("enabled", True):
                print(f"[LLMRouter] {name}: disabled, skipping")
                continue

            supports_images = backend.get("supports_images", False)
            is_vision_service = backend["type"] == "vision_service"

            # vision_service only used when images provided
            if is_vision_service and not images:
                print(f"[LLMRouter] {name}: vision_service skipped (no images)")
                continue

            # non-vision backends skipped when images provided and they don't support it
            if images and not supports_images and not is_vision_service:
                print(f"[LLMRouter] {name}: no image support, skipping")
                continue

            if is_vision_service:
                if not self._is_reachable(backend["base_url"]):
                    print(f"[LLMRouter] {name}: unreachable, skipping")
                    continue
                try:
                    resp = requests.post(
                        backend["base_url"].rstrip("/") + "/analyze",
                        json={
                            "prompt": prompt,
                            "image_base64": images[0] if images else "",
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    print(f"[LLMRouter] Used backend: {name} (vision_service)")
                    return resp.json()["text"]
                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue

            elif backend["type"] == "openai_compat":
                if not self._is_reachable(backend["base_url"]):
                    print(f"[LLMRouter] {name}: unreachable, skipping")
                    continue
                # --- cf_orch: optionally override base_url with coordinator-allocated URL ---
                orch_ctx = orch_alloc = None
                orch_result = self._try_cf_orch_alloc(backend)
                if orch_result is not None:
                    orch_ctx, orch_alloc = orch_result
                    backend = {**backend, "base_url": orch_alloc.url + "/v1"}
                try:
                    client = OpenAI(
                        base_url=backend["base_url"],
                        api_key=backend.get("api_key") or "any",
                    )
                    raw_model = model_override or backend["model"]
                    model = self._resolve_model(client, raw_model)
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    if images and supports_images:
                        content = [{"type": "text", "text": prompt}]
                        for img in images:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img}"},
                            })
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": prompt})

                    create_kwargs: dict = {"model": model, "messages": messages}
                    if max_tokens is not None:
                        create_kwargs["max_tokens"] = max_tokens
                    resp = client.chat.completions.create(**create_kwargs)
                    print(f"[LLMRouter] Used backend: {name} ({model})")
                    return resp.choices[0].message.content

                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue
                finally:
                    if orch_ctx is not None:
                        try:
                            orch_ctx.__exit__(None, None, None)
                        except Exception:
                            pass

            elif backend["type"] == "anthropic":
                api_key = os.environ.get(backend["api_key_env"], "")
                if not api_key:
                    print(f"[LLMRouter] {name}: {backend['api_key_env']} not set, skipping")
                    continue
                try:
                    import anthropic as _anthropic
                    client = _anthropic.Anthropic(api_key=api_key)
                    if images and supports_images:
                        content = []
                        for img in images:
                            content.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": img},
                            })
                        content.append({"type": "text", "text": prompt})
                    else:
                        content = prompt
                    kwargs: dict = {
                        "model": backend["model"],
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": content}],
                    }
                    if system:
                        kwargs["system"] = system
                    msg = client.messages.create(**kwargs)
                    print(f"[LLMRouter] Used backend: {name}")
                    return msg.content[0].text
                except Exception as e:
                    print(f"[LLMRouter] {name}: error — {e}, trying next")
                    continue

        raise RuntimeError("All LLM backends exhausted")


# Module-level singleton for convenience
_router: LLMRouter | None = None


def complete(prompt: str, system: str | None = None) -> str:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router.complete(prompt, system)
