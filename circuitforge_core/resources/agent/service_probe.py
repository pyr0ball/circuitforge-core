"""
Probe running services to detect which models are currently loaded in VRAM.

Two probe strategies run together:

1. Well-known ports — always checked, regardless of who started the service.
   Catches ollama, vLLM, etc. running outside cf-orch management.

2. Managed services — services cf-orch started via ServiceManager.
   Checked on their configured host_port, deduplicates with well-known results.

Each service exposes a different introspection API:
  - vllm:   GET /v1/models  → {"data": [{"id": "<model-name>"}]}
  - ollama: GET /api/ps     → {"models": [{"name": "<model>", "size_vram": <bytes>}]}

ollama can have multiple models loaded simultaneously; each is reported as a
separate entry so the dashboard shows per-model residency.

The probe is best-effort: a timeout or connection refusal means model_name=None
but the service is still reported as resident.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from circuitforge_core.resources.profiles.schema import DockerSpec

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 2.0

# Well-known service ports probed on every heartbeat.
# key → (service_name, prober_key)
_WELL_KNOWN_PORTS: dict[int, str] = {
    11434: "ollama",
    8000:  "vllm",
    8080:  "vllm",  # common alt vLLM port
}


def _fetch_json(url: str) -> dict[str, Any] | None:
    """GET a URL and parse JSON; returns None on any error."""
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.debug("Probe %s: %s", url, exc)
        return None


def _probe_vllm(port: int) -> list[str]:
    data = _fetch_json(f"http://127.0.0.1:{port}/v1/models")
    if data and data.get("data"):
        return [m["id"] for m in data["data"] if m.get("id")]
    return []


def _probe_ollama(port: int) -> list[str]:
    # /api/ps lists models currently *loaded in memory*, not just downloaded.
    data = _fetch_json(f"http://127.0.0.1:{port}/api/ps")
    if data and data.get("models"):
        return [m["name"] for m in data["models"] if m.get("name")]
    return []


_PROBERS: dict[str, Any] = {
    "vllm":   _probe_vllm,
    "ollama": _probe_ollama,
}


def probe_all(service_manager: Any) -> list[dict[str, Any]]:
    """
    Probe all services — both well-known ports and cf-orch managed services.

    Returns a list of dicts: [{"service": str, "model_name": str | None}].
    Multiple loaded models in one service (e.g. two ollama models) each get
    their own entry, disambiguated as "ollama/0", "ollama/1", etc.
    """
    results: list[dict[str, Any]] = []
    seen_ports: set[int] = set()

    # ── 1. Well-known ports ──────────────────────────────────────────
    for port, service in _WELL_KNOWN_PORTS.items():
        prober = _PROBERS.get(service)
        if prober is None:
            continue
        models = prober(port)
        if not models:
            continue  # nothing on this port right now
        seen_ports.add(port)
        if len(models) == 1:
            results.append({"service": service, "model_name": models[0]})
        else:
            for i, model in enumerate(models):
                results.append({"service": f"{service}/{i}", "model_name": model})

    # ── 2. Managed services (cf-orch started) ───────────────────────
    if service_manager is not None:
        for service in service_manager.list_running():
            spec = service_manager._get_spec(service)
            if not isinstance(spec, DockerSpec):
                continue
            if spec.host_port in seen_ports:
                continue  # already captured by well-known probe
            prober = _PROBERS.get(service)
            if prober is None:
                results.append({"service": service, "model_name": None})
                continue
            models = prober(spec.host_port)
            seen_ports.add(spec.host_port)
            if not models:
                results.append({"service": service, "model_name": None})
            elif len(models) == 1:
                results.append({"service": service, "model_name": models[0]})
            else:
                for i, model in enumerate(models):
                    results.append({"service": f"{service}/{i}", "model_name": model})

    return results
