# circuitforge_core/resources/profiles/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

SUPPORTED_SCHEMA_VERSION = 1


class DockerSpec(BaseModel):
    """Spec for a Docker-managed service."""

    image: str
    port: int
    host_port: int
    command_template: str = ""
    volumes: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    runtime: str = "nvidia"
    ipc: str = "host"

    model_config = {"frozen": True}


class ProcessSpec(BaseModel):
    """Spec for a process-managed service (non-Docker, e.g. conda env)."""

    exec_path: str
    args_template: str = ""
    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    port: int = 0
    host_port: int = 0
    # adopt=True: if the service is already listening on host_port, claim it rather
    # than spawning a new process (useful for system daemons like Ollama).
    adopt: bool = False
    # Override the health probe path; defaults to /health (Ollama uses /api/tags).
    health_path: str = "/health"

    model_config = {"frozen": True}


class ServiceProfile(BaseModel):
    max_mb: int
    priority: int
    shared: bool = False
    max_concurrent: int = 1
    always_on: bool = False
    idle_stop_after_s: int = 0
    backend: str | None = None
    consumers: list[str] = Field(default_factory=list)
    managed: DockerSpec | ProcessSpec | None = None

    model_config = {"frozen": True}

    @model_validator(mode="before")
    @classmethod
    def _parse_managed(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        raw = values.get("managed")
        if raw is None:
            return values
        if not isinstance(raw, dict):
            return values
        spec_type = raw.get("type")
        managed_fields = {k: v for k, v in raw.items() if k != "type"}
        if spec_type == "docker":
            values["managed"] = DockerSpec(**managed_fields)
        elif spec_type == "process":
            values["managed"] = ProcessSpec(**managed_fields)
        else:
            raise ValueError(f"Unknown managed service type: {spec_type!r}")
        return values


class GpuNodeEntry(BaseModel):
    id: int
    vram_mb: int
    role: str
    card: str = "unknown"
    always_on: bool = False
    services: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class NodeProfile(BaseModel):
    gpus: list[GpuNodeEntry]
    agent_url: str | None = None
    nas_mount: str | None = None

    model_config = {"frozen": True}


class GpuProfile(BaseModel):
    schema_version: int
    name: str
    vram_total_mb: int | None = None
    eviction_timeout_s: float = 10.0
    services: dict[str, ServiceProfile] = Field(default_factory=dict)
    model_size_hints: dict[str, str] = Field(default_factory=dict)
    nodes: dict[str, NodeProfile] = Field(default_factory=dict)

    model_config = {"frozen": True}


def load_profile(path: Path) -> GpuProfile:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Profile file {path} must be a YAML mapping, got {type(raw).__name__}")
    version = raw.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r} in {path}. "
            f"Expected {SUPPORTED_SCHEMA_VERSION}."
        )
    return GpuProfile.model_validate(raw)
