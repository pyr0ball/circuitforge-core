# circuitforge_core/resources/profiles/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

SUPPORTED_SCHEMA_VERSION = 1


class ServiceProfile(BaseModel):
    max_mb: int
    priority: int
    shared: bool = False
    max_concurrent: int = 1
    always_on: bool = False
    backend: str | None = None
    consumers: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


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
    version = raw.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r} in {path}. "
            f"Expected {SUPPORTED_SCHEMA_VERSION}."
        )
    return GpuProfile.model_validate(raw)
