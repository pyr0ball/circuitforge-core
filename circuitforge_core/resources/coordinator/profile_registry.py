# circuitforge_core/resources/coordinator/profile_registry.py
from __future__ import annotations

import logging
from pathlib import Path

from circuitforge_core.resources.models import GpuInfo
from circuitforge_core.resources.profiles.schema import GpuProfile, load_profile

_PUBLIC_DIR = Path(__file__).parent.parent / "profiles" / "public"

# VRAM thresholds for public profile selection (MB)
_PROFILE_THRESHOLDS = [
    (22000, "single-gpu-24gb"),
    (14000, "single-gpu-16gb"),
    (8000, "single-gpu-8gb"),
    (5500, "single-gpu-6gb"),
    (3500, "single-gpu-4gb"),
    (0, "single-gpu-2gb"),
]

_log = logging.getLogger(__name__)


class ProfileRegistry:
    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self._profiles: dict[str, GpuProfile] = {}
        self._load_dir(_PUBLIC_DIR)
        for d in (extra_dirs or []):
            if d.exists():
                self._load_dir(d)

    def _load_dir(self, directory: Path) -> None:
        for yaml_file in directory.glob("*.yaml"):
            try:
                profile = load_profile(yaml_file)
                self._profiles[profile.name] = profile
            except Exception as exc:
                _log.warning("Skipping %s: %s", yaml_file, exc)

    def load(self, path: Path) -> GpuProfile:
        profile = load_profile(path)
        self._profiles[profile.name] = profile
        return profile

    def list_public(self) -> list[GpuProfile]:
        return [
            p for p in self._profiles.values()
            if p.name.startswith("single-gpu-")
        ]

    def get(self, name: str) -> GpuProfile | None:
        return self._profiles.get(name)

    def auto_detect(self, gpus: list[GpuInfo]) -> GpuProfile:
        primary_vram = gpus[0].vram_total_mb if gpus else 0
        for threshold_mb, profile_name in _PROFILE_THRESHOLDS:
            if primary_vram >= threshold_mb:
                profile = self._profiles.get(profile_name)
                if profile:
                    return profile
        return self._profiles["single-gpu-2gb"]
