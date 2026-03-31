from __future__ import annotations

import logging
import subprocess

from circuitforge_core.resources.models import GpuInfo

logger = logging.getLogger(__name__)

_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]


class GpuMonitor:
    def poll(self) -> list[GpuInfo]:
        try:
            result = subprocess.run(
                _NVIDIA_SMI_CMD,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("nvidia-smi unavailable: %s", exc)
            return []

        if result.returncode != 0:
            logger.warning("nvidia-smi exited %d", result.returncode)
            return []

        return self._parse(result.stdout)

    def _parse(self, output: str) -> list[GpuInfo]:
        gpus: list[GpuInfo] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                gpus.append(GpuInfo(
                    gpu_id=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    vram_free_mb=int(parts[4]),
                ))
            except ValueError:
                logger.debug("Skipping malformed nvidia-smi line: %r", line)
        return gpus
