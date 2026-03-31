from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from circuitforge_core.resources.agent.eviction_executor import EvictionExecutor
from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)


class EvictRequest(BaseModel):
    pid: int
    grace_period_s: float = 5.0


def create_agent_app(
    node_id: str,
    monitor: GpuMonitor | None = None,
    executor: EvictionExecutor | None = None,
) -> FastAPI:
    _monitor = monitor or GpuMonitor()
    _executor = executor or EvictionExecutor()

    app = FastAPI(title=f"cforch-agent [{node_id}]")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "node_id": node_id}

    @app.get("/gpu-info")
    def gpu_info() -> dict[str, Any]:
        gpus = _monitor.poll()
        return {
            "node_id": node_id,
            "gpus": [
                {
                    "gpu_id": g.gpu_id,
                    "name": g.name,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_used_mb": g.vram_used_mb,
                    "vram_free_mb": g.vram_free_mb,
                }
                for g in gpus
            ],
        }

    @app.post("/evict")
    def evict(req: EvictRequest) -> dict[str, Any]:
        result = _executor.evict_pid(pid=req.pid, grace_period_s=req.grace_period_s)
        return {
            "success": result.success,
            "method": result.method,
            "message": result.message,
        }

    return app
