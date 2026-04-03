from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from circuitforge_core.resources.agent.eviction_executor import EvictionExecutor
from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor
from circuitforge_core.resources.agent.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class EvictRequest(BaseModel):
    pid: int
    grace_period_s: float = 5.0


class ServiceStartRequest(BaseModel):
    gpu_id: int = 0
    params: dict[str, str] = {}


def create_agent_app(
    node_id: str,
    monitor: GpuMonitor | None = None,
    executor: EvictionExecutor | None = None,
    service_manager: ServiceManager | None = None,
) -> FastAPI:
    _monitor = monitor or GpuMonitor()
    _executor = executor or EvictionExecutor()

    app = FastAPI(title=f"cf-orch-agent [{node_id}]")

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

    @app.get("/resident-info")
    def resident_info() -> dict[str, Any]:
        """Return which models are currently loaded in each running managed service."""
        if service_manager is None:
            return {"residents": []}
        from circuitforge_core.resources.agent.service_probe import probe_all
        return {"residents": probe_all(service_manager)}

    if service_manager is not None:
        @app.get("/services")
        def list_services() -> dict:
            return {"running": service_manager.list_running()}

        @app.get("/services/{service}")
        def service_status(service: str) -> dict:
            running = service_manager.is_running(service)
            url = service_manager.get_url(service) if running else None
            return {"service": service, "running": running, "url": url}

        @app.post("/services/{service}/start")
        def start_service(service: str, req: ServiceStartRequest) -> dict:
            try:
                already_running = service_manager.is_running(service)
                url = service_manager.start(service, req.gpu_id, req.params)
                # adopted=True signals the coordinator to treat this instance as
                # immediately running rather than waiting for the probe loop.
                adopted = already_running and service_manager.is_running(service)
                return {"service": service, "url": url, "running": True, "adopted": adopted}
            except (ValueError, NotImplementedError) as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to start {service}: {exc}")

        @app.post("/services/{service}/stop")
        def stop_service(service: str) -> dict:
            stopped = service_manager.stop(service)
            return {"service": service, "stopped": stopped}

    return app
