from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.eviction_engine import EvictionEngine
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()


class LeaseRequest(BaseModel):
    node_id: str
    gpu_id: int
    mb: int
    service: str
    priority: int = 2
    ttl_s: float = 0.0


class NodeRegisterRequest(BaseModel):
    node_id: str
    agent_url: str  # e.g. "http://10.1.10.71:7701"


def create_coordinator_app(
    lease_manager: LeaseManager,
    profile_registry: ProfileRegistry,
    agent_supervisor: AgentSupervisor,
) -> FastAPI:
    eviction_engine = EvictionEngine(lease_manager=lease_manager)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
        import asyncio
        task = asyncio.create_task(agent_supervisor.run_heartbeat_loop())
        yield
        agent_supervisor.stop()
        task.cancel()

    app = FastAPI(title="cf-orch-coordinator", lifespan=_lifespan)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/nodes")
    def get_nodes() -> dict[str, Any]:
        nodes = agent_supervisor.all_nodes()
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "agent_url": n.agent_url,
                    "last_heartbeat": n.last_heartbeat,
                    "gpus": [
                        {
                            "gpu_id": g.gpu_id,
                            "name": g.name,
                            "vram_total_mb": g.vram_total_mb,
                            "vram_used_mb": g.vram_used_mb,
                            "vram_free_mb": g.vram_free_mb,
                        }
                        for g in n.gpus
                    ],
                }
                for n in nodes
            ]
        }

    @app.post("/api/nodes")
    async def register_node(req: NodeRegisterRequest) -> dict[str, Any]:
        """Agents call this to self-register. Coordinator immediately polls for GPU info."""
        agent_supervisor.register(req.node_id, req.agent_url)
        await agent_supervisor.poll_agent(req.node_id)
        return {"registered": True, "node_id": req.node_id}

    @app.get("/api/profiles")
    def get_profiles() -> dict[str, Any]:
        return {
            "profiles": [
                {"name": p.name, "vram_total_mb": p.vram_total_mb}
                for p in profile_registry.list_public()
            ]
        }

    @app.get("/api/leases")
    def get_leases() -> dict[str, Any]:
        return {
            "leases": [
                {
                    "lease_id": lease.lease_id,
                    "node_id": lease.node_id,
                    "gpu_id": lease.gpu_id,
                    "mb_granted": lease.mb_granted,
                    "holder_service": lease.holder_service,
                    "priority": lease.priority,
                    "expires_at": lease.expires_at,
                }
                for lease in lease_manager.all_leases()
            ]
        }

    @app.post("/api/leases")
    async def request_lease(req: LeaseRequest) -> dict[str, Any]:
        node_info = agent_supervisor.get_node_info(req.node_id)
        if node_info is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown node_id {req.node_id!r} — node not registered",
            )
        agent_url = node_info.agent_url

        lease = await eviction_engine.request_lease(
            node_id=req.node_id,
            gpu_id=req.gpu_id,
            mb=req.mb,
            service=req.service,
            priority=req.priority,
            agent_url=agent_url,
            ttl_s=req.ttl_s,
        )
        if lease is None:
            raise HTTPException(
                status_code=503,
                detail="Insufficient VRAM — no eviction candidates available",
            )
        return {
            "lease": {
                "lease_id": lease.lease_id,
                "node_id": lease.node_id,
                "gpu_id": lease.gpu_id,
                "mb_granted": lease.mb_granted,
                "holder_service": lease.holder_service,
                "priority": lease.priority,
                "expires_at": lease.expires_at,
            }
        }

    @app.delete("/api/leases/{lease_id}")
    async def release_lease(lease_id: str) -> dict[str, Any]:
        released = await lease_manager.release(lease_id)
        if not released:
            raise HTTPException(status_code=404, detail=f"Lease {lease_id!r} not found")
        return {"released": True, "lease_id": lease_id}

    return app
