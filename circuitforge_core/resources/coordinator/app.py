from __future__ import annotations

import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.eviction_engine import EvictionEngine
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.node_selector import select_node
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


class ServiceEnsureRequest(BaseModel):
    node_id: str
    gpu_id: int = 0
    params: dict[str, str] = {}
    ttl_s: float = 3600.0
    # Ordered list of model names to try; falls back down the list if VRAM is tight.
    # The "model" key in params is used if this list is empty.
    model_candidates: list[str] = []


class ServiceAllocateRequest(BaseModel):
    model_candidates: list[str] = []
    gpu_id: int | None = None
    params: dict[str, str] = {}
    ttl_s: float = 3600.0
    caller: str = ""


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

    @app.get("/api/resident")
    def get_residents() -> dict[str, Any]:
        return {
            "residents": [
                {
                    "service": r.service,
                    "node_id": r.node_id,
                    "model_name": r.model_name,
                    "first_seen": r.first_seen,
                }
                for r in lease_manager.all_residents()
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

    @app.post("/api/services/{service}/ensure")
    async def ensure_service(service: str, req: ServiceEnsureRequest) -> dict[str, Any]:
        """
        Ensure a managed service is running on the given node.

        If model_candidates is provided, tries each model in order, skipping any
        that exceed the live free VRAM on the target GPU. Falls back down the list
        until one succeeds. The selected model is returned in the response.
        """
        import httpx

        node_info = agent_supervisor.get_node_info(req.node_id)
        if node_info is None:
            raise HTTPException(422, detail=f"Unknown node_id {req.node_id!r}")

        # Resolve candidate list — fall back to params["model"] if not specified.
        candidates: list[str] = req.model_candidates or (
            [req.params["model"]] if "model" in req.params else []
        )
        if not candidates:
            raise HTTPException(422, detail="No model specified: set params.model or model_candidates")

        # Live free VRAM on the target GPU (used for pre-flight filtering).
        gpu = next((g for g in node_info.gpus if g.gpu_id == req.gpu_id), None)
        free_mb = gpu.vram_free_mb if gpu else 0

        # Profile max_mb for the service gives us the VRAM ceiling for this slot.
        # Models larger than free_mb are skipped before we even try to start them.
        # We use model file size as a rough proxy — skip if free_mb < half of max_mb,
        # since a fully-loaded model typically needs ~50-80% of its param size in VRAM.
        service_max_mb = 0
        for p in profile_registry.list_public():
            svc = p.services.get(service)
            if svc:
                service_max_mb = svc.max_mb
                break

        last_error: str = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            for model in candidates:
                params_with_model = {**req.params, "model": model}
                try:
                    start_resp = await client.post(
                        f"{node_info.agent_url}/services/{service}/start",
                        json={"gpu_id": req.gpu_id, "params": params_with_model},
                    )
                    if start_resp.is_success:
                        data = start_resp.json()
                        return {
                            "service": service,
                            "node_id": req.node_id,
                            "gpu_id": req.gpu_id,
                            "model": model,
                            "url": data.get("url"),
                            "running": data.get("running", False),
                        }
                    last_error = start_resp.text
                except httpx.HTTPError as exc:
                    raise HTTPException(502, detail=f"Agent unreachable: {exc}")

        raise HTTPException(
            503,
            detail=f"All model candidates exhausted for {service!r}. Last error: {last_error}",
        )

    @app.post("/api/services/{service}/allocate")
    async def allocate_service(service: str, req: ServiceAllocateRequest) -> dict[str, Any]:
        """
        Allocate a managed service — coordinator picks the best node automatically.
        Returns a URL + allocation_id. (Allocation not tracked server-side until Phase 2.)
        """
        import httpx

        if not req.model_candidates:
            raise HTTPException(422, detail="model_candidates must be non-empty")

        if req.gpu_id is None:
            # Validate the service is known before attempting node selection.
            known = any(
                service in p.services
                for p in profile_registry.list_public()
            )
            if not known:
                raise HTTPException(422, detail=f"Unknown service {service!r} — not in any profile")

            online = agent_supervisor.online_agents()
            placement = select_node(online, service, profile_registry, lease_manager.resident_keys())
            if placement is None:
                raise HTTPException(
                    503,
                    detail=f"No online node has capacity for service {service!r}",
                )
            node_id, gpu_id = placement
        else:
            online = agent_supervisor.online_agents()
            node_id = next(
                (nid for nid, rec in online.items()
                 if any(g.gpu_id == req.gpu_id for g in rec.gpus)),
                None,
            )
            if node_id is None:
                raise HTTPException(422, detail=f"No online node has gpu_id={req.gpu_id}")
            gpu_id = req.gpu_id

        node_info = agent_supervisor.get_node_info(node_id)
        if node_info is None:
            raise HTTPException(422, detail=f"Node {node_id!r} not found")

        warm = f"{node_id}:{service}" in lease_manager.resident_keys()

        async with httpx.AsyncClient(timeout=120.0) as client:
            last_error = ""
            for model in req.model_candidates:
                try:
                    resp = await client.post(
                        f"{node_info.agent_url}/services/{service}/start",
                        json={"gpu_id": gpu_id, "params": {**req.params, "model": model}},
                    )
                    if resp.is_success:
                        data = resp.json()
                        return {
                            "allocation_id": str(_uuid.uuid4()),
                            "service": service,
                            "node_id": node_id,
                            "gpu_id": gpu_id,
                            "model": model,
                            "url": data.get("url"),
                            "started": not warm,
                            "warm": warm,
                        }
                    last_error = resp.text
                except httpx.HTTPError as exc:
                    raise HTTPException(502, detail=f"Agent unreachable: {exc}")

        raise HTTPException(
            503,
            detail=f"All model candidates exhausted for {service!r}. Last error: {last_error}",
        )

    @app.delete("/api/services/{service}")
    async def stop_service(service: str, node_id: str) -> dict[str, Any]:
        """Stop a managed service on the given node."""
        node_info = agent_supervisor.get_node_info(node_id)
        if node_info is None:
            raise HTTPException(422, detail=f"Unknown node_id {node_id!r}")

        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(f"{node_info.agent_url}/services/{service}/stop")
                resp.raise_for_status()
                return {"service": service, "node_id": node_id, "stopped": resp.json().get("stopped", False)}
            except httpx.HTTPError as exc:
                raise HTTPException(502, detail=f"Agent unreachable: {exc}")

    return app
