from __future__ import annotations

import logging
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
from circuitforge_core.resources.coordinator.eviction_engine import EvictionEngine
from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.node_selector import select_node
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.coordinator.service_registry import ServiceRegistry
from circuitforge_core.resources.profiles.schema import ProcessSpec

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()


def _get_health_path(profile_registry: ProfileRegistry, service: str) -> str:
    """Return the health_path for a service from the first matching profile spec."""
    for profile in profile_registry.list_public():
        svc = profile.services.get(service)
        if svc and isinstance(svc.managed, ProcessSpec):
            return svc.managed.health_path
    return "/health"

_PROBE_INTERVAL_S = 5.0    # how often to poll starting instances
_PROBE_TIMEOUT_S = 300.0   # give up and mark stopped after this many seconds


async def _run_instance_probe_loop(service_registry: ServiceRegistry) -> None:
    """
    Background loop: transition 'starting' instances to 'running' once their
    /health endpoint responds, or to 'stopped' after PROBE_TIMEOUT_S.
    """
    import asyncio

    start_times: dict[str, float] = {}  # instance key → time first seen as starting

    while True:
        await asyncio.sleep(_PROBE_INTERVAL_S)
        now = time.time()
        for inst in service_registry.all_instances():
            if inst.state != "starting":
                start_times.pop(f"{inst.service}:{inst.node_id}:{inst.gpu_id}", None)
                continue
            key = f"{inst.service}:{inst.node_id}:{inst.gpu_id}"
            start_times.setdefault(key, now)

            healthy = False
            if inst.url:
                try:
                    with urllib.request.urlopen(
                        inst.url.rstrip("/") + inst.health_path, timeout=2.0
                    ) as resp:
                        healthy = resp.status == 200
                except Exception:
                    pass

            if healthy:
                service_registry.upsert_instance(
                    service=inst.service, node_id=inst.node_id, gpu_id=inst.gpu_id,
                    state="running", model=inst.model, url=inst.url,
                )
                start_times.pop(key, None)
                logger.info("Instance %s/%s gpu=%s transitioned to running", inst.service, inst.node_id, inst.gpu_id)
            elif now - start_times[key] > _PROBE_TIMEOUT_S:
                service_registry.upsert_instance(
                    service=inst.service, node_id=inst.node_id, gpu_id=inst.gpu_id,
                    state="stopped", model=inst.model, url=inst.url,
                )
                start_times.pop(key, None)
                logger.warning("Instance %s/%s gpu=%s timed out in starting state — marked stopped", inst.service, inst.node_id, inst.gpu_id)


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
    service_registry: ServiceRegistry,
) -> FastAPI:
    eviction_engine = EvictionEngine(lease_manager=lease_manager)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
        import asyncio
        heartbeat_task = asyncio.create_task(agent_supervisor.run_heartbeat_loop())
        probe_task = asyncio.create_task(_run_instance_probe_loop(service_registry))
        yield
        agent_supervisor.stop()
        heartbeat_task.cancel()
        probe_task.cancel()

    app = FastAPI(title="cf-orch-coordinator", lifespan=_lifespan)

    # Optional Heimdall auth — enabled when HEIMDALL_URL env var is set.
    # Self-hosted coordinators skip this entirely; the CF-hosted public endpoint
    # (orch.circuitforge.tech) sets HEIMDALL_URL to gate paid+ access.
    from circuitforge_core.resources.coordinator.auth import HeimdallAuthMiddleware
    _auth = HeimdallAuthMiddleware.from_env()
    if _auth is not None:
        app.middleware("http")(_auth)

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

        # Filter candidates by VRAM headroom — require free VRAM >= service ceiling
        # so the model can actually load without competing for VRAM with other processes.
        if service_max_mb > 0 and free_mb < service_max_mb:
            raise HTTPException(
                503,
                detail=f"Insufficient VRAM on gpu {req.gpu_id}: {free_mb}MB free, need {service_max_mb}MB",
            )

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

        # Validate service is known in at least one profile, regardless of gpu_id
        if not any(service in p.services for p in profile_registry.list_public()):
            raise HTTPException(422, detail=f"Unknown service {service!r} — not in any profile")

        residents = lease_manager.resident_keys()

        if req.gpu_id is None:
            online = agent_supervisor.online_agents()
            placement = select_node(online, service, profile_registry, residents)
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

        warm = f"{node_id}:{service}" in residents

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
                        svc_url = data.get("url", "")
                        alloc = service_registry.allocate(
                            service=service,
                            node_id=node_id,
                            gpu_id=gpu_id,
                            model=model,
                            caller=req.caller,
                            url=svc_url,
                            ttl_s=req.ttl_s,
                        )
                        # Seed the instance state for first-time starts.
                        # adopted=True means the agent found it already running.
                        adopted = data.get("adopted", False)
                        instance_state = "running" if (warm or adopted) else "starting"
                        health_path = _get_health_path(profile_registry, service)
                        service_registry.upsert_instance(
                            service=service,
                            node_id=node_id,
                            gpu_id=gpu_id,
                            state=instance_state,
                            model=model,
                            url=svc_url,
                            health_path=health_path,
                        )
                        return {
                            "allocation_id": alloc.allocation_id,
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

    @app.delete("/api/services/{service}/allocations/{allocation_id}")
    async def release_allocation(service: str, allocation_id: str) -> dict[str, Any]:
        existing = service_registry.get_allocation(allocation_id)
        if existing is None or existing.service != service:
            raise HTTPException(404, detail=f"Allocation {allocation_id!r} not found for service {service!r}")
        released = service_registry.release(allocation_id)
        if not released:
            raise HTTPException(404, detail=f"Allocation {allocation_id!r} not found")
        return {"released": True, "allocation_id": allocation_id}

    @app.get("/api/services/{service}/status")
    def get_service_status(service: str) -> dict[str, Any]:
        instances = [i for i in service_registry.all_instances() if i.service == service]
        allocations = [a for a in service_registry.all_allocations() if a.service == service]
        return {
            "service": service,
            "instances": [
                {
                    "node_id": i.node_id,
                    "gpu_id": i.gpu_id,
                    "state": i.state,
                    "model": i.model,
                    "url": i.url,
                    "idle_since": i.idle_since,
                }
                for i in instances
            ],
            "allocations": [
                {
                    "allocation_id": a.allocation_id,
                    "node_id": a.node_id,
                    "gpu_id": a.gpu_id,
                    "model": a.model,
                    "caller": a.caller,
                    "url": a.url,
                    "expires_at": a.expires_at,
                }
                for a in allocations
            ],
        }

    @app.get("/api/services")
    def list_services() -> dict[str, Any]:
        instances = service_registry.all_instances()
        return {
            "services": [
                {
                    "service": i.service,
                    "node_id": i.node_id,
                    "gpu_id": i.gpu_id,
                    "state": i.state,
                    "model": i.model,
                    "url": i.url,
                }
                for i in instances
            ]
        }

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
