from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
from circuitforge_core.resources.coordinator.service_registry import ServiceRegistry
from circuitforge_core.resources.models import GpuInfo, NodeInfo, ResidentAllocation

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 10.0
_AGENT_TIMEOUT_S = 5.0


@dataclass
class AgentRecord:
    node_id: str
    agent_url: str
    last_seen: float = field(default_factory=time.time)
    gpus: list[GpuInfo] = field(default_factory=list)
    online: bool = False


class AgentSupervisor:
    def __init__(
        self,
        lease_manager: LeaseManager,
        service_registry: ServiceRegistry | None = None,
        profile_registry: ProfileRegistry | None = None,
    ) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._lease_manager = lease_manager
        self._running = False
        self._service_registry = service_registry
        self._profile_registry = profile_registry
        self._heartbeat_tick = 0

    def register(self, node_id: str, agent_url: str) -> None:
        if node_id not in self._agents:
            self._agents[node_id] = AgentRecord(node_id=node_id, agent_url=agent_url)
            logger.info("Registered agent node: %s @ %s", node_id, agent_url)

    def get_node_info(self, node_id: str) -> NodeInfo | None:
        record = self._agents.get(node_id)
        if record is None:
            return None
        return NodeInfo(
            node_id=record.node_id,
            agent_url=record.agent_url,
            gpus=record.gpus,
            last_heartbeat=record.last_seen,
        )

    def all_nodes(self) -> list[NodeInfo]:
        return [
            NodeInfo(
                node_id=r.node_id,
                agent_url=r.agent_url,
                gpus=r.gpus,
                last_heartbeat=r.last_seen,
            )
            for r in self._agents.values()
        ]

    def online_agents(self) -> "dict[str, AgentRecord]":
        """Return only currently-online agents, keyed by node_id."""
        return {nid: rec for nid, rec in self._agents.items() if rec.online}

    async def poll_agent(self, node_id: str) -> bool:
        record = self._agents.get(node_id)
        if record is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=_AGENT_TIMEOUT_S) as client:
                gpu_resp = await client.get(f"{record.agent_url}/gpu-info")
                gpu_resp.raise_for_status()

                # Resident-info is best-effort — older agents may not have the endpoint.
                try:
                    res_resp = await client.get(f"{record.agent_url}/resident-info")
                    resident_data = res_resp.json() if res_resp.is_success else {}
                except Exception:
                    resident_data = {}

            data = gpu_resp.json()
            gpus = [
                GpuInfo(
                    gpu_id=g["gpu_id"],
                    name=g["name"],
                    vram_total_mb=g["vram_total_mb"],
                    vram_used_mb=g["vram_used_mb"],
                    vram_free_mb=g["vram_free_mb"],
                )
                for g in data.get("gpus", [])
            ]
            record.gpus = gpus
            record.last_seen = time.time()
            record.online = True
            for gpu in gpus:
                self._lease_manager.register_gpu(node_id, gpu.gpu_id, gpu.vram_total_mb)

            residents = [
                (r["service"], r.get("model_name"))
                for r in resident_data.get("residents", [])
            ]
            self._lease_manager.set_residents_for_node(node_id, residents)

            return True
        except Exception as exc:
            logger.warning("Agent %s unreachable: %s", node_id, exc)
            record.online = False
            return False

    async def poll_all(self) -> None:
        await asyncio.gather(*[self.poll_agent(nid) for nid in self._agents])

    def _build_idle_stop_config(self) -> dict[str, int]:
        if self._profile_registry is None:
            return {}
        config: dict[str, int] = {}
        for profile in self._profile_registry.list_public():
            for svc_name, svc in profile.services.items():
                if svc.idle_stop_after_s > 0:
                    existing = config.get(svc_name, 0)
                    config[svc_name] = min(existing, svc.idle_stop_after_s) if existing > 0 else svc.idle_stop_after_s
        return config

    async def _http_post(self, url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url)
                return resp.is_success
        except Exception as exc:
            logger.warning("HTTP POST %s failed: %s", url, exc)
            return False

    async def _run_idle_sweep(self) -> None:
        if self._service_registry is None:
            return
        idle_stop_config = self._build_idle_stop_config()
        if not idle_stop_config:
            return
        timed_out = self._service_registry.idle_past_timeout(idle_stop_config)
        for instance in timed_out:
            node_info = self.get_node_info(instance.node_id)
            if node_info is None:
                continue
            stop_url = f"{node_info.agent_url}/services/{instance.service}/stop"
            logger.info(
                "Idle sweep: stopping %s on %s gpu%s (idle timeout)",
                instance.service, instance.node_id, instance.gpu_id,
            )
            await self._http_post(stop_url)

    async def run_heartbeat_loop(self) -> None:
        self._running = True
        while self._running:
            await self.poll_all()
            self._heartbeat_tick += 1
            if self._heartbeat_tick % 3 == 0:
                await self._run_idle_sweep()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    def stop(self) -> None:
        self._running = False
