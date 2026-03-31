from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.models import GpuInfo, NodeInfo

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
    def __init__(self, lease_manager: LeaseManager) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._lease_manager = lease_manager
        self._running = False

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

    async def poll_agent(self, node_id: str) -> bool:
        record = self._agents.get(node_id)
        if record is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=_AGENT_TIMEOUT_S) as client:
                resp = await client.get(f"{record.agent_url}/gpu-info")
            resp.raise_for_status()
            data = resp.json()
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
            return True
        except Exception as exc:
            logger.warning("Agent %s unreachable: %s", node_id, exc)
            record.online = False
            return False

    async def poll_all(self) -> None:
        await asyncio.gather(*[self.poll_agent(nid) for nid in self._agents])

    async def run_heartbeat_loop(self) -> None:
        self._running = True
        while self._running:
            await self.poll_all()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    def stop(self) -> None:
        self._running = False
