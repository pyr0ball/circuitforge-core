from __future__ import annotations

import asyncio
import logging

import httpx

from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
from circuitforge_core.resources.models import VRAMLease

logger = logging.getLogger(__name__)

_DEFAULT_EVICTION_TIMEOUT_S = 10.0


class EvictionEngine:
    def __init__(
        self,
        lease_manager: LeaseManager,
        eviction_timeout_s: float = _DEFAULT_EVICTION_TIMEOUT_S,
    ) -> None:
        self.lease_manager = lease_manager
        self._timeout = eviction_timeout_s

    async def request_lease(
        self,
        node_id: str,
        gpu_id: int,
        mb: int,
        service: str,
        priority: int,
        agent_url: str,
        ttl_s: float = 0.0,
    ) -> VRAMLease | None:
        # Fast path: enough free VRAM
        lease = await self.lease_manager.try_grant(
            node_id, gpu_id, mb, service, priority, ttl_s
        )
        if lease is not None:
            return lease

        # Find eviction candidates
        candidates = self.lease_manager.get_eviction_candidates(
            node_id=node_id, gpu_id=gpu_id,
            needed_mb=mb, requester_priority=priority,
        )
        if not candidates:
            logger.info(
                "No eviction candidates for %s on %s:GPU%d (%dMB needed)",
                service, node_id, gpu_id, mb,
            )
            return None

        # Evict candidates
        freed_mb = sum(c.mb_granted for c in candidates)
        logger.info(
            "Evicting %d lease(s) to free %dMB for %s",
            len(candidates), freed_mb, service,
        )
        for candidate in candidates:
            await self._evict_lease(candidate, agent_url)

        # Wait for evictions to free up VRAM (poll with timeout)
        deadline = asyncio.get_event_loop().time() + self._timeout
        while asyncio.get_event_loop().time() < deadline:
            lease = await self.lease_manager.try_grant(
                node_id, gpu_id, mb, service, priority, ttl_s
            )
            if lease is not None:
                return lease
            await asyncio.sleep(0.1)

        logger.warning("Eviction timed out for %s after %.1fs", service, self._timeout)
        return None

    async def _evict_lease(self, lease: VRAMLease, agent_url: str) -> None:
        """Release lease accounting. Process-level eviction deferred to Plan B."""
        await self.lease_manager.release(lease.lease_id)

    async def _call_agent_evict(self, agent_url: str, lease: VRAMLease) -> bool:
        """POST /evict to the agent. Stub for v1 — real process lookup in Plan B."""
        return True
