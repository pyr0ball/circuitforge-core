from __future__ import annotations

import asyncio
from collections import defaultdict

from circuitforge_core.resources.models import VRAMLease


class LeaseManager:
    def __init__(self) -> None:
        self._leases: dict[str, VRAMLease] = {}
        self._gpu_total: dict[tuple[str, int], int] = {}
        self._gpu_used: dict[tuple[str, int], int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def register_gpu(self, node_id: str, gpu_id: int, total_mb: int) -> None:
        self._gpu_total[(node_id, gpu_id)] = total_mb

    def gpu_total_mb(self, node_id: str, gpu_id: int) -> int:
        return self._gpu_total.get((node_id, gpu_id), 0)

    def used_mb(self, node_id: str, gpu_id: int) -> int:
        return self._gpu_used[(node_id, gpu_id)]

    async def try_grant(
        self,
        node_id: str,
        gpu_id: int,
        mb: int,
        service: str,
        priority: int,
        ttl_s: float = 0.0,
    ) -> VRAMLease | None:
        async with self._lock:
            total = self._gpu_total.get((node_id, gpu_id), 0)
            used = self._gpu_used[(node_id, gpu_id)]
            if total - used < mb:
                return None
            lease = VRAMLease.create(
                gpu_id=gpu_id, node_id=node_id, mb=mb,
                service=service, priority=priority, ttl_s=ttl_s,
            )
            self._leases[lease.lease_id] = lease
            self._gpu_used[(node_id, gpu_id)] += mb
            return lease

    async def release(self, lease_id: str) -> bool:
        async with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return False
            self._gpu_used[(lease.node_id, lease.gpu_id)] -= lease.mb_granted
            return True

    def get_eviction_candidates(
        self,
        node_id: str,
        gpu_id: int,
        needed_mb: int,
        requester_priority: int,
    ) -> list[VRAMLease]:
        candidates = [
            lease for lease in self._leases.values()
            if lease.node_id == node_id
            and lease.gpu_id == gpu_id
            and lease.priority > requester_priority
        ]
        candidates.sort(key=lambda l: l.priority, reverse=True)
        selected: list[VRAMLease] = []
        freed = 0
        for candidate in candidates:
            selected.append(candidate)
            freed += candidate.mb_granted
            if freed >= needed_mb:
                break
        return selected

    def list_leases(
        self, node_id: str | None = None, gpu_id: int | None = None
    ) -> list[VRAMLease]:
        return [
            lease for lease in self._leases.values()
            if (node_id is None or lease.node_id == node_id)
            and (gpu_id is None or lease.gpu_id == gpu_id)
        ]

    def all_leases(self) -> list[VRAMLease]:
        return list(self._leases.values())
