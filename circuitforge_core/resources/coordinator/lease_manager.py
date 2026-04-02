from __future__ import annotations

import asyncio
from collections import defaultdict

from circuitforge_core.resources.models import ResidentAllocation, VRAMLease


class LeaseManager:
    def __init__(self) -> None:
        self._leases: dict[str, VRAMLease] = {}
        self._gpu_total: dict[tuple[str, int], int] = {}
        self._gpu_used: dict[tuple[str, int], int] = defaultdict(int)
        self._lock = asyncio.Lock()
        # Resident allocations — keyed "node_id:service", updated by heartbeat.
        # No lock needed: only the single heartbeat task writes this dict.
        self._residents: dict[str, ResidentAllocation] = {}

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
        candidates.sort(key=lambda lease: lease.priority, reverse=True)
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

    # ── resident tracking ────────────────────────────────────────────

    def set_residents_for_node(
        self,
        node_id: str,
        residents: list[tuple[str, str | None]],  # (service, model_name)
    ) -> None:
        """
        Replace the resident snapshot for a node.

        Preserves first_seen for entries whose service+model_name are unchanged,
        so the dashboard can show how long a model has been warm.
        """
        new_keys = {f"{node_id}:{service}" for service, _ in residents}

        # Remove stale entries (service no longer running on this node).
        for key in list(self._residents):
            if key.startswith(f"{node_id}:") and key not in new_keys:
                del self._residents[key]

        # Upsert: preserve first_seen when model is unchanged, reset otherwise.
        for service, model_name in residents:
            key = f"{node_id}:{service}"
            existing = self._residents.get(key)
            if existing is not None and existing.model_name == model_name:
                continue  # same model still loaded — keep original first_seen
            self._residents[key] = ResidentAllocation(
                service=service,
                node_id=node_id,
                model_name=model_name,
            )

    def all_residents(self) -> list[ResidentAllocation]:
        return list(self._residents.values())

    def resident_keys(self) -> set[str]:
        """Return set of 'node_id:service' strings for currently-warm services."""
        return set(self._residents.keys())
