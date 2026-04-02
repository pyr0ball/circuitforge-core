from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass
from typing import Literal


@dataclass
class ServiceAllocation:
    allocation_id: str
    service: str
    node_id: str
    gpu_id: int
    model: str | None
    caller: str
    url: str
    created_at: float
    expires_at: float  # 0 = no expiry


@dataclass
class ServiceInstance:
    service: str
    node_id: str
    gpu_id: int
    state: Literal["starting", "running", "idle", "stopped"]
    model: str | None
    url: str | None
    idle_since: float | None = None


class ServiceRegistry:
    """
    In-memory registry of service allocations and instance state.

    Allocations: per-caller request — many per service instance.
    Instances: per (service, node_id, gpu_id) — one per running container.
    """

    def __init__(self) -> None:
        self._allocations: dict[str, ServiceAllocation] = {}
        self._instances: dict[str, ServiceInstance] = {}  # key: "service:node_id:gpu_id"

    # ── allocation API ────────────────────────────────────────────────

    def allocate(
        self,
        service: str,
        node_id: str,
        gpu_id: int,
        model: str | None,
        url: str,
        caller: str,
        ttl_s: float,
    ) -> ServiceAllocation:
        alloc = ServiceAllocation(
            allocation_id=str(uuid.uuid4()),
            service=service,
            node_id=node_id,
            gpu_id=gpu_id,
            model=model,
            caller=caller,
            url=url,
            created_at=time.time(),
            expires_at=time.time() + ttl_s if ttl_s > 0 else 0.0,
        )
        self._allocations[alloc.allocation_id] = alloc

        # If an instance exists in idle/stopped state, mark it running again
        key = f"{service}:{node_id}:{gpu_id}"
        if key in self._instances:
            inst = self._instances[key]
            if inst.state in ("idle", "stopped"):
                self._instances[key] = dataclasses.replace(
                    inst, state="running", idle_since=None
                )
        return alloc

    def release(self, allocation_id: str) -> bool:
        alloc = self._allocations.pop(allocation_id, None)
        if alloc is None:
            return False
        # If no active allocations remain for this instance, mark it idle
        key = f"{alloc.service}:{alloc.node_id}:{alloc.gpu_id}"
        if self.active_allocations(alloc.service, alloc.node_id, alloc.gpu_id) == 0:
            if key in self._instances:
                self._instances[key] = dataclasses.replace(
                    self._instances[key], state="idle", idle_since=time.time()
                )
        return True

    def active_allocations(self, service: str, node_id: str, gpu_id: int) -> int:
        return sum(
            1 for a in self._allocations.values()
            if a.service == service and a.node_id == node_id and a.gpu_id == gpu_id
        )

    # ── instance API ─────────────────────────────────────────────────

    def upsert_instance(
        self,
        service: str,
        node_id: str,
        gpu_id: int,
        state: Literal["starting", "running", "idle", "stopped"],
        model: str | None,
        url: str | None,
    ) -> ServiceInstance:
        key = f"{service}:{node_id}:{gpu_id}"
        existing = self._instances.get(key)
        idle_since: float | None = None
        if state == "idle":
            # Preserve idle_since if already idle; set now if transitioning into idle
            idle_since = existing.idle_since if (existing and existing.state == "idle") else time.time()
        inst = ServiceInstance(
            service=service, node_id=node_id, gpu_id=gpu_id,
            state=state, model=model, url=url, idle_since=idle_since,
        )
        self._instances[key] = inst
        return inst

    def all_allocations(self) -> list[ServiceAllocation]:
        return list(self._allocations.values())

    def all_instances(self) -> list[ServiceInstance]:
        return list(self._instances.values())

    def mark_stopped(self, service: str, node_id: str, gpu_id: int) -> None:
        """Transition an instance to 'stopped' state and clear idle_since."""
        key = f"{service}:{node_id}:{gpu_id}"
        if key in self._instances:
            self._instances[key] = dataclasses.replace(
                self._instances[key], state="stopped", idle_since=None
            )

    def idle_past_timeout(self, idle_stop_config: dict[str, int]) -> list[ServiceInstance]:
        """
        Return instances in 'idle' state whose idle time exceeds their configured timeout.
        idle_stop_config: {service_name: seconds} — 0 means never stop automatically.
        """
        now = time.time()
        result = []
        for inst in self._instances.values():
            if inst.state != "idle" or inst.idle_since is None:
                continue
            timeout = idle_stop_config.get(inst.service, 0)
            if timeout > 0 and (now - inst.idle_since) >= timeout:
                result.append(inst)
        return result
