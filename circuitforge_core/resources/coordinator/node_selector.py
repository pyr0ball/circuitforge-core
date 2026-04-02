from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuitforge_core.resources.coordinator.agent_supervisor import AgentRecord
    from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry

_WARM_BONUS_MB = 1000


@dataclass(frozen=True)
class _Scored:
    node_id: str
    gpu_id: int
    vram_free_mb: int
    effective_free_mb: int
    can_fit: bool
    warm: bool


def select_node(
    agents: "dict[str, AgentRecord]",
    service: str,
    profile_registry: "ProfileRegistry",
    resident_keys: set[str],
) -> tuple[str, int] | None:
    """
    Pick the best (node_id, gpu_id) for the requested service.
    Warm nodes (service already running) get priority, then sorted by free VRAM.
    Returns None if no suitable node exists.
    """
    service_max_mb = _find_service_max_mb(service, profile_registry)
    if service_max_mb is None:
        return None  # service not in any profile

    candidates: list[_Scored] = []
    for node_id, record in agents.items():
        if not record.online:
            continue
        for gpu in record.gpus:
            warm = f"{node_id}:{service}" in resident_keys
            effective = gpu.vram_free_mb + (_WARM_BONUS_MB if warm else 0)
            can_fit = gpu.vram_free_mb >= service_max_mb
            candidates.append(_Scored(
                node_id=node_id,
                gpu_id=gpu.gpu_id,
                vram_free_mb=gpu.vram_free_mb,
                effective_free_mb=effective,
                can_fit=can_fit,
                warm=warm,
            ))
    if not candidates:
        return None
    # Prefer: (1) warm nodes (model already resident — no cold start)
    #         (2) cold nodes that can fit the service (free >= half of max_mb)
    # Fallback: best-effort node when nothing fits and nothing is warm
    #   (coordinator will attempt to start the service anyway; it may evict or fail)
    # Note: resident_keys are per-node, not per-GPU. On multi-GPU nodes, the warm
    #   bonus applies to all GPUs on the node. This is a known coarseness —
    #   per-GPU resident tracking requires a resident_key format change.
    preferred = [c for c in candidates if c.warm or c.can_fit]
    pool = preferred if preferred else candidates
    best = max(pool, key=lambda c: (c.warm, c.effective_free_mb))
    return best.node_id, best.gpu_id


def _find_service_max_mb(service: str, profile_registry: "ProfileRegistry") -> int | None:
    for profile in profile_registry.list_public():
        svc = profile.services.get(service)
        if svc is not None:
            return svc.max_mb
    return None
