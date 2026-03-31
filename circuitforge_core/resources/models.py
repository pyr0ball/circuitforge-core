from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VRAMLease:
    lease_id: str
    gpu_id: int
    node_id: str
    mb_granted: int
    holder_service: str
    priority: int
    expires_at: float  # unix timestamp; 0.0 = no expiry

    @classmethod
    def create(
        cls,
        gpu_id: int,
        node_id: str,
        mb: int,
        service: str,
        priority: int,
        ttl_s: float = 0.0,
    ) -> VRAMLease:
        return cls(
            lease_id=str(uuid.uuid4()),
            gpu_id=gpu_id,
            node_id=node_id,
            mb_granted=mb,
            holder_service=service,
            priority=priority,
            expires_at=time.time() + ttl_s if ttl_s > 0.0 else 0.0,
        )

    def is_expired(self) -> bool:
        return self.expires_at > 0.0 and time.time() > self.expires_at


@dataclass(frozen=True)
class GpuInfo:
    gpu_id: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int


@dataclass
class NodeInfo:
    node_id: str
    agent_url: str
    gpus: list[GpuInfo]
    last_heartbeat: float = field(default_factory=time.time)
