from __future__ import annotations

import logging
import os
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    allocation_id: str
    service: str
    node_id: str
    gpu_id: int
    model: str | None
    url: str
    started: bool
    warm: bool


class CFOrchClient:
    """
    Client for cf-orch coordinator allocation.

    Sync usage (in LLMRouter or other sync code):
        client = CFOrchClient(os.environ["CF_ORCH_URL"])
        with client.allocate("vllm", model_candidates=["Ouro-1.4B"]) as alloc:
            # alloc.url is the inference endpoint

    Async usage (in FastAPI apps):
        async with client.allocate_async("vllm", model_candidates=["Ouro-1.4B"]) as alloc:
            ...

    Authentication:
        Pass api_key explicitly, or set CF_LICENSE_KEY env var. When set, every
        request carries Authorization: Bearer <key>. Required for the hosted
        CircuitForge coordinator (orch.circuitforge.tech); optional for local
        self-hosted coordinators.

    Raises ValueError immediately if coordinator_url is empty.
    """

    def __init__(self, coordinator_url: str, api_key: str | None = None) -> None:
        if not coordinator_url:
            raise ValueError("coordinator_url is empty — cf-orch not configured")
        self._url = coordinator_url.rstrip("/")
        self._api_key = api_key or os.environ.get("CF_LICENSE_KEY", "")

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    def _build_body(self, model_candidates: list[str] | None, ttl_s: float, caller: str) -> dict:
        return {
            "model_candidates": model_candidates or [],
            "ttl_s": ttl_s,
            "caller": caller,
        }

    def _parse_allocation(self, data: dict, service: str) -> Allocation:
        return Allocation(
            allocation_id=data["allocation_id"],
            service=service,
            node_id=data["node_id"],
            gpu_id=data["gpu_id"],
            model=data.get("model"),
            url=data["url"],
            started=data.get("started", False),
            warm=data.get("warm", False),
        )

    @contextmanager
    def allocate(
        self,
        service: str,
        *,
        model_candidates: list[str] | None = None,
        ttl_s: float = 3600.0,
        caller: str = "",
    ):
        """Sync context manager. Allocates on enter, releases on exit."""
        resp = httpx.post(
            f"{self._url}/api/services/{service}/allocate",
            json=self._build_body(model_candidates, ttl_s, caller),
            headers=self._headers(),
            timeout=120.0,
        )
        if not resp.is_success:
            raise RuntimeError(
                f"cf-orch allocation failed for {service!r}: "
                f"HTTP {resp.status_code} — {resp.text[:200]}"
            )
        alloc = self._parse_allocation(resp.json(), service)
        try:
            yield alloc
        finally:
            try:
                httpx.delete(
                    f"{self._url}/api/services/{service}/allocations/{alloc.allocation_id}",
                    headers=self._headers(),
                    timeout=10.0,
                )
            except Exception as exc:
                logger.debug("cf-orch release failed (non-fatal): %s", exc)

    @asynccontextmanager
    async def allocate_async(
        self,
        service: str,
        *,
        model_candidates: list[str] | None = None,
        ttl_s: float = 3600.0,
        caller: str = "",
    ):
        """Async context manager. Allocates on enter, releases on exit."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._url}/api/services/{service}/allocate",
                json=self._build_body(model_candidates, ttl_s, caller),
                headers=self._headers(),
            )
            if not resp.is_success:
                raise RuntimeError(
                    f"cf-orch allocation failed for {service!r}: "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
            alloc = self._parse_allocation(resp.json(), service)
            try:
                yield alloc
            finally:
                try:
                    await client.delete(
                        f"{self._url}/api/services/{service}/allocations/{alloc.allocation_id}",
                        headers=self._headers(),
                        timeout=10.0,
                    )
                except Exception as exc:
                    logger.debug("cf-orch async release failed (non-fatal): %s", exc)
