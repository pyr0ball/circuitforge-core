# tests/test_resources/test_coordinator_probe.py
"""
Unit tests for _run_instance_probe_loop in coordinator/app.py.

Covers:
  - healthy path:   /health → 200 → state transitions starting → running
  - timeout path:   no healthy response within _PROBE_TIMEOUT_S → starting → stopped
  - cleanup path:   non-starting instance cleans up its start_times entry
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.resources.coordinator.app import (
    _PROBE_TIMEOUT_S,
    _run_instance_probe_loop,
)
from circuitforge_core.resources.coordinator.service_registry import ServiceInstance, ServiceRegistry


# ── helpers ──────────────────────────────────────────────────────────────────

def _inst(**kwargs) -> ServiceInstance:
    defaults = dict(
        service="vllm", node_id="node1", gpu_id=0,
        state="starting", model="qwen", url="http://localhost:8000",
    )
    defaults.update(kwargs)
    return ServiceInstance(**defaults)


def _registry(*instances: ServiceInstance) -> MagicMock:
    reg = MagicMock(spec=ServiceRegistry)
    reg.all_instances.return_value = list(instances)
    return reg


def _health_resp(status: int = 200) -> MagicMock:
    """Context-manager mock that simulates an HTTP response."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: resp
    resp.__exit__ = MagicMock(return_value=False)
    return resp


async def _one_tick(coro_fn, registry, *, time_val: float = 1000.0, **url_patch):
    """
    Run the probe loop for exactly one iteration then cancel it.

    asyncio.sleep is patched to return immediately on the first call
    and raise CancelledError on the second (ending the loop cleanly).
    """
    calls = 0

    async def _fake_sleep(_delay):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise asyncio.CancelledError()

    patches = [
        patch("asyncio.sleep", new=_fake_sleep),
        patch("time.time", return_value=time_val),
    ]
    if url_patch:
        patches.append(patch("urllib.request.urlopen", **url_patch))

    ctx = [p.__enter__() for p in patches]
    try:
        await coro_fn(registry)
    except asyncio.CancelledError:
        pass
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_probe_transitions_starting_to_running():
    """GET /health → 200 while in starting state → upsert_instance(state='running')."""
    reg = _registry(_inst(state="starting", url="http://localhost:8000"))

    calls = 0

    async def fake_sleep(_delay):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", new=fake_sleep), \
         patch("time.time", return_value=1000.0), \
         patch("urllib.request.urlopen", return_value=_health_resp(200)):
        try:
            await _run_instance_probe_loop(reg)
        except asyncio.CancelledError:
            pass

    reg.upsert_instance.assert_called_once_with(
        service="vllm", node_id="node1", gpu_id=0,
        state="running", model="qwen", url="http://localhost:8000",
    )


@pytest.mark.asyncio
async def test_probe_transitions_starting_to_stopped_on_timeout():
    """No healthy response + time past _PROBE_TIMEOUT_S → upsert_instance(state='stopped').

    Tick 1: seeds start_times[key] = 1000.0
    Tick 2: time has advanced past _PROBE_TIMEOUT_S → timeout fires → stopped
    Tick 3: CancelledError exits the loop
    """
    reg = _registry(_inst(state="starting", url="http://localhost:8000"))

    tick = 0
    # Tick 1: t=1000 (seed); Tick 2: t=far_future (timeout fires)
    times = [1000.0, 1000.0 + _PROBE_TIMEOUT_S + 1.0]

    async def fake_sleep(_delay):
        nonlocal tick
        tick += 1
        if tick > 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", new=fake_sleep), \
         patch("time.time", side_effect=times * 10), \
         patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        try:
            await _run_instance_probe_loop(reg)
        except asyncio.CancelledError:
            pass

    reg.upsert_instance.assert_called_once_with(
        service="vllm", node_id="node1", gpu_id=0,
        state="stopped", model="qwen", url="http://localhost:8000",
    )


@pytest.mark.asyncio
async def test_probe_cleans_up_start_times_for_non_starting():
    """
    An instance that is no longer in 'starting' state should not cause
    upsert_instance to be called, and its key should be removed from start_times.

    We verify this indirectly: run two ticks — first with state='starting' (seeds
    the key and transitions to running), second with the updated registry returning
    state='running' (should not call upsert again).
    """
    starting_inst = _inst(state="starting", url="http://localhost:8000")
    running_inst = _inst(state="running", url="http://localhost:8000")

    tick = 0

    # First tick: instance is starting → transitions to running
    # Second tick: registry now returns running → no upsert
    # Third tick: cancel
    def instances_side_effect():
        if tick <= 1:
            return [starting_inst]
        return [running_inst]

    reg = MagicMock(spec=ServiceRegistry)
    reg.all_instances.side_effect = instances_side_effect

    async def fake_sleep(_delay):
        nonlocal tick
        tick += 1
        if tick > 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", new=fake_sleep), \
         patch("time.time", return_value=1000.0), \
         patch("urllib.request.urlopen", return_value=_health_resp(200)):
        try:
            await _run_instance_probe_loop(reg)
        except asyncio.CancelledError:
            pass

    # upsert should have been called exactly once (the starting→running transition)
    assert reg.upsert_instance.call_count == 1
    reg.upsert_instance.assert_called_once_with(
        service="vllm", node_id="node1", gpu_id=0,
        state="running", model="qwen", url="http://localhost:8000",
    )


@pytest.mark.asyncio
async def test_probe_no_url_does_not_attempt_health_check():
    """Instance with no URL stays in starting state (no health check, no timeout yet)."""
    reg = _registry(_inst(state="starting", url=None))

    tick = 0

    async def fake_sleep(_delay):
        nonlocal tick
        tick += 1
        if tick > 1:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", new=fake_sleep), \
         patch("time.time", return_value=1000.0), \
         patch("urllib.request.urlopen") as mock_urlopen:
        try:
            await _run_instance_probe_loop(reg)
        except asyncio.CancelledError:
            pass

    mock_urlopen.assert_not_called()
    reg.upsert_instance.assert_not_called()
