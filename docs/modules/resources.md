# resources

VRAM allocation engine and GPU profile registry. Works alongside the [tasks](tasks.md) module to prevent GPU OOM errors across concurrent LLM workloads.

```python
from circuitforge_core.resources import ResourceCoordinator, VRAMSlot
```

## Architecture

The resource coordinator runs as a sidecar alongside each product (via `compose.override.yml`) and registers with the cf-orch coordinator at `http://10.1.10.71:7700`. The coordinator maintains a global view of VRAM allocation across all products and all GPUs.

```
Product A (kiwi)     ─┐
Product B (peregrine) ─┤ → cf-orch coordinator → GPU 0 (24GB)
Product C (snipe)    ─┘                        → GPU 1 (8GB)
```

## VRAM allocation

`VRAMSlot` represents a lease on a fixed VRAM budget:

```python
slot = VRAMSlot(service="kiwi", task_type="recipe_llm", vram_gb=4.0)
async with coordinator.lease(slot):
    result = await run_inference(prompt)
# VRAM released automatically on context exit
```

If the requested VRAM is not available, the coordinator queues the request. Tasks are executed in FIFO order within each priority class.

## Eviction engine

When a high-priority task needs VRAM that is held by a lower-priority task, the eviction engine signals the lower-priority task to checkpoint and pause. Eviction is cooperative, not forced — tasks must implement the `checkpoint()` callback.

## GPU profile registry

The registry maps GPU models to capability profiles:

```python
from circuitforge_core.resources import get_gpu_profile

profile = get_gpu_profile("RTX 4090")
# GpuProfile(vram_gb=24.0, fp16=True, int8=True, int4=True, max_batch=32)
```

Profiles are used by the LLM router to determine which model quantizations a GPU can run.

## Local fallback

When the cf-orch coordinator is not reachable (local dev without the sidecar), the resource coordinator falls back to a local-only mode: tasks run sequentially with no cross-product coordination. This is safe for development but should not be used in production if multiple products are running concurrently on the same GPU.
