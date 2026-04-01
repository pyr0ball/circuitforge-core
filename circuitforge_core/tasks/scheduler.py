# circuitforge_core/tasks/scheduler.py
"""Resource-aware batch scheduler for LLM background tasks.

Generic scheduler that any CircuitForge product can use. Products supply:
  - task_types: frozenset[str]       — task type strings routed through this scheduler
  - vram_budgets: dict[str, float]   — VRAM GB estimate per task type
  - run_task_fn                      — product's task execution function

VRAM detection priority:
  1. cf-orch coordinator /api/nodes — free VRAM (lease-aware, cooperative)
  2. scripts.preflight.get_gpus()   — total GPU VRAM (Peregrine-era fallback)
  3. 999.0                          — unlimited (CPU-only or no detection available)

Public API:
    TaskScheduler           — the scheduler class
    detect_available_vram_gb() — standalone VRAM query helper
    get_scheduler()         — lazy process-level singleton
    reset_scheduler()       — test teardown only
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Optional

try:
    import httpx as httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

RunTaskFn = Callable[["Path", int, str, int, Optional[str]], None]


class TaskSpec(NamedTuple):
    id: int
    job_id: int
    params: Optional[str]

_DEFAULT_MAX_QUEUE_DEPTH = 500


def detect_available_vram_gb(
    coordinator_url: str = "http://localhost:7700",
) -> float:
    """Detect available VRAM GB for task scheduling.

    Returns free VRAM via cf-orch (sum across all nodes/GPUs) so the scheduler
    cooperates with other cf-orch consumers. Falls back to preflight total VRAM,
    then 999.0 (unlimited) if nothing is reachable.
    """
    # 1. Try cf-orch: use free VRAM so the scheduler cooperates with other
    #    cf-orch consumers (vision service, inference services, etc.)
    if httpx is not None:
        try:
            resp = httpx.get(f"{coordinator_url}/api/nodes", timeout=2.0)
            if resp.status_code == 200:
                nodes = resp.json().get("nodes", [])
                total_free_mb = sum(
                    gpu.get("vram_free_mb", 0)
                    for node in nodes
                    for gpu in node.get("gpus", [])
                )
                if total_free_mb > 0:
                    free_gb = total_free_mb / 1024.0
                    logger.debug(
                        "Scheduler VRAM from cf-orch: %.1f GB free", free_gb
                    )
                    return free_gb
        except Exception:
            pass

    # 2. Try preflight (systems with nvidia-smi; Peregrine-era fallback)
    try:
        from scripts.preflight import get_gpus  # type: ignore[import]

        gpus = get_gpus()
        if gpus:
            total_gb = sum(g.get("vram_total_gb", 0.0) for g in gpus)
            logger.debug(
                "Scheduler VRAM from preflight: %.1f GB total", total_gb
            )
            return total_gb
    except Exception:
        pass

    logger.debug(
        "Scheduler VRAM detection unavailable — using unlimited (999 GB)"
    )
    return 999.0


class TaskScheduler:
    """Resource-aware LLM task batch scheduler.

    Runs one batch-worker thread per task type while total reserved VRAM
    stays within the detected available budget. Always allows at least one
    batch to start even if its budget exceeds available VRAM (prevents
    permanent starvation on low-VRAM systems).

    Thread-safety: all queue/active state protected by self._lock.

    Usage::

        sched = get_scheduler(
            db_path=Path("data/app.db"),
            run_task_fn=my_run_task,
            task_types=frozenset({"cover_letter", "research"}),
            vram_budgets={"cover_letter": 2.5, "research": 5.0},
        )
        task_id, is_new = insert_task(db_path, "cover_letter", job_id)
        if is_new:
            enqueued = sched.enqueue(task_id, "cover_letter", job_id, params_json)
            if not enqueued:
                mark_task_failed(db_path, task_id, "Queue full")
    """

    def __init__(
        self,
        db_path: Path,
        run_task_fn: RunTaskFn,
        task_types: frozenset[str],
        vram_budgets: dict[str, float],
        available_vram_gb: Optional[float] = None,
        max_queue_depth: int = _DEFAULT_MAX_QUEUE_DEPTH,
        coordinator_url: str = "http://localhost:7700",
        service_name: str = "peregrine",
        lease_priority: int = 2,
    ) -> None:
        self._db_path = db_path
        self._run_task = run_task_fn
        self._task_types = frozenset(task_types)
        self._budgets: dict[str, float] = dict(vram_budgets)
        self._max_queue_depth = max_queue_depth

        self._coordinator_url = coordinator_url.rstrip("/")
        self._service_name = service_name
        self._lease_priority = lease_priority

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._queues: dict[str, deque[TaskSpec]] = {}
        self._active: dict[str, threading.Thread] = {}
        self._reserved_vram: float = 0.0
        self._thread: Optional[threading.Thread] = None

        self._available_vram: float = (
            available_vram_gb
            if available_vram_gb is not None
            else detect_available_vram_gb()
        )

        for t in self._task_types:
            if t not in self._budgets:
                logger.warning(
                    "No VRAM budget defined for task type %r — "
                    "defaulting to 0.0 GB (no VRAM gating for this type)",
                    t,
                )

        self._load_queued_tasks()

    def enqueue(
        self,
        task_id: int,
        task_type: str,
        job_id: int,
        params: Optional[str],
    ) -> bool:
        """Add a task to the scheduler queue.

        Returns True if enqueued successfully.
        Returns False if the queue is full — caller should mark the task failed.
        """
        with self._lock:
            q = self._queues.setdefault(task_type, deque())
            if len(q) >= self._max_queue_depth:
                logger.warning(
                    "Queue depth limit for %s (max=%d) — task %d dropped",
                    task_type,
                    self._max_queue_depth,
                    task_id,
                )
                return False
            q.append(TaskSpec(task_id, job_id, params))
        self._wake.set()
        return True

    def start(self) -> None:
        """Start the background scheduler loop thread. Call once after construction."""
        self._thread = threading.Thread(
            target=self._scheduler_loop, name="task-scheduler", daemon=True
        )
        self._thread.start()
        # Wake the loop immediately so tasks loaded from DB at startup are dispatched
        with self._lock:
            if any(self._queues.values()):
                self._wake.set()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal the scheduler to stop and wait for it to exit."""
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=30)
            self._wake.clear()
            with self._lock:
                # Reap batch threads that finished without waking us.
                # VRAM accounting is handled solely by _batch_worker's finally block;
                # the reaper only removes dead entries from _active.
                for t, thread in list(self._active.items()):
                    if not thread.is_alive():
                        del self._active[t]
                # Start new type batches while VRAM budget allows
                candidates = sorted(
                    [
                        t
                        for t in self._queues
                        if self._queues[t] and t not in self._active
                    ],
                    key=lambda t: len(self._queues[t]),
                    reverse=True,
                )
                for task_type in candidates:
                    budget = self._budgets.get(task_type, 0.0)
                    # Always allow at least one batch to run
                    if (
                        self._reserved_vram == 0.0
                        or self._reserved_vram + budget <= self._available_vram
                    ):
                        thread = threading.Thread(
                            target=self._batch_worker,
                            args=(task_type,),
                            name=f"batch-{task_type}",
                            daemon=True,
                        )
                        self._active[task_type] = thread
                        self._reserved_vram += budget
                        thread.start()

    def _acquire_lease(self, task_type: str) -> Optional[str]:
        """Request a VRAM lease from the coordinator. Returns lease_id or None."""
        if httpx is None:
            return None
        budget_gb = self._budgets.get(task_type, 0.0)
        if budget_gb <= 0:
            return None
        mb = int(budget_gb * 1024)
        try:
            # Pick the GPU with the most free VRAM on the first registered node
            resp = httpx.get(f"{self._coordinator_url}/api/nodes", timeout=2.0)
            if resp.status_code != 200:
                return None
            nodes = resp.json().get("nodes", [])
            if not nodes:
                return None
            best_node = best_gpu = best_free = None
            for node in nodes:
                for gpu in node.get("gpus", []):
                    free = gpu.get("vram_free_mb", 0)
                    if best_free is None or free > best_free:
                        best_node = node["node_id"]
                        best_gpu = gpu["gpu_id"]
                        best_free = free
            if best_node is None:
                return None
            lease_resp = httpx.post(
                f"{self._coordinator_url}/api/leases",
                json={
                    "node_id": best_node,
                    "gpu_id": best_gpu,
                    "mb": mb,
                    "service": self._service_name,
                    "priority": self._lease_priority,
                },
                timeout=3.0,
            )
            if lease_resp.status_code == 200:
                lease_id = lease_resp.json()["lease"]["lease_id"]
                logger.debug(
                    "Acquired VRAM lease %s for task_type=%s (%d MB)",
                    lease_id, task_type, mb,
                )
                return lease_id
        except Exception as exc:
            logger.debug("Lease acquire failed (non-fatal): %s", exc)
        return None

    def _release_lease(self, lease_id: str) -> None:
        """Release a coordinator VRAM lease. Best-effort; failures are logged only."""
        if httpx is None or not lease_id:
            return
        try:
            httpx.delete(
                f"{self._coordinator_url}/api/leases/{lease_id}",
                timeout=3.0,
            )
            logger.debug("Released VRAM lease %s", lease_id)
        except Exception as exc:
            logger.debug("Lease release failed (non-fatal): %s", exc)

    def _batch_worker(self, task_type: str) -> None:
        """Serial consumer for one task type. Runs until the type's deque is empty."""
        lease_id: Optional[str] = self._acquire_lease(task_type)
        try:
            while True:
                with self._lock:
                    q = self._queues.get(task_type)
                    if not q:
                        break
                    task = q.popleft()
                self._run_task(
                    self._db_path, task.id, task_type, task.job_id, task.params
                )
        finally:
            if lease_id:
                self._release_lease(lease_id)
            with self._lock:
                self._active.pop(task_type, None)
                self._reserved_vram -= self._budgets.get(task_type, 0.0)
            self._wake.set()

    def _load_queued_tasks(self) -> None:
        """Reload surviving 'queued' tasks from SQLite into deques at startup."""
        if not self._task_types:
            return
        task_types_list = sorted(self._task_types)
        placeholders = ",".join("?" * len(task_types_list))
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    f"SELECT id, task_type, job_id, params FROM background_tasks"
                    f" WHERE status='queued' AND task_type IN ({placeholders})"
                    f" ORDER BY created_at ASC",
                    task_types_list,
                ).fetchall()
        except sqlite3.OperationalError:
            # Table not yet created (first run before migrations)
            rows = []

        for row_id, task_type, job_id, params in rows:
            q = self._queues.setdefault(task_type, deque())
            q.append(TaskSpec(row_id, job_id, params))

        if rows:
            logger.info(
                "Scheduler: resumed %d queued task(s) from prior run", len(rows)
            )


# ── Process-level singleton ────────────────────────────────────────────────────

_scheduler: Optional[TaskScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler(
    db_path: Path,
    run_task_fn: Optional[RunTaskFn] = None,
    task_types: Optional[frozenset[str]] = None,
    vram_budgets: Optional[dict[str, float]] = None,
    max_queue_depth: int = _DEFAULT_MAX_QUEUE_DEPTH,
    coordinator_url: str = "http://localhost:7700",
    service_name: str = "peregrine",
) -> TaskScheduler:
    """Return the process-level TaskScheduler singleton.

    ``run_task_fn``, ``task_types``, and ``vram_budgets`` are required on the
    first call; ignored on subsequent calls (singleton already constructed).

    VRAM detection (which may involve a network call) is performed outside the
    lock so the lock is never held across blocking I/O.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    # Build outside the lock — TaskScheduler.__init__ may call detect_available_vram_gb()
    # which makes an httpx network call (up to 2 s). Holding the lock during that
    # would block any concurrent caller for the full duration.
    if run_task_fn is None or task_types is None or vram_budgets is None:
        raise ValueError(
            "run_task_fn, task_types, and vram_budgets are required "
            "on the first call to get_scheduler()"
        )
    candidate = TaskScheduler(
        db_path=db_path,
        run_task_fn=run_task_fn,
        task_types=task_types,
        vram_budgets=vram_budgets,
        max_queue_depth=max_queue_depth,
        coordinator_url=coordinator_url,
        service_name=service_name,
    )
    candidate.start()
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = candidate
        else:
            # Another thread beat us — shut down our candidate and use the winner.
            candidate.shutdown()
    return _scheduler


def reset_scheduler() -> None:
    """Shut down and clear the singleton. TEST TEARDOWN ONLY."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown()
            _scheduler = None
