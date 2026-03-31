"""Tests for circuitforge_core.tasks.scheduler."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.tasks.scheduler import (
    TaskScheduler,
    detect_available_vram_gb,
    get_scheduler,
    reset_scheduler,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """SQLite DB with background_tasks table."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE background_tasks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type  TEXT    NOT NULL,
            job_id     INTEGER NOT NULL DEFAULT 0,
            status     TEXT    NOT NULL DEFAULT 'queued',
            params     TEXT,
            error      TEXT,
            created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    return db


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Always tear down the scheduler singleton between tests."""
    yield
    reset_scheduler()


TASK_TYPES = frozenset({"fast_task"})
BUDGETS = {"fast_task": 1.0}


# ── detect_available_vram_gb ──────────────────────────────────────────────────

def test_detect_vram_from_cfortch():
    """Uses cf-orch free VRAM when coordinator is reachable."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "nodes": [
            {"node_id": "local", "gpus": [{"vram_free_mb": 4096}, {"vram_free_mb": 4096}]}
        ]
    }
    with patch("circuitforge_core.tasks.scheduler.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = detect_available_vram_gb(coordinator_url="http://localhost:7700")
    assert result == pytest.approx(8.0)  # 4096 + 4096 MB → 8 GB


def test_detect_vram_cforch_unavailable_falls_back_to_unlimited():
    """Falls back to 999.0 when cf-orch is unreachable and preflight unavailable."""
    with patch("circuitforge_core.tasks.scheduler.httpx") as mock_httpx:
        mock_httpx.get.side_effect = ConnectionRefusedError()
        result = detect_available_vram_gb()
    assert result == 999.0


def test_detect_vram_cforch_empty_nodes_falls_back():
    """If cf-orch returns no nodes with GPUs, falls back to unlimited."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"nodes": []}
    with patch("circuitforge_core.tasks.scheduler.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        result = detect_available_vram_gb()
    assert result == 999.0


def test_detect_vram_preflight_fallback():
    """Falls back to preflight total VRAM when cf-orch is unreachable."""
    # Build a fake scripts.preflight module with get_gpus returning two GPUs.
    fake_scripts = ModuleType("scripts")
    fake_preflight = ModuleType("scripts.preflight")
    fake_preflight.get_gpus = lambda: [  # type: ignore[attr-defined]
        {"vram_total_gb": 8.0},
        {"vram_total_gb": 4.0},
    ]
    fake_scripts.preflight = fake_preflight  # type: ignore[attr-defined]

    with patch("circuitforge_core.tasks.scheduler.httpx") as mock_httpx, \
         patch.dict(
             __import__("sys").modules,
             {"scripts": fake_scripts, "scripts.preflight": fake_preflight},
         ):
        mock_httpx.get.side_effect = ConnectionRefusedError()
        result = detect_available_vram_gb()

    assert result == pytest.approx(12.0)  # 8.0 + 4.0 GB


# ── TaskScheduler basic behaviour ─────────────────────────────────────────────

def test_enqueue_returns_true_on_success(tmp_db: Path):
    ran: List[int] = []

    def run_fn(db_path, task_id, task_type, job_id, params):
        ran.append(task_id)

    sched = TaskScheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS, available_vram_gb=8.0)
    sched.start()
    result = sched.enqueue(1, "fast_task", 0, None)
    sched.shutdown()
    assert result is True


def test_scheduler_runs_task(tmp_db: Path):
    """Enqueued task is executed by the batch worker."""
    ran: List[int] = []
    event = threading.Event()

    def run_fn(db_path, task_id, task_type, job_id, params):
        ran.append(task_id)
        event.set()

    sched = TaskScheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS, available_vram_gb=8.0)
    sched.start()
    sched.enqueue(42, "fast_task", 0, None)
    assert event.wait(timeout=3.0), "Task was not executed within 3 seconds"
    sched.shutdown()
    assert ran == [42]


def test_enqueue_returns_false_when_queue_full(tmp_db: Path):
    """Returns False and does not enqueue when max_queue_depth is reached."""
    gate = threading.Event()

    def blocking_run_fn(db_path, task_id, task_type, job_id, params):
        gate.wait()

    sched = TaskScheduler(
        tmp_db, blocking_run_fn, TASK_TYPES, BUDGETS,
        available_vram_gb=8.0, max_queue_depth=2
    )
    sched.start()
    results = [sched.enqueue(i, "fast_task", 0, None) for i in range(1, 10)]
    gate.set()
    sched.shutdown()
    assert False in results


def test_scheduler_drains_multiple_tasks(tmp_db: Path):
    """All enqueued tasks of the same type are run serially."""
    ran: List[int] = []
    done = threading.Event()
    TOTAL = 5

    def run_fn(db_path, task_id, task_type, job_id, params):
        ran.append(task_id)
        if len(ran) >= TOTAL:
            done.set()

    sched = TaskScheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS, available_vram_gb=8.0)
    sched.start()
    for i in range(1, TOTAL + 1):
        sched.enqueue(i, "fast_task", 0, None)
    assert done.wait(timeout=5.0), f"Only ran {len(ran)} of {TOTAL} tasks"
    sched.shutdown()
    assert sorted(ran) == list(range(1, TOTAL + 1))


def test_vram_budget_blocks_second_type(tmp_db: Path):
    """Second task type is not started when VRAM would be exceeded."""
    gate_a = threading.Event()
    gate_b = threading.Event()
    started = []

    def run_fn(db_path, task_id, task_type, job_id, params):
        started.append(task_type)
        if task_type == "type_a":
            gate_a.wait()
        else:
            gate_b.wait()

    two_types = frozenset({"type_a", "type_b"})
    tight_budgets = {"type_a": 4.0, "type_b": 4.0}  # 4+4 > 6 GB available

    sched = TaskScheduler(
        tmp_db, run_fn, two_types, tight_budgets, available_vram_gb=6.0
    )
    sched.start()
    sched.enqueue(1, "type_a", 0, None)
    sched.enqueue(2, "type_b", 0, None)

    time.sleep(0.2)
    assert started == ["type_a"]

    gate_a.set()
    time.sleep(0.2)
    gate_b.set()
    sched.shutdown()
    assert "type_b" in started


def test_get_scheduler_singleton(tmp_db: Path):
    """get_scheduler() returns the same instance on repeated calls."""
    run_fn = MagicMock()
    s1 = get_scheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS)
    s2 = get_scheduler(tmp_db)  # no run_fn — should reuse existing
    assert s1 is s2


def test_reset_scheduler_clears_singleton(tmp_db: Path):
    """reset_scheduler() allows a new singleton to be constructed."""
    run_fn = MagicMock()
    s1 = get_scheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS)
    reset_scheduler()
    s2 = get_scheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS)
    assert s1 is not s2


def test_load_queued_tasks_on_startup(tmp_db: Path):
    """Tasks with status='queued' in the DB at startup are loaded into the deque."""
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO background_tasks (task_type, job_id, status) VALUES ('fast_task', 0, 'queued')"
    )
    conn.commit()
    conn.close()

    ran: List[int] = []
    done = threading.Event()

    def run_fn(db_path, task_id, task_type, job_id, params):
        ran.append(task_id)
        done.set()

    sched = TaskScheduler(tmp_db, run_fn, TASK_TYPES, BUDGETS, available_vram_gb=8.0)
    sched.start()
    assert done.wait(timeout=3.0), "Pre-loaded task was not run"
    sched.shutdown()
    assert len(ran) == 1


def test_load_queued_tasks_missing_table_does_not_crash(tmp_path: Path):
    """Scheduler does not crash if background_tasks table doesn't exist yet."""
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()

    run_fn = MagicMock()
    sched = TaskScheduler(db, run_fn, TASK_TYPES, BUDGETS, available_vram_gb=8.0)
    sched.start()
    sched.shutdown()
    # No exception = pass
