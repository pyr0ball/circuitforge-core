# tests/test_manage/test_native_mode.py
"""Unit tests for NativeManager — PID file process management."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.manage.config import ManageConfig, NativeService
from circuitforge_core.manage.native_mode import NativeManager


def _cfg(*services: NativeService) -> ManageConfig:
    return ManageConfig(app_name="testapp", services=list(services))


def _svc(name: str = "api", port: int = 8000) -> NativeService:
    return NativeService(name=name, command="python -c 'import time; time.sleep(999)'", port=port)


@pytest.fixture
def mgr(tmp_path: Path) -> NativeManager:
    cfg = _cfg(_svc("api", 8000), _svc("frontend", 8001))
    m = NativeManager(cfg, root=tmp_path)
    # redirect pid/log dirs into tmp_path for test isolation
    m._pid_dir = tmp_path / "pids"
    m._log_dir = tmp_path / "logs"
    m._pid_dir.mkdir()
    m._log_dir.mkdir()
    return m


# ── PID file helpers ──────────────────────────────────────────────────────────

def test_write_and_read_pid(mgr: NativeManager, tmp_path: Path) -> None:
    mgr._write_pid("api", 12345, "python server.py")
    assert mgr._read_pid("api") == 12345


def test_read_pid_missing_returns_none(mgr: NativeManager) -> None:
    assert mgr._read_pid("nonexistent") is None


def test_read_pid_corrupt_returns_none(mgr: NativeManager, tmp_path: Path) -> None:
    mgr._pid_dir.mkdir(exist_ok=True)
    (mgr._pid_dir / "api.pid").write_text("notanumber\n")
    assert mgr._read_pid("api") is None


# ── is_running ────────────────────────────────────────────────────────────────

def test_is_running_true_when_pid_alive(mgr: NativeManager) -> None:
    mgr._write_pid("api", os.getpid(), "test")  # use current PID — guaranteed alive
    assert mgr.is_running("api") is True


def test_is_running_false_when_no_pid_file(mgr: NativeManager) -> None:
    assert mgr.is_running("api") is False


def test_is_running_false_when_pid_dead(mgr: NativeManager) -> None:
    mgr._write_pid("api", 999999999, "dead")  # very unlikely PID
    with patch.object(mgr, "_pid_alive", return_value=False):
        assert mgr.is_running("api") is False


# ── start ─────────────────────────────────────────────────────────────────────

def test_start_spawns_process(mgr: NativeManager, tmp_path: Path) -> None:
    mock_proc = MagicMock()
    mock_proc.pid = 42

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        started = mgr.start("api")

    assert "api" in started
    mock_popen.assert_called_once()
    assert mgr._read_pid("api") == 42


def test_start_all_spawns_all_services(mgr: NativeManager) -> None:
    mock_proc = MagicMock()
    mock_proc.pid = 99

    with patch("subprocess.Popen", return_value=mock_proc):
        started = mgr.start()  # no name = all services

    assert set(started) == {"api", "frontend"}


def test_start_skips_already_running(mgr: NativeManager) -> None:
    mgr._write_pid("api", os.getpid(), "test")  # current PID = alive

    with patch("subprocess.Popen") as mock_popen:
        started = mgr.start("api")

    mock_popen.assert_not_called()
    assert "api" not in started


def test_start_unknown_service_raises(mgr: NativeManager) -> None:
    with pytest.raises(ValueError, match="Unknown service"):
        mgr.start("doesnotexist")


# ── stop ──────────────────────────────────────────────────────────────────────

def test_stop_kills_and_removes_pid_file(mgr: NativeManager) -> None:
    mgr._write_pid("api", 55555, "test")

    with patch.object(mgr, "_pid_alive", return_value=True), \
         patch.object(mgr, "_kill") as mock_kill:
        stopped = mgr.stop("api")

    assert "api" in stopped
    mock_kill.assert_called_once_with(55555)
    assert not mgr._pid_path("api").exists()


def test_stop_all(mgr: NativeManager) -> None:
    mgr._write_pid("api", 1001, "test")
    mgr._write_pid("frontend", 1002, "test")

    with patch.object(mgr, "_pid_alive", return_value=True), \
         patch.object(mgr, "_kill"):
        stopped = mgr.stop()

    assert set(stopped) == {"api", "frontend"}


def test_stop_not_running_returns_empty(mgr: NativeManager) -> None:
    # No PID file → nothing to stop
    stopped = mgr.stop("api")
    assert stopped == []


# ── status ────────────────────────────────────────────────────────────────────

def test_status_running(mgr: NativeManager) -> None:
    mgr._write_pid("api", os.getpid(), "test")

    rows = mgr.status()
    api_row = next(r for r in rows if r.name == "api")
    assert api_row.running is True
    assert api_row.pid == os.getpid()
    assert api_row.port == 8000


def test_status_stopped(mgr: NativeManager) -> None:
    rows = mgr.status()
    for row in rows:
        assert row.running is False
        assert row.pid is None


# ── logs ──────────────────────────────────────────────────────────────────────

def test_logs_prints_last_lines(mgr: NativeManager, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
    log = mgr._log_path("api")
    log.write_text("\n".join(f"line {i}" for i in range(100)))

    mgr.logs("api", follow=False, lines=10)

    out = capsys.readouterr().out
    assert "line 99" in out
    assert "line 90" in out
    assert "line 89" not in out   # only last 10


def test_logs_missing_file_prints_warning(mgr: NativeManager, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
    mgr.logs("api", follow=False)
    err = capsys.readouterr().err
    assert "No log file" in err
