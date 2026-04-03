"""Tests for ServiceManager ProcessSpec support."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from circuitforge_core.resources.agent.service_manager import ServiceManager
from circuitforge_core.resources.profiles.schema import (
    GpuProfile,
    ProcessSpec,
    ServiceProfile,
)


def _make_profile(args_template: str = "--port {port} --gpu-id {gpu_id}") -> GpuProfile:
    return GpuProfile(
        schema_version=1,
        name="test",
        vram_total_mb=8192,
        services={
            "vllm": ServiceProfile(
                max_mb=5120,
                priority=1,
                managed=ProcessSpec(
                    exec_path="/usr/bin/python",
                    args_template=args_template,
                    port=8000,
                    host_port=8000,
                    cwd="/tmp",
                ),
            ),
            "no_managed": ServiceProfile(max_mb=1024, priority=2),
        },
    )


@pytest.fixture
def manager():
    return ServiceManager(node_id="test-node", profile=_make_profile(), advertise_host="127.0.0.1")


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


def test_is_running_returns_false_when_no_proc(manager):
    assert manager.is_running("vllm") is False


def test_is_running_returns_false_when_proc_exited(manager):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited
    manager._procs["vllm"] = mock_proc
    assert manager.is_running("vllm") is False


def test_is_running_returns_false_when_port_not_listening(manager):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    manager._procs["vllm"] = mock_proc

    with patch("socket.create_connection", side_effect=OSError("refused")):
        assert manager.is_running("vllm") is False


def test_is_running_returns_true_when_proc_alive_and_port_open(manager):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    manager._procs["vllm"] = mock_proc

    mock_socket = MagicMock()
    mock_socket.__enter__ = MagicMock(return_value=mock_socket)
    mock_socket.__exit__ = MagicMock(return_value=False)
    with patch("socket.create_connection", return_value=mock_socket):
        assert manager.is_running("vllm") is True


def test_is_running_unknown_service_returns_false(manager):
    assert manager.is_running("nonexistent") is False


def test_is_running_no_managed_spec_returns_false(manager):
    assert manager.is_running("no_managed") is False


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


def test_start_launches_process_and_returns_url(manager):
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(manager, "is_running", return_value=False):
        mock_popen.return_value = MagicMock()
        url = manager.start("vllm", gpu_id=0, params={"model": "mymodel"})

    assert url == "http://127.0.0.1:8000"
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "/usr/bin/python"
    assert "--port" in cmd
    assert "8000" in cmd
    assert "--gpu-id" in cmd
    assert "0" in cmd


def test_start_returns_url_immediately_when_already_running(manager):
    with patch.object(manager, "is_running", return_value=True):
        with patch("subprocess.Popen") as mock_popen:
            url = manager.start("vllm", gpu_id=0, params={})

    assert url == "http://127.0.0.1:8000"
    mock_popen.assert_not_called()


def test_start_raises_for_unknown_service(manager):
    with pytest.raises(ValueError, match="not in profile"):
        manager.start("nonexistent", gpu_id=0, params={})


def test_start_stores_proc_in_procs(manager):
    mock_proc = MagicMock()
    with patch("subprocess.Popen", return_value=mock_proc), \
         patch.object(manager, "is_running", return_value=False):
        manager.start("vllm", gpu_id=0, params={})

    assert manager._procs["vllm"] is mock_proc


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_terminates_running_process(manager):
    mock_proc = MagicMock()
    manager._procs["vllm"] = mock_proc

    result = manager.stop("vllm")

    assert result is True
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()
    assert "vllm" not in manager._procs


def test_stop_kills_process_that_wont_terminate(manager):
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = Exception("timeout")
    manager._procs["vllm"] = mock_proc

    result = manager.stop("vllm")

    assert result is True
    mock_proc.kill.assert_called_once()


def test_stop_returns_true_when_no_proc_tracked(manager):
    # No proc in _procs — still returns True (idempotent stop)
    result = manager.stop("vllm")
    assert result is True


def test_stop_returns_false_for_unknown_service(manager):
    result = manager.stop("nonexistent")
    assert result is False


# ---------------------------------------------------------------------------
# list_running / get_url
# ---------------------------------------------------------------------------


def test_list_running_returns_running_services(manager):
    def _is_running(svc: str) -> bool:
        return svc == "vllm"

    with patch.object(manager, "is_running", side_effect=_is_running):
        running = manager.list_running()

    assert running == ["vllm"]


def test_get_url_returns_none_when_not_running(manager):
    with patch.object(manager, "is_running", return_value=False):
        assert manager.get_url("vllm") is None


def test_get_url_returns_url_when_running(manager):
    with patch.object(manager, "is_running", return_value=True):
        assert manager.get_url("vllm") == "http://127.0.0.1:8000"
