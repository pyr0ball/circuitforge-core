# tests/test_manage/test_docker_mode.py
"""Unit tests for DockerManager — compose wrapper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from circuitforge_core.manage.config import DockerConfig, ManageConfig
from circuitforge_core.manage.docker_mode import DockerManager, docker_available


def _cfg(compose_file: str = "compose.yml", project: str = "testapp") -> ManageConfig:
    return ManageConfig(
        app_name="testapp",
        docker=DockerConfig(compose_file=compose_file, project=project),
    )


@pytest.fixture
def mgr(tmp_path: Path) -> DockerManager:
    (tmp_path / "compose.yml").write_text("version: '3'")
    return DockerManager(_cfg(), root=tmp_path)


# ── docker_available ──────────────────────────────────────────────────────────

def test_docker_available_true() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert docker_available() is True


def test_docker_available_false_on_exception() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert docker_available() is False


# ── compose_file_exists ───────────────────────────────────────────────────────

def test_compose_file_exists(mgr: DockerManager, tmp_path: Path) -> None:
    assert mgr.compose_file_exists() is True


def test_compose_file_missing(tmp_path: Path) -> None:
    m = DockerManager(_cfg(compose_file="nope.yml"), root=tmp_path)
    assert m.compose_file_exists() is False


# ── start / stop / restart / status ──────────────────────────────────────────

def test_start_runs_up(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.start()
    args = mock_run.call_args[0][0]
    assert "up" in args
    assert "-d" in args
    assert "--build" in args


def test_start_specific_service(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.start("api")
    args = mock_run.call_args[0][0]
    assert "api" in args


def test_stop_all_runs_down(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.stop()
    args = mock_run.call_args[0][0]
    assert "down" in args


def test_stop_service_runs_stop(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.stop("frontend")
    args = mock_run.call_args[0][0]
    assert "stop" in args
    assert "frontend" in args


def test_restart_runs_restart(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.restart()
    args = mock_run.call_args[0][0]
    assert "restart" in args


def test_status_runs_ps(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.status()
    args = mock_run.call_args[0][0]
    assert "ps" in args


def test_build_no_cache(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.build(no_cache=True)
    args = mock_run.call_args[0][0]
    assert "build" in args
    assert "--no-cache" in args


def test_compose_project_flag_included(mgr: DockerManager) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.start()
    args = mock_run.call_args[0][0]
    assert "-p" in args
    assert "testapp" in args


def test_compose_file_flag_included(mgr: DockerManager, tmp_path: Path) -> None:
    with patch("subprocess.run") as mock_run:
        mgr.start()
    args = mock_run.call_args[0][0]
    assert "-f" in args
