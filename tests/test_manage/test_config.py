# tests/test_manage/test_config.py
"""Unit tests for ManageConfig TOML parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from circuitforge_core.manage.config import DockerConfig, ManageConfig, NativeService


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "manage.toml"
    p.write_text(content)
    return p


def test_minimal_config(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
[app]
name = "kiwi"
default_url = "http://localhost:8511"
""")
    cfg = ManageConfig.from_cwd(tmp_path)
    assert cfg.app_name == "kiwi"
    assert cfg.default_url == "http://localhost:8511"
    assert cfg.services == []
    assert cfg.docker.compose_file == "compose.yml"


def test_docker_section(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
[app]
name = "peregrine"

[docker]
compose_file = "compose.prod.yml"
project = "prng"
""")
    cfg = ManageConfig.from_cwd(tmp_path)
    assert cfg.docker.compose_file == "compose.prod.yml"
    assert cfg.docker.project == "prng"


def test_native_services(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
[app]
name = "snipe"
default_url = "http://localhost:8509"

[[native.services]]
name = "api"
command = "uvicorn app.main:app --port 8510"
port = 8510

[[native.services]]
name = "frontend"
command = "npm run preview"
port = 8509
cwd = "frontend"
""")
    cfg = ManageConfig.from_cwd(tmp_path)
    assert len(cfg.services) == 2
    api = cfg.services[0]
    assert api.name == "api"
    assert api.port == 8510
    assert api.cwd == ""
    fe = cfg.services[1]
    assert fe.name == "frontend"
    assert fe.cwd == "frontend"


def test_native_service_env(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
[app]
name = "avocet"

[[native.services]]
name = "api"
command = "uvicorn app.main:app --port 8503"
port = 8503
env = { PYTHONPATH = ".", DEBUG = "1" }
""")
    cfg = ManageConfig.from_cwd(tmp_path)
    assert cfg.services[0].env == {"PYTHONPATH": ".", "DEBUG": "1"}


def test_from_cwd_no_toml_falls_back_to_dirname(tmp_path: Path) -> None:
    """No manage.toml → app_name inferred from directory name."""
    cfg = ManageConfig.from_cwd(tmp_path)
    assert cfg.app_name == tmp_path.name
    assert cfg.services == []


def test_docker_project_defaults_to_app_name(tmp_path: Path) -> None:
    _write_toml(tmp_path, "[app]\nname = \"osprey\"\n")
    cfg = ManageConfig.from_cwd(tmp_path)
    assert cfg.docker.project == "osprey"
