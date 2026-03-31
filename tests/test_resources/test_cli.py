from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from circuitforge_core.resources.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cf-orch" in result.output.lower() or "Usage" in result.output


def test_status_command_shows_no_coordinator_message():
    with patch("httpx.get", side_effect=ConnectionRefusedError("refused")):
        result = runner.invoke(app, ["status"])
    assert result.exit_code != 0 or "unreachable" in result.output.lower() \
        or "coordinator" in result.output.lower()


def test_install_service_creates_systemd_unit(tmp_path: Path):
    unit_path = tmp_path / "cf-orch.service"
    with patch(
        "circuitforge_core.resources.cli._SYSTEMD_UNIT_PATH", unit_path
    ):
        result = runner.invoke(app, ["install-service", "--dry-run"])
    assert result.exit_code == 0
    assert "cf-orch.service" in result.output or "systemd" in result.output.lower()
