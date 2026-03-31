from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

app = typer.Typer(name="cf-orch", help="CircuitForge GPU resource orchestrator")

_SYSTEMD_UNIT_PATH = Path("/etc/systemd/system/cf-orch.service")

_SYSTEMD_UNIT_TEMPLATE = """\
[Unit]
Description=CircuitForge GPU Resource Orchestrator
After=network.target

[Service]
Type=simple
ExecStart={python} -m circuitforge_core.resources.cli start
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


@app.command()
def start(
    profile: Annotated[Optional[Path], typer.Option(help="Profile YAML path")] = None,
    host: str = "0.0.0.0",
    port: int = 7700,
    agent_port: int = 7701,
) -> None:
    """Start the cf-orch coordinator (auto-detects GPU profile if not specified)."""
    from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
    from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
    from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
    from circuitforge_core.resources.coordinator.app import create_coordinator_app
    from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor

    lease_manager = LeaseManager()
    profile_registry = ProfileRegistry()
    supervisor = AgentSupervisor(lease_manager=lease_manager)

    monitor = GpuMonitor()
    gpus = monitor.poll()
    if not gpus:
        typer.echo("Warning: no GPUs detected via nvidia-smi — coordinator running with 0 VRAM")
    else:
        for gpu in gpus:
            lease_manager.register_gpu("local", gpu.gpu_id, gpu.vram_total_mb)
        typer.echo(f"Detected {len(gpus)} GPU(s)")

    if profile:
        active_profile = profile_registry.load(profile)
        typer.echo(f"Using profile: {active_profile.name} (from {profile})")
    else:
        active_profile = profile_registry.auto_detect(gpus) if gpus else profile_registry.list_public()[-1]
        typer.echo(f"Auto-selected profile: {active_profile.name}")

    coordinator_app = create_coordinator_app(
        lease_manager=lease_manager,
        profile_registry=profile_registry,
        agent_supervisor=supervisor,
    )

    typer.echo(f"Starting cf-orch coordinator on {host}:{port}")
    uvicorn.run(coordinator_app, host=host, port=port)


@app.command()
def agent(
    coordinator: str = "http://localhost:7700",
    node_id: str = "local",
    host: str = "0.0.0.0",
    port: int = 7701,
) -> None:
    """Start a cf-orch node agent (for remote nodes like Navi, Huginn)."""
    from circuitforge_core.resources.agent.app import create_agent_app

    agent_app = create_agent_app(node_id=node_id)
    typer.echo(f"Starting cf-orch agent [{node_id}] on {host}:{port}")
    uvicorn.run(agent_app, host=host, port=port)


@app.command()
def status(coordinator: str = "http://localhost:7700") -> None:
    """Show GPU and lease status from the coordinator."""
    import httpx
    try:
        resp = httpx.get(f"{coordinator}/api/nodes", timeout=5.0)
        resp.raise_for_status()
        nodes = resp.json().get("nodes", [])
        for node in nodes:
            typer.echo(f"\nNode: {node['node_id']}")
            for gpu in node.get("gpus", []):
                typer.echo(
                    f"  GPU {gpu['gpu_id']}: {gpu['name']} — "
                    f"{gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB used"
                )
    except Exception as exc:
        typer.echo(f"Coordinator unreachable at {coordinator}: {exc}", err=True)
        raise typer.Exit(1)


@app.command("install-service")
def install_service(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print unit file without writing"),
) -> None:
    """Write a systemd unit file for cf-orch (requires root)."""
    python = sys.executable
    unit_content = _SYSTEMD_UNIT_TEMPLATE.format(python=python)
    if dry_run:
        typer.echo(f"Would write to {_SYSTEMD_UNIT_PATH}:\n")
        typer.echo(unit_content)
        return
    try:
        _SYSTEMD_UNIT_PATH.write_text(unit_content)
        typer.echo(f"Written: {_SYSTEMD_UNIT_PATH}")
        typer.echo("Run: sudo systemctl daemon-reload && sudo systemctl enable --now cf-orch")
    except PermissionError:
        typer.echo(f"Permission denied writing to {_SYSTEMD_UNIT_PATH}. Run as root.", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
