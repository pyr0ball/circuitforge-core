from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

logger = logging.getLogger(__name__)

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
    node_id: str = "local",
    agent_port: int = 7701,
) -> None:
    """Start the cf-orch coordinator (auto-detects GPU profile if not specified).

    Automatically pre-registers the local agent so its GPUs appear on the
    dashboard immediately. Remote nodes self-register via POST /api/nodes.
    """
    from circuitforge_core.resources.coordinator.lease_manager import LeaseManager
    from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry
    from circuitforge_core.resources.coordinator.agent_supervisor import AgentSupervisor
    from circuitforge_core.resources.coordinator.app import create_coordinator_app
    from circuitforge_core.resources.coordinator.service_registry import ServiceRegistry
    from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor

    from circuitforge_core.resources.coordinator.node_store import NodeStore

    lease_manager = LeaseManager()
    profile_registry = ProfileRegistry()
    service_registry = ServiceRegistry()
    node_store = NodeStore()
    supervisor = AgentSupervisor(
        lease_manager=lease_manager,
        service_registry=service_registry,
        profile_registry=profile_registry,
        node_store=node_store,
    )
    restored = supervisor.restore_from_store()
    if restored:
        typer.echo(f"Restored {restored} known node(s) from previous session")

    monitor = GpuMonitor()
    gpus = monitor.poll()
    if not gpus:
        typer.echo(
            "Warning: no GPUs detected via nvidia-smi — coordinator running with 0 VRAM"
        )
    else:
        typer.echo(f"Detected {len(gpus)} GPU(s)")

    if profile:
        active_profile = profile_registry.load(profile)
        typer.echo(f"Using profile: {active_profile.name} (from {profile})")
    else:
        active_profile = (
            profile_registry.auto_detect(gpus)
            if gpus
            else profile_registry.list_public()[-1]
        )
        typer.echo(f"Auto-selected profile: {active_profile.name}")

    # Pre-register the local agent — the heartbeat loop will poll it for live GPU data.
    local_agent_url = f"http://127.0.0.1:{agent_port}"
    supervisor.register(node_id, local_agent_url)
    typer.echo(f"Registered local node '{node_id}' → {local_agent_url}")

    coordinator_app = create_coordinator_app(
        lease_manager=lease_manager,
        profile_registry=profile_registry,
        agent_supervisor=supervisor,
        service_registry=service_registry,
    )

    typer.echo(f"Starting cf-orch coordinator on {host}:{port}")
    uvicorn.run(coordinator_app, host=host, port=port)


@app.command()
def agent(
    coordinator: str = "http://localhost:7700",
    node_id: str = "local",
    host: str = "0.0.0.0",
    port: int = 7701,
    advertise_host: Optional[str] = None,
    profile: Annotated[Optional[Path], typer.Option(help="Profile YAML path")] = None,
) -> None:
    """Start a cf-orch node agent and self-register with the coordinator.

    The agent starts its HTTP server, then POSTs its URL to the coordinator
    so it appears on the dashboard without manual configuration.

    Use --advertise-host to override the IP the coordinator should use to
    reach this agent (e.g. on a multi-homed or NATted host).
    """
    import threading
    import httpx
    from circuitforge_core.resources.agent.app import create_agent_app
    from circuitforge_core.resources.agent.service_manager import ServiceManager
    from circuitforge_core.resources.coordinator.profile_registry import ProfileRegistry

    # The URL the coordinator should use to reach this agent.
    reach_host = advertise_host or ("127.0.0.1" if host in ("0.0.0.0", "::") else host)
    agent_url = f"http://{reach_host}:{port}"

    _RECONNECT_INTERVAL_S = 30.0

    def _reconnect_loop() -> None:
        """
        Persistently re-register this agent with the coordinator.

        Runs as a daemon thread for the lifetime of the agent process:
        - Waits 2 s on first run (uvicorn needs time to bind)
        - Re-registers every 30 s thereafter
        - If the coordinator is down, silently retries — no crashing
        - When the coordinator restarts, the agent re-appears within one cycle

        This means coordinator restarts require no manual intervention on agent hosts.
        """
        import time
        first = True
        while True:
            time.sleep(2.0 if first else _RECONNECT_INTERVAL_S)
            first = False
            try:
                resp = httpx.post(
                    f"{coordinator}/api/nodes",
                    json={"node_id": node_id, "agent_url": agent_url},
                    timeout=5.0,
                )
                if resp.is_success:
                    logger.debug("Registered with coordinator at %s as '%s'", coordinator, node_id)
                else:
                    logger.warning(
                        "Coordinator registration returned %s", resp.status_code
                    )
            except Exception as exc:
                logger.debug("Coordinator at %s unreachable, will retry: %s", coordinator, exc)

    # Fire reconnect loop in a daemon thread so uvicorn.run() can start blocking immediately.
    threading.Thread(target=_reconnect_loop, daemon=True, name="cf-orch-reconnect").start()
    typer.echo(f"Reconnect loop started — will register with {coordinator} every {int(_RECONNECT_INTERVAL_S)}s")

    service_manager = None
    try:
        from circuitforge_core.resources.agent.gpu_monitor import GpuMonitor
        pr = ProfileRegistry()
        gpus = GpuMonitor().poll()
        p = pr.load(Path(profile)) if profile else pr.auto_detect(gpus)
        service_manager = ServiceManager(node_id=node_id, profile=p, advertise_host=reach_host)
        typer.echo(f"ServiceManager ready with profile: {p.name}")
    except Exception as exc:
        typer.echo(f"Warning: ServiceManager unavailable ({exc})", err=True)

    agent_app = create_agent_app(node_id=node_id, service_manager=service_manager)
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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print unit file without writing"
    ),
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
        typer.echo(
            "Run: sudo systemctl daemon-reload && sudo systemctl enable --now cf-orch"
        )
    except PermissionError:
        typer.echo(
            f"Permission denied writing to {_SYSTEMD_UNIT_PATH}. Run as root.", err=True
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
