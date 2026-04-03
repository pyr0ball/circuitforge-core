"""
circuitforge_core.manage.cli — cross-platform product manager CLI.

Usage (from any product directory):
    python -m circuitforge_core.manage start
    python -m circuitforge_core.manage stop
    python -m circuitforge_core.manage restart
    python -m circuitforge_core.manage status
    python -m circuitforge_core.manage logs [SERVICE]
    python -m circuitforge_core.manage open
    python -m circuitforge_core.manage build
    python -m circuitforge_core.manage install-shims

Products shim into this via a thin manage.sh / manage.ps1 that finds Python
and delegates: exec python -m circuitforge_core.manage "$@"
"""
from __future__ import annotations

import sys
import webbrowser
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import ManageConfig
from .docker_mode import DockerManager, docker_available
from .native_mode import NativeManager

app = typer.Typer(
    name="manage",
    help="CircuitForge cross-platform product manager",
    no_args_is_help=True,
)


class Mode(str, Enum):
    auto = "auto"
    docker = "docker"
    native = "native"


def _resolve(
    mode: Mode,
    root: Path,
    config: ManageConfig,
) -> tuple[str, DockerManager | NativeManager]:
    """Return (mode_name, manager) based on mode flag and environment."""
    if mode == Mode.docker or (
        mode == Mode.auto
        and docker_available()
        and (root / config.docker.compose_file).exists()
    ):
        return "docker", DockerManager(config, root)
    return "native", NativeManager(config, root)


def _load(root: Path) -> ManageConfig:
    return ManageConfig.from_cwd(root)


# ── commands ──────────────────────────────────────────────────────────────────

@app.command()
def start(
    service: Annotated[Optional[str], typer.Argument(help="Service name (omit for all)")] = None,
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Start services."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    typer.echo(f"[{config.app_name}] Starting ({mode_name} mode)…")
    if isinstance(mgr, DockerManager):
        mgr.start(service or "")
    else:
        started = mgr.start(service)
        if started:
            typer.echo(f"[{config.app_name}] Started: {', '.join(started)}")
        else:
            typer.echo(f"[{config.app_name}] All services already running")


@app.command()
def stop(
    service: Annotated[Optional[str], typer.Argument(help="Service name (omit for all)")] = None,
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Stop services."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    typer.echo(f"[{config.app_name}] Stopping ({mode_name} mode)…")
    if isinstance(mgr, DockerManager):
        mgr.stop(service or "")
    else:
        stopped = mgr.stop(service)
        if stopped:
            typer.echo(f"[{config.app_name}] Stopped: {', '.join(stopped)}")
        else:
            typer.echo(f"[{config.app_name}] No running services to stop")


@app.command()
def restart(
    service: Annotated[Optional[str], typer.Argument(help="Service name (omit for all)")] = None,
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Restart services."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    typer.echo(f"[{config.app_name}] Restarting ({mode_name} mode)…")
    if isinstance(mgr, DockerManager):
        mgr.restart(service or "")
    else:
        mgr.stop(service)
        mgr.start(service)


@app.command()
def status(
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Show service status."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    if isinstance(mgr, DockerManager):
        mgr.status()
    else:
        rows = mgr.status()
        if not rows:
            typer.echo(f"[{config.app_name}] No native services defined in manage.toml")
            return
        typer.echo(f"\n  {config.app_name} — native services\n")
        for svc in rows:
            indicator = typer.style("●", fg=typer.colors.GREEN) if svc.running \
                else typer.style("○", fg=typer.colors.RED)
            pid_str = f"  pid={svc.pid}" if svc.pid else ""
            port_str = f"  port={svc.port}" if svc.port else ""
            typer.echo(f"  {indicator} {svc.name:<20}{pid_str}{port_str}")
        typer.echo("")


@app.command()
def logs(
    service: Annotated[Optional[str], typer.Argument(help="Service name")] = None,
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f/-F"),
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Tail service logs."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    if isinstance(mgr, DockerManager):
        mgr.logs(service or "", follow=follow)
    else:
        if not service:
            # Default to first service when none specified
            if not config.services:
                typer.echo("No native services defined", err=True)
                raise typer.Exit(1)
            service = config.services[0].name
        mgr.logs(service, follow=follow)


@app.command()
def build(
    no_cache: bool = False,
    mode: Mode = Mode.auto,
    root: Path = Path("."),
) -> None:
    """Build/rebuild service images (Docker mode only)."""
    config = _load(root.resolve())
    mode_name, mgr = _resolve(mode, root.resolve(), config)
    if isinstance(mgr, NativeManager):
        typer.echo("build is only available in Docker mode", err=True)
        raise typer.Exit(1)
    typer.echo(f"[{config.app_name}] Building images…")
    mgr.build(no_cache=no_cache)


@app.command("open")
def open_browser(
    url: Annotated[Optional[str], typer.Option(help="Override URL")] = None,
    root: Path = Path("."),
) -> None:
    """Open the product web UI in the default browser."""
    config = _load(root.resolve())
    target = url or config.default_url
    if not target:
        typer.echo("No URL configured. Set default_url in manage.toml or pass --url.", err=True)
        raise typer.Exit(1)
    typer.echo(f"Opening {target}")
    webbrowser.open(target)


@app.command("install-shims")
def install_shims(
    root: Path = Path("."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing shims"),
) -> None:
    """
    Write manage.sh and manage.ps1 shims into the product directory.

    The shims auto-detect the Python environment (conda, venv, or system Python)
    and delegate all arguments to `python -m circuitforge_core.manage`.
    """
    from importlib.resources import files as _res_files

    target = root.resolve()
    templates_pkg = "circuitforge_core.manage.templates"

    for filename in ("manage.sh", "manage.ps1"):
        dest = target / filename
        if dest.exists() and not force:
            typer.echo(f"  skipped {filename} (already exists — use --force to overwrite)")
            continue
        content = (_res_files(templates_pkg) / filename).read_text()
        dest.write_text(content)
        if filename.endswith(".sh"):
            dest.chmod(dest.stat().st_mode | 0o111)  # make executable
        typer.echo(f"  wrote {dest}")

    toml_example = target / "manage.toml.example"
    if not toml_example.exists() or force:
        content = (_res_files(templates_pkg) / "manage.toml.example").read_text()
        toml_example.write_text(content)
        typer.echo(f"  wrote {toml_example}")

    typer.echo("\nDone. Rename manage.toml.example → manage.toml and edit for your services.")


if __name__ == "__main__":
    app()
