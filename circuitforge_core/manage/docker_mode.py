"""
circuitforge_core.manage.docker_mode — Docker Compose wrapper.

All commands delegate to `docker compose` (v2 plugin syntax).
Falls back to `docker-compose` (v1 standalone) if the plugin is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .config import ManageConfig


def _compose_bin() -> list[str]:
    """Return the docker compose command as a list (handles v1/v2 difference)."""
    # Docker Compose v2: `docker compose` (space, built-in plugin)
    # Docker Compose v1: `docker-compose` (hyphen, standalone binary)
    if shutil.which("docker"):
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("Neither 'docker' nor 'docker-compose' found on PATH")


def docker_available() -> bool:
    """Return True if Docker is reachable (docker info succeeds)."""
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except Exception:
        return False


class DockerManager:
    """
    Wraps `docker compose` for a single product directory.

    Args:
        config:   ManageConfig for the current product.
        root:     Product root directory (where compose file lives).
    """

    def __init__(self, config: ManageConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self._compose_file = root / config.docker.compose_file

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        cmd = [
            *_compose_bin(),
            "-f", str(self._compose_file),
            "-p", self.config.docker.project or self.config.app_name,
            *args,
        ]
        return subprocess.run(cmd, cwd=self.root, check=check)

    def _stream(self, *args: str) -> None:
        """Run a compose command, streaming output directly to the terminal."""
        cmd = [
            *_compose_bin(),
            "-f", str(self._compose_file),
            "-p", self.config.docker.project or self.config.app_name,
            *args,
        ]
        with subprocess.Popen(cmd, cwd=self.root) as proc:
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()

    def compose_file_exists(self) -> bool:
        return self._compose_file.exists()

    def start(self, service: str = "") -> None:
        args = ["up", "-d", "--build"]
        if service:
            args.append(service)
        self._run(*args)

    def stop(self, service: str = "") -> None:
        if service:
            self._run("stop", service)
        else:
            self._run("down")

    def restart(self, service: str = "") -> None:
        args = ["restart"]
        if service:
            args.append(service)
        self._run(*args)

    def status(self) -> None:
        self._run("ps", check=False)

    def logs(self, service: str = "", follow: bool = True) -> None:
        args = ["logs"]
        if follow:
            args.append("-f")
        if service:
            args.append(service)
        self._stream(*args)

    def build(self, no_cache: bool = False) -> None:
        args = ["build"]
        if no_cache:
            args.append("--no-cache")
        self._run(*args)
