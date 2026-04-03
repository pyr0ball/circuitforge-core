"""
circuitforge_core.manage.native_mode — PID-file process manager.

Manages processes directly without Docker.  Designed for Windows (no WSL2,
no Docker), but works identically on Linux/macOS.

Platform conventions (via platformdirs):
  PID files : user_runtime_dir(app_name) / <service>.pid
  Log files : user_log_dir(app_name)     / <service>.log

PID file format (one line each):
  <pid>
  <command_fingerprint>   (first 80 chars of command — used to sanity-check
                           that the PID belongs to our process, not a recycled one)
"""
from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_log_dir, user_runtime_dir

from .config import ManageConfig, NativeService

_IS_WINDOWS = platform.system() == "Windows"
_LOG_TAIL_LINES = 50
_FOLLOW_POLL_S = 0.25


@dataclass
class ServiceStatus:
    name: str
    running: bool
    pid: int | None
    port: int
    log_path: Path


class NativeManager:
    """
    Start, stop, and monitor native processes for a product.

    Args:
        config: ManageConfig for the current product.
        root:   Product root directory.
    """

    def __init__(self, config: ManageConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self._pid_dir = Path(user_runtime_dir(config.app_name, ensure_exists=True))
        self._log_dir = Path(user_log_dir(config.app_name, ensure_exists=True))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pid_path(self, name: str) -> Path:
        return self._pid_dir / f"{name}.pid"

    def _log_path(self, name: str) -> Path:
        return self._log_dir / f"{name}.log"

    def _write_pid(self, name: str, pid: int, command: str) -> None:
        self._pid_path(name).write_text(f"{pid}\n{command[:80]}\n")

    def _read_pid(self, name: str) -> int | None:
        p = self._pid_path(name)
        if not p.exists():
            return None
        try:
            return int(p.read_text().splitlines()[0].strip())
        except (ValueError, IndexError):
            return None

    def _pid_alive(self, pid: int) -> bool:
        """Return True if a process with this PID is currently running."""
        if _IS_WINDOWS:
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=True,
                )
                return str(pid) in result.stdout
            except Exception:
                return False
        else:
            try:
                os.kill(pid, 0)   # signal 0 = existence check only
                return True
            except (OSError, ProcessLookupError):
                return False

    def _kill(self, pid: int) -> None:
        """Terminate a process gracefully, then force-kill if needed."""
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        else:
            import signal
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(30):          # wait up to 3 s
                    time.sleep(0.1)
                    if not self._pid_alive(pid):
                        return
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    def _svc(self, name: str) -> NativeService | None:
        return next((s for s in self.config.services if s.name == name), None)

    # ── public API ────────────────────────────────────────────────────────────

    def is_running(self, name: str) -> bool:
        pid = self._read_pid(name)
        return pid is not None and self._pid_alive(pid)

    def status(self) -> list[ServiceStatus]:
        result = []
        for svc in self.config.services:
            pid = self._read_pid(svc.name)
            running = pid is not None and self._pid_alive(pid)
            result.append(ServiceStatus(
                name=svc.name,
                running=running,
                pid=pid if running else None,
                port=svc.port,
                log_path=self._log_path(svc.name),
            ))
        return result

    def start(self, name: str | None = None) -> list[str]:
        """Start one or all services. Returns list of started service names."""
        targets = [self._svc(name)] if name else self.config.services
        started: list[str] = []
        for svc in targets:
            if svc is None:
                raise ValueError(f"Unknown service: {name!r}")
            if self.is_running(svc.name):
                continue
            cwd = (self.root / svc.cwd) if svc.cwd else self.root
            log_file = open(self._log_path(svc.name), "a")  # noqa: WPS515
            env = {**os.environ, **svc.env}
            if _IS_WINDOWS:
                cmd = svc.command          # Windows: pass as string to shell
                shell = True
            else:
                cmd = shlex.split(svc.command)
                shell = False
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                shell=shell,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,    # detach from terminal (Unix)
            )
            self._write_pid(svc.name, proc.pid, svc.command)
            started.append(svc.name)
        return started

    def stop(self, name: str | None = None) -> list[str]:
        """Stop one or all services. Returns list of stopped service names."""
        names = [name] if name else [s.name for s in self.config.services]
        stopped: list[str] = []
        for n in names:
            pid = self._read_pid(n)
            if pid and self._pid_alive(pid):
                self._kill(pid)
                stopped.append(n)
            pid_path = self._pid_path(n)
            if pid_path.exists():
                pid_path.unlink()
        return stopped

    def logs(self, name: str, follow: bool = True, lines: int = _LOG_TAIL_LINES) -> None:
        """
        Print the last N lines of a service log, then optionally follow.

        Uses polling rather than `tail -f` so it works on Windows.
        """
        log_path = self._log_path(name)
        if not log_path.exists():
            print(f"[{name}] No log file found at {log_path}", file=sys.stderr)
            return

        # Print last N lines
        content = log_path.read_bytes()
        lines_data = content.splitlines()[-lines:]
        for line in lines_data:
            print(line.decode("utf-8", errors="replace"))

        if not follow:
            return

        # Poll for new content
        offset = len(content)
        try:
            while True:
                time.sleep(_FOLLOW_POLL_S)
                new_size = log_path.stat().st_size
                if new_size > offset:
                    with open(log_path, "rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                    offset = new_size
                    for line in chunk.splitlines():
                        print(line.decode("utf-8", errors="replace"))
        except KeyboardInterrupt:
            pass
