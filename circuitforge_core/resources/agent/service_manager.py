"""
ServiceManager — start/stop Docker containers and processes for cf-orch managed services.

Container naming convention: cf-orch-{service}-{node_id}
"""
from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from typing import Any

from circuitforge_core.resources.profiles.schema import DockerSpec, GpuProfile, ProcessSpec


def _expand_volume(v: str) -> str:
    """Expand bash-style volume strings including ${VAR:-default} and $VAR."""
    def _sub(m: re.Match) -> str:  # type: ignore[type-arg]
        var, default = m.group(1), m.group(2) or ""
        return os.environ.get(var) or default
    v = re.sub(r"\$\{(\w+)(?::-(.*?))?\}", _sub, v)
    v = re.sub(r"\$(\w+)", lambda m: os.environ.get(m.group(1), m.group(0)), v)
    return v


class ServiceManager:
    def __init__(
        self,
        node_id: str,
        profile: GpuProfile,
        advertise_host: str = "127.0.0.1",
    ) -> None:
        self.node_id = node_id
        self.profile = profile
        self.advertise_host = advertise_host
        self._procs: dict[str, Any] = {}

    def container_name(self, service: str) -> str:
        return f"cf-orch-{service}-{self.node_id}"

    def _get_spec(self, service: str) -> DockerSpec | ProcessSpec | None:
        svc = self.profile.services.get(service)
        if svc is None:
            return None
        return svc.managed

    def is_running(self, service: str) -> bool:
        spec = self._get_spec(service)
        if spec is None:
            return False
        if isinstance(spec, DockerSpec):
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Running}}",
                        self.container_name(service),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout.strip() == "true"
            except subprocess.CalledProcessError:
                return False
        if isinstance(spec, ProcessSpec):
            proc = self._procs.get(service)
            if proc is None or proc.poll() is not None:
                return False
            import socket
            try:
                with socket.create_connection(("127.0.0.1", spec.host_port), timeout=1):
                    return True
            except OSError:
                return False
        return False

    def start(self, service: str, gpu_id: int, params: dict[str, str]) -> str:
        spec = self._get_spec(service)
        if spec is None:
            raise ValueError(f"Service {service!r} not in profile or has no managed spec")

        if self.is_running(service):
            return f"http://{self.advertise_host}:{spec.host_port}"

        if isinstance(spec, DockerSpec):
            expanded_volumes = [_expand_volume(v) for v in spec.volumes]

            filler: dict[str, str] = defaultdict(str, params)
            expanded_command = spec.command_template.format_map(filler).split()

            cmd = [
                "docker", "run", "-d", "--rm",
                "--name", self.container_name(service),
                "--runtime", spec.runtime,
                "--gpus", f"device={gpu_id}",
                "--ipc", spec.ipc,
                "-p", f"{spec.host_port}:{spec.port}",
            ]
            for vol in expanded_volumes:
                cmd += ["-v", vol]
            for key, val in spec.env.items():
                cmd += ["-e", f"{key}={val}"]
            cmd.append(spec.image)
            cmd.extend(expanded_command)

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"http://{self.advertise_host}:{spec.host_port}"

        if isinstance(spec, ProcessSpec):
            import shlex
            import subprocess as _sp

            filler = defaultdict(str, params)
            filler.setdefault("port", str(spec.port))
            filler.setdefault("gpu_id", str(gpu_id))
            args_expanded = spec.args_template.format_map(filler).split()

            cmd = [spec.exec_path] + args_expanded
            env = {**__import__("os").environ}
            proc = _sp.Popen(
                cmd,
                cwd=spec.cwd or None,
                env=env,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            self._procs[service] = proc
            return f"http://{self.advertise_host}:{spec.host_port}"

        raise NotImplementedError(f"Unknown spec type: {type(spec)}")

    def stop(self, service: str) -> bool:
        spec = self._get_spec(service)
        if spec is None:
            return False
        if isinstance(spec, DockerSpec):
            try:
                subprocess.run(
                    ["docker", "stop", self.container_name(service)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        if isinstance(spec, ProcessSpec):
            proc = self._procs.pop(service, None)
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
            return True
        return False

    def list_running(self) -> list[str]:
        return [svc for svc in self.profile.services if self.is_running(svc)]

    def get_url(self, service: str) -> str | None:
        spec = self._get_spec(service)
        if spec is None or not self.is_running(service):
            return None
        return f"http://{self.advertise_host}:{spec.host_port}"
