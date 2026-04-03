"""
circuitforge_core.manage.config — ManageConfig parsed from manage.toml.

Products drop a manage.toml in their root directory.  manage.py reads it to
discover the app name, compose file, and native service definitions.

Minimal manage.toml (Docker-only):
----------------------------------------------------------------------
[app]
name = "kiwi"
default_url = "http://localhost:8511"
----------------------------------------------------------------------

Full manage.toml (Docker + native services):
----------------------------------------------------------------------
[app]
name = "kiwi"
default_url = "http://localhost:8511"

[docker]
compose_file = "compose.yml"     # default
project = "kiwi"                 # defaults to app.name

[[native.services]]
name = "api"
command = "uvicorn app.main:app --host 0.0.0.0 --port 8512"
port = 8512

[[native.services]]
name = "frontend"
command = "npm run preview -- --host --port 8511"
port = 8511
cwd = "frontend"
----------------------------------------------------------------------
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

_DEFAULT_COMPOSE_FILE = "compose.yml"


@dataclass
class NativeService:
    """One process to manage in native mode."""
    name: str
    command: str                        # shell command string
    port: int = 0                       # for status / open URL
    cwd: str = ""                       # relative to project root; "" = root
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class DockerConfig:
    compose_file: str = _DEFAULT_COMPOSE_FILE
    project: str = ""                   # docker compose -p; defaults to app name


@dataclass
class ManageConfig:
    app_name: str
    default_url: str = ""
    docker: DockerConfig = field(default_factory=DockerConfig)
    services: list[NativeService] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "ManageConfig":
        """Load from a manage.toml file."""
        raw = tomllib.loads(path.read_text())
        app = raw.get("app", {})
        name = app.get("name") or path.parent.name  # fallback to directory name
        default_url = app.get("default_url", "")

        docker_raw = raw.get("docker", {})
        docker = DockerConfig(
            compose_file=docker_raw.get("compose_file", _DEFAULT_COMPOSE_FILE),
            project=docker_raw.get("project", name),
        )

        services: list[NativeService] = []
        for svc in raw.get("native", {}).get("services", []):
            services.append(NativeService(
                name=svc["name"],
                command=svc["command"],
                port=svc.get("port", 0),
                cwd=svc.get("cwd", ""),
                env=svc.get("env", {}),
            ))

        return cls(
            app_name=name,
            default_url=default_url,
            docker=docker,
            services=services,
        )

    @classmethod
    def from_cwd(cls, cwd: Path | None = None) -> "ManageConfig":
        """
        Load from manage.toml in cwd, or return a minimal config derived from
        the directory name if no manage.toml exists (Docker-only products work
        without one).
        """
        root = cwd or Path.cwd()
        toml_path = root / "manage.toml"
        if toml_path.exists():
            return cls.load(toml_path)
        # Fallback: infer from directory name, look for compose.yml
        return cls(app_name=root.name)
