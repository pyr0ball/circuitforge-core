"""Env validation and .env loader for CircuitForge products."""
from __future__ import annotations
import os
from pathlib import Path


def require_env(key: str) -> str:
    """Return env var value or raise EnvironmentError with clear message."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable {key!r} is not set. "
            f"Check your .env file."
        )
    return value


def load_env(path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ. Skips missing files."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
