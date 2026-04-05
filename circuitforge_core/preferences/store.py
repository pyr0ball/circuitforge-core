"""Preference store backends.

``LocalFileStore`` reads and writes a single YAML file at a configurable
path (default: ``~/.config/circuitforge/preferences.yaml``).

The ``PreferenceStore`` protocol describes the interface any backend must
satisfy. The Heimdall cloud backend will implement the same protocol once
Heimdall#5 (user_preferences column) lands — products swap backends by
passing a different store instance.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .paths import get_path, set_path

logger = logging.getLogger(__name__)

_DEFAULT_PREFS_PATH = Path.home() / ".config" / "circuitforge" / "preferences.yaml"


@runtime_checkable
class PreferenceStore(Protocol):
    """Read/write interface for user preferences.

    ``user_id`` is passed through for cloud backends that store per-user
    data.  Local single-user backends accept it but ignore it.
    """

    def get(self, user_id: str | None, path: str, default: Any = None) -> Any:
        """Return the value at *path*, or *default* if missing."""
        ...

    def set(self, user_id: str | None, path: str, value: Any) -> None:
        """Persist *value* at *path*."""
        ...


class LocalFileStore:
    """Single-user preference store backed by a YAML file.

    Thread-safe for typical single-process use (reads the file on every
    ``get`` call, writes atomically via a temp-file rename on ``set``).
    Not suitable for concurrent multi-process writes.
    """

    def __init__(self, prefs_path: Path = _DEFAULT_PREFS_PATH) -> None:
        self._path = Path(prefs_path)

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            import yaml  # type: ignore[import]
            text = self._path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("preferences: could not read %s: %s", self._path, exc)
            return {}

    def _save(self, data: dict) -> None:
        import yaml  # type: ignore[import]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
        tmp.replace(self._path)

    def get(self, user_id: str | None, path: str, default: Any = None) -> Any:  # noqa: ARG002
        return get_path(self._load(), path, default=default)

    def set(self, user_id: str | None, path: str, value: Any) -> None:  # noqa: ARG002
        self._save(set_path(self._load(), path, value))


_DEFAULT_STORE: PreferenceStore = LocalFileStore()
