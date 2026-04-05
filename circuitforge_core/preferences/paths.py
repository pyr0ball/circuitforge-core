"""Dot-path utilities for reading and writing nested preference dicts.

All operations are immutable: set_path returns a new dict rather than
mutating the input.

Path format: dot-separated keys, e.g. "affiliate.byok_ids.ebay"
"""
from __future__ import annotations

from typing import Any


def get_path(data: dict, path: str, default: Any = None) -> Any:
    """Return the value at *path* inside *data*, or *default* if missing.

    Example::

        prefs = {"affiliate": {"opt_out": False, "byok_ids": {"ebay": "my-id"}}}
        get_path(prefs, "affiliate.byok_ids.ebay")  # "my-id"
        get_path(prefs, "affiliate.missing", default="x")  # "x"
    """
    keys = path.split(".")
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, _SENTINEL)
        if node is _SENTINEL:
            return default
    return node


def set_path(data: dict, path: str, value: Any) -> dict:
    """Return a new dict with *value* written at *path*.

    Intermediate dicts are created as needed; existing values at other paths
    are preserved.  The original *data* dict is never mutated.

    Example::

        prefs = {}
        updated = set_path(prefs, "affiliate.opt_out", True)
        # {"affiliate": {"opt_out": True}}
    """
    keys = path.split(".")
    return _set_recursive(data, keys, value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _set_recursive(node: Any, keys: list[str], value: Any) -> dict:
    if not isinstance(node, dict):
        node = {}
    key, rest = keys[0], keys[1:]
    if rest:
        child = _set_recursive(node.get(key, {}), rest, value)
    else:
        child = value
    return {**node, key: child}
