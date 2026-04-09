from . import store as store_module
from .paths import get_path, set_path
from .store import LocalFileStore, PreferenceStore


def get_user_preference(
    user_id: str | None,
    path: str,
    default=None,
    store: PreferenceStore | None = None,
):
    """Read a preference value at dot-separated *path*.

    Args:
        user_id:  User identifier (passed to store; local store ignores it).
        path:     Dot-separated preference path, e.g. ``"affiliate.opt_out"``.
        default:  Returned when the path is not set.
        store:    Optional store override; defaults to ``LocalFileStore`` at
                  ``~/.config/circuitforge/preferences.yaml``.
    """
    s = store or store_module._DEFAULT_STORE
    return s.get(user_id=user_id, path=path, default=default)


def set_user_preference(
    user_id: str | None,
    path: str,
    value,
    store: PreferenceStore | None = None,
) -> None:
    """Write *value* at dot-separated *path*.

    Args:
        user_id:  User identifier (passed to store; local store ignores it).
        path:     Dot-separated preference path, e.g. ``"affiliate.byok_ids.ebay"``.
        value:    Value to persist.
        store:    Optional store override; defaults to ``LocalFileStore``.
    """
    s = store or store_module._DEFAULT_STORE
    s.set(user_id=user_id, path=path, value=value)


from . import accessibility as accessibility

__all__ = [
    "get_path", "set_path",
    "get_user_preference", "set_user_preference",
    "LocalFileStore", "PreferenceStore",
    "accessibility",
]
