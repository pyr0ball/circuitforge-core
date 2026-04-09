# circuitforge_core/preferences/accessibility.py — a11y preference keys
#
# First-class accessibility preferences so every product UI reads from
# the same store path without each implementing it separately.
#
# All keys use the "accessibility.*" namespace in the preference store.
# Products read these via get_user_preference() or the convenience helpers here.
from __future__ import annotations

from circuitforge_core.preferences import get_user_preference, set_user_preference

# ── Preference key constants ──────────────────────────────────────────────────

PREF_REDUCED_MOTION = "accessibility.prefers_reduced_motion"
PREF_HIGH_CONTRAST  = "accessibility.high_contrast"
PREF_FONT_SIZE      = "accessibility.font_size"          # "default" | "large" | "xlarge"
PREF_SCREEN_READER  = "accessibility.screen_reader_mode"  # reduces decorative content

_DEFAULTS: dict[str, object] = {
    PREF_REDUCED_MOTION: False,
    PREF_HIGH_CONTRAST:  False,
    PREF_FONT_SIZE:      "default",
    PREF_SCREEN_READER:  False,
}


# ── Convenience helpers ───────────────────────────────────────────────────────

def is_reduced_motion_preferred(
    user_id: str | None = None,
    store=None,
) -> bool:
    """
    Return True if the user has requested reduced motion.

    Products must honour this in all animated UI elements: transitions,
    auto-playing content, parallax, loaders. This maps to the CSS
    `prefers-reduced-motion: reduce` media query and is the canonical
    source of truth across all CF product UIs.

    Default: False.
    """
    val = get_user_preference(
        user_id, PREF_REDUCED_MOTION, default=False, store=store
    )
    return bool(val)


def is_high_contrast(user_id: str | None = None, store=None) -> bool:
    """Return True if the user has requested high-contrast mode."""
    return bool(get_user_preference(user_id, PREF_HIGH_CONTRAST, default=False, store=store))


def get_font_size(user_id: str | None = None, store=None) -> str:
    """Return the user's preferred font size: 'default' | 'large' | 'xlarge'."""
    val = get_user_preference(user_id, PREF_FONT_SIZE, default="default", store=store)
    if val not in ("default", "large", "xlarge"):
        return "default"
    return str(val)


def is_screen_reader_mode(user_id: str | None = None, store=None) -> bool:
    """Return True if the user has requested screen reader optimised output."""
    return bool(get_user_preference(user_id, PREF_SCREEN_READER, default=False, store=store))


def set_reduced_motion(
    value: bool,
    user_id: str | None = None,
    store=None,
) -> None:
    """Persist the user's reduced-motion preference."""
    set_user_preference(user_id, PREF_REDUCED_MOTION, value, store=store)
