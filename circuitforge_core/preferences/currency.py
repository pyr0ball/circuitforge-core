# circuitforge_core/preferences/currency.py — currency preference + display formatting
#
# Stores a per-user ISO 4217 currency code and provides format_currency() so every
# product formats prices consistently without rolling its own formatter.
#
# Priority fallback chain for get_currency_code():
#   1. User preference store  ("currency.code")
#   2. CURRENCY_DEFAULT env var
#   3. Hard default: "USD"
#
# format_currency() tries babel for full locale support; falls back to a built-in
# symbol table when babel is not installed (no hard dependency on cf-core).
#
# MIT licensed.
from __future__ import annotations

import os

from circuitforge_core.preferences import get_user_preference, set_user_preference

# ── Preference key constants ──────────────────────────────────────────────────

PREF_CURRENCY_CODE = "currency.code"
DEFAULT_CURRENCY_CODE = "USD"

# ── Built-in symbol table (babel fallback) ────────────────────────────────────
# Covers the currencies most likely to appear across CF product consumers.
# Symbol is prepended; decimal places follow ISO 4217 minor-unit convention.

_CURRENCY_META: dict[str, tuple[str, int]] = {
    # (symbol, decimal_places)
    "USD": ("$",    2),
    "CAD": ("CA$",  2),
    "AUD": ("A$",   2),
    "NZD": ("NZ$",  2),
    "GBP": ("£",    2),
    "EUR": ("€",    2),
    "CHF": ("CHF ", 2),
    "SEK": ("kr",   2),
    "NOK": ("kr",   2),
    "DKK": ("kr",   2),
    "JPY": ("¥",    0),
    "CNY": ("¥",    2),
    "KRW": ("₩",    0),
    "INR": ("₹",    2),
    "BRL": ("R$",   2),
    "MXN": ("$",    2),
    "ZAR": ("R",    2),
    "SGD": ("S$",   2),
    "HKD": ("HK$",  2),
    "THB": ("฿",    2),
    "PLN": ("zł",   2),
    "CZK": ("Kč",   2),
    "HUF": ("Ft",   0),
    "RUB": ("₽",    2),
    "TRY": ("₺",    2),
    "ILS": ("₪",    2),
    "AED": ("د.إ",  2),
    "SAR": ("﷼",    2),
    "CLP": ("$",    0),
    "COP": ("$",    0),
    "ARS": ("$",    2),
    "VND": ("₫",    0),
    "IDR": ("Rp",   0),
    "MYR": ("RM",   2),
    "PHP": ("₱",    2),
}

# ── Preference helpers ────────────────────────────────────────────────────────


def get_currency_code(
    user_id: str | None = None,
    store=None,
) -> str:
    """
    Return the user's preferred ISO 4217 currency code.

    Fallback chain:
    1. Value in preference store at "currency.code"
    2. CURRENCY_DEFAULT environment variable
    3. "USD"
    """
    stored = get_user_preference(user_id, PREF_CURRENCY_CODE, default=None, store=store)
    if stored is not None:
        return str(stored).upper()
    env_default = os.environ.get("CURRENCY_DEFAULT", "").strip().upper()
    if env_default:
        return env_default
    return DEFAULT_CURRENCY_CODE


def set_currency_code(
    currency_code: str,
    user_id: str | None = None,
    store=None,
) -> None:
    """Persist *currency_code* (ISO 4217, e.g. 'GBP') to the preference store."""
    set_user_preference(user_id, PREF_CURRENCY_CODE, currency_code.upper(), store=store)


# ── Formatting ────────────────────────────────────────────────────────────────


def format_currency(
    amount: float,
    currency_code: str,
    locale: str = "en_US",
) -> str:
    """
    Format *amount* as a locale-aware currency string.

    Examples::

        format_currency(12.5, "GBP")          # "£12.50"
        format_currency(1234.99, "USD")        # "$1,234.99"
        format_currency(1500, "JPY")           # "¥1,500"

    Uses ``babel.numbers.format_currency`` when babel is installed, which gives
    full locale-aware grouping, decimal separators, and symbol placement.
    Falls back to a built-in symbol table for the common currencies.

    Args:
        amount:        Numeric amount to format.
        currency_code: ISO 4217 code (e.g. "USD", "GBP", "EUR").
        locale:        BCP 47 locale string (e.g. "en_US", "de_DE"). Only used
                       when babel is available.

    Returns:
        Formatted string, e.g. "£12.50".
    """
    code = currency_code.upper()
    try:
        from babel.numbers import format_currency as babel_format  # type: ignore[import]
        return babel_format(amount, code, locale=locale)
    except ImportError:
        return _fallback_format(amount, code)


def _fallback_format(amount: float, code: str) -> str:
    """Format without babel using the built-in symbol table."""
    symbol, decimals = _CURRENCY_META.get(code, (f"{code} ", 2))
    # Group thousands with commas
    if decimals == 0:
        value_str = f"{int(round(amount)):,}"
    else:
        value_str = f"{amount:,.{decimals}f}"
    return f"{symbol}{value_str}"
