"""Tests for circuitforge_core.preferences path utilities."""
import pytest
from circuitforge_core.preferences import get_path, set_path


class TestGetPath:
    def test_top_level_key(self):
        assert get_path({"a": 1}, "a") == 1

    def test_nested_key(self):
        data = {"affiliate": {"opt_out": False}}
        assert get_path(data, "affiliate.opt_out") is False

    def test_deeply_nested(self):
        data = {"affiliate": {"byok_ids": {"ebay": "my-tag"}}}
        assert get_path(data, "affiliate.byok_ids.ebay") == "my-tag"

    def test_missing_key_returns_default(self):
        assert get_path({}, "missing", default="x") == "x"

    def test_missing_nested_returns_default(self):
        assert get_path({"a": {}}, "a.b.c", default=42) == 42

    def test_default_is_none_when_omitted(self):
        assert get_path({}, "nope") is None

    def test_non_dict_intermediate_returns_default(self):
        assert get_path({"a": "string"}, "a.b", default="d") == "d"


class TestSetPath:
    def test_top_level_key(self):
        result = set_path({}, "opt_out", True)
        assert result == {"opt_out": True}

    def test_nested_key_created(self):
        result = set_path({}, "affiliate.opt_out", True)
        assert result == {"affiliate": {"opt_out": True}}

    def test_deeply_nested(self):
        result = set_path({}, "affiliate.byok_ids.ebay", "my-tag")
        assert result == {"affiliate": {"byok_ids": {"ebay": "my-tag"}}}

    def test_preserves_sibling_keys(self):
        data = {"affiliate": {"opt_out": False, "byok_ids": {}}}
        result = set_path(data, "affiliate.opt_out", True)
        assert result["affiliate"]["opt_out"] is True
        assert result["affiliate"]["byok_ids"] == {}

    def test_preserves_unrelated_top_level_keys(self):
        data = {"other": "value", "affiliate": {"opt_out": False}}
        result = set_path(data, "affiliate.opt_out", True)
        assert result["other"] == "value"

    def test_does_not_mutate_original(self):
        data = {"affiliate": {"opt_out": False}}
        set_path(data, "affiliate.opt_out", True)
        assert data["affiliate"]["opt_out"] is False

    def test_overwrites_existing_value(self):
        data = {"affiliate": {"byok_ids": {"ebay": "old-tag"}}}
        result = set_path(data, "affiliate.byok_ids.ebay", "new-tag")
        assert result["affiliate"]["byok_ids"]["ebay"] == "new-tag"

    def test_non_dict_intermediate_replaced(self):
        data = {"affiliate": "not-a-dict"}
        result = set_path(data, "affiliate.opt_out", True)
        assert result == {"affiliate": {"opt_out": True}}

    def test_roundtrip_get_after_set(self):
        prefs = {}
        prefs = set_path(prefs, "affiliate.opt_out", True)
        prefs = set_path(prefs, "affiliate.byok_ids.ebay", "tag-123")
        assert get_path(prefs, "affiliate.opt_out") is True
        assert get_path(prefs, "affiliate.byok_ids.ebay") == "tag-123"


import os
import tempfile
from pathlib import Path
from circuitforge_core.preferences.store import LocalFileStore


class TestLocalFileStore:
    def _store(self, tmp_path) -> LocalFileStore:
        return LocalFileStore(prefs_path=tmp_path / "preferences.yaml")

    def test_get_returns_default_when_file_missing(self, tmp_path):
        store = self._store(tmp_path)
        assert store.get(user_id=None, path="affiliate.opt_out", default=False) is False

    def test_set_then_get_roundtrip(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path="affiliate.opt_out", value=True)
        assert store.get(user_id=None, path="affiliate.opt_out", default=False) is True

    def test_set_nested_path(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path="affiliate.byok_ids.ebay", value="my-tag")
        assert store.get(user_id=None, path="affiliate.byok_ids.ebay") == "my-tag"

    def test_set_preserves_sibling_keys(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path="affiliate.opt_out", value=False)
        store.set(user_id=None, path="affiliate.byok_ids.ebay", value="tag")
        assert store.get(user_id=None, path="affiliate.opt_out") is False
        assert store.get(user_id=None, path="affiliate.byok_ids.ebay") == "tag"

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "deep" / "nested" / "preferences.yaml"
        store = LocalFileStore(prefs_path=deep_path)
        store.set(user_id=None, path="x", value=1)
        assert deep_path.exists()

    def test_user_id_ignored_for_local_store(self, tmp_path):
        """LocalFileStore is single-user; user_id is accepted but ignored."""
        store = self._store(tmp_path)
        store.set(user_id="u123", path="affiliate.opt_out", value=True)
        assert store.get(user_id="u456", path="affiliate.opt_out", default=False) is True


from circuitforge_core.preferences import get_user_preference, set_user_preference
from circuitforge_core.preferences.accessibility import (
    is_reduced_motion_preferred,
    is_high_contrast,
    get_font_size,
    is_screen_reader_mode,
    set_reduced_motion,
    PREF_REDUCED_MOTION,
    PREF_HIGH_CONTRAST,
    PREF_FONT_SIZE,
    PREF_SCREEN_READER,
)


class TestAccessibilityPreferences:
    def _store(self, tmp_path) -> LocalFileStore:
        return LocalFileStore(prefs_path=tmp_path / "preferences.yaml")

    def test_reduced_motion_default_false(self, tmp_path):
        store = self._store(tmp_path)
        assert is_reduced_motion_preferred(store=store) is False

    def test_set_reduced_motion_persists(self, tmp_path):
        store = self._store(tmp_path)
        set_reduced_motion(True, store=store)
        assert is_reduced_motion_preferred(store=store) is True

    def test_reduced_motion_false_roundtrip(self, tmp_path):
        store = self._store(tmp_path)
        set_reduced_motion(True, store=store)
        set_reduced_motion(False, store=store)
        assert is_reduced_motion_preferred(store=store) is False

    def test_high_contrast_default_false(self, tmp_path):
        store = self._store(tmp_path)
        assert is_high_contrast(store=store) is False

    def test_high_contrast_set_and_read(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path=PREF_HIGH_CONTRAST, value=True)
        assert is_high_contrast(store=store) is True

    def test_font_size_default(self, tmp_path):
        store = self._store(tmp_path)
        assert get_font_size(store=store) == "default"

    def test_font_size_large(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path=PREF_FONT_SIZE, value="large")
        assert get_font_size(store=store) == "large"

    def test_font_size_xlarge(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path=PREF_FONT_SIZE, value="xlarge")
        assert get_font_size(store=store) == "xlarge"

    def test_font_size_invalid_falls_back_to_default(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path=PREF_FONT_SIZE, value="gigantic")
        assert get_font_size(store=store) == "default"

    def test_screen_reader_mode_default_false(self, tmp_path):
        store = self._store(tmp_path)
        assert is_screen_reader_mode(store=store) is False

    def test_screen_reader_mode_set(self, tmp_path):
        store = self._store(tmp_path)
        store.set(user_id=None, path=PREF_SCREEN_READER, value=True)
        assert is_screen_reader_mode(store=store) is True

    def test_preferences_are_independent(self, tmp_path):
        """Setting one a11y pref doesn't affect others."""
        store = self._store(tmp_path)
        set_reduced_motion(True, store=store)
        assert is_high_contrast(store=store) is False
        assert get_font_size(store=store) == "default"
        assert is_screen_reader_mode(store=store) is False

    def test_user_id_threaded_through(self, tmp_path):
        """user_id param is accepted (LocalFileStore ignores it, but must not error)."""
        store = self._store(tmp_path)
        set_reduced_motion(True, user_id="u999", store=store)
        assert is_reduced_motion_preferred(user_id="u999", store=store) is True

    def test_accessibility_exported_from_package(self):
        from circuitforge_core.preferences import accessibility
        assert hasattr(accessibility, "is_reduced_motion_preferred")
        assert hasattr(accessibility, "PREF_REDUCED_MOTION")


class TestPreferenceHelpers:
    def _store(self, tmp_path) -> LocalFileStore:
        return LocalFileStore(prefs_path=tmp_path / "preferences.yaml")

    def test_get_returns_default_when_unset(self, tmp_path):
        store = self._store(tmp_path)
        result = get_user_preference(user_id=None, path="affiliate.opt_out",
                                     default=False, store=store)
        assert result is False

    def test_set_then_get(self, tmp_path):
        store = self._store(tmp_path)
        set_user_preference(user_id=None, path="affiliate.opt_out", value=True, store=store)
        result = get_user_preference(user_id=None, path="affiliate.opt_out",
                                     default=False, store=store)
        assert result is True

    def test_default_store_is_local(self, tmp_path, monkeypatch):
        """When no store is passed, helpers use LocalFileStore at default path."""
        from circuitforge_core.preferences import store as store_module
        local = self._store(tmp_path)
        monkeypatch.setattr(store_module, "_DEFAULT_STORE", local)
        set_user_preference(user_id=None, path="x.y", value=42)
        assert get_user_preference(user_id=None, path="x.y") == 42


# ── Currency preference tests ─────────────────────────────────────────────────

from circuitforge_core.preferences.currency import (
    PREF_CURRENCY_CODE,
    DEFAULT_CURRENCY_CODE,
    get_currency_code,
    set_currency_code,
    format_currency,
    _fallback_format,
)


class TestCurrencyPreference:
    def _store(self, tmp_path) -> LocalFileStore:
        return LocalFileStore(prefs_path=tmp_path / "preferences.yaml")

    def test_default_is_usd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURRENCY_DEFAULT", raising=False)
        store = self._store(tmp_path)
        assert get_currency_code(store=store) == "USD"

    def test_set_then_get(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURRENCY_DEFAULT", raising=False)
        store = self._store(tmp_path)
        set_currency_code("GBP", store=store)
        assert get_currency_code(store=store) == "GBP"

    def test_stored_code_uppercased_on_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURRENCY_DEFAULT", raising=False)
        store = self._store(tmp_path)
        set_currency_code("eur", store=store)
        assert get_currency_code(store=store) == "EUR"

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURRENCY_DEFAULT", "CAD")
        store = self._store(tmp_path)
        # No stored preference — env var kicks in
        assert get_currency_code(store=store) == "CAD"

    def test_stored_preference_beats_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURRENCY_DEFAULT", "CAD")
        store = self._store(tmp_path)
        set_currency_code("AUD", store=store)
        assert get_currency_code(store=store) == "AUD"

    def test_user_id_threaded_through(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURRENCY_DEFAULT", raising=False)
        store = self._store(tmp_path)
        set_currency_code("JPY", user_id="u42", store=store)
        assert get_currency_code(user_id="u42", store=store) == "JPY"

    def test_pref_key_constant(self):
        assert PREF_CURRENCY_CODE == "currency.code"

    def test_default_constant(self):
        assert DEFAULT_CURRENCY_CODE == "USD"

    def test_currency_exported_from_package(self):
        from circuitforge_core.preferences import currency
        assert hasattr(currency, "get_currency_code")
        assert hasattr(currency, "set_currency_code")
        assert hasattr(currency, "format_currency")
        assert hasattr(currency, "PREF_CURRENCY_CODE")


class TestFallbackFormat:
    """Tests for _fallback_format — the no-babel code path."""

    def test_usd_basic(self):
        assert _fallback_format(12.5, "USD") == "$12.50"

    def test_gbp_symbol(self):
        assert _fallback_format(12.5, "GBP") == "£12.50"

    def test_eur_symbol(self):
        assert _fallback_format(12.5, "EUR") == "€12.50"

    def test_jpy_no_decimals(self):
        assert _fallback_format(1500, "JPY") == "¥1,500"

    def test_krw_no_decimals(self):
        assert _fallback_format(10000, "KRW") == "₩10,000"

    def test_thousands_separator(self):
        result = _fallback_format(1234567.89, "USD")
        assert result == "$1,234,567.89"

    def test_zero_amount(self):
        assert _fallback_format(0, "USD") == "$0.00"

    def test_unknown_currency_uses_code_prefix(self):
        result = _fallback_format(10.5, "XYZ")
        assert "XYZ" in result
        assert "10.50" in result

    def test_cad_symbol(self):
        assert _fallback_format(99.99, "CAD") == "CA$99.99"

    def test_inr_symbol(self):
        assert _fallback_format(500.0, "INR") == "₹500.00"


class TestFormatCurrency:
    """Integration tests for format_currency() — exercises the full dispatch."""

    def test_returns_string(self):
        result = format_currency(12.5, "USD")
        assert isinstance(result, str)

    def test_contains_amount(self):
        result = format_currency(12.5, "GBP")
        assert "12" in result

    def test_usd_includes_dollar(self):
        result = format_currency(10.0, "USD")
        assert "$" in result or "USD" in result

    def test_gbp_includes_pound(self):
        result = format_currency(10.0, "GBP")
        # babel gives "£10.00", fallback gives "£10.00" — both contain £
        assert "£" in result or "GBP" in result

    def test_jpy_no_cents(self):
        # JPY has 0 decimal places — both babel and fallback should omit .00
        result = format_currency(1000, "JPY")
        assert ".00" not in result

    def test_currency_code_case_insensitive(self):
        upper = format_currency(10.0, "USD")
        lower = format_currency(10.0, "usd")
        assert upper == lower
