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
