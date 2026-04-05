"""Tests for affiliate URL wrapping resolution logic."""
import pytest
from circuitforge_core.affiliates.router import wrap_url


def _pref_store(prefs: dict):
    """Return a get_preference callable backed by a plain dict."""
    def get_preference(user_id, path, default=None):
        keys = path.split(".")
        node = prefs
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k)
            if node is None:
                return default
        return node
    return get_preference


class TestWrapUrlEnvVarMode:
    """No get_preference injected — env-var-only mode."""

    def test_returns_affiliate_url_when_env_set(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "camp123")
        result = wrap_url("https://www.ebay.com/itm/1", retailer="ebay")
        assert "campid=camp123" in result

    def test_returns_plain_url_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("EBAY_AFFILIATE_CAMPAIGN_ID", raising=False)
        result = wrap_url("https://www.ebay.com/itm/1", retailer="ebay")
        assert result == "https://www.ebay.com/itm/1"

    def test_returns_plain_url_for_unknown_retailer(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "camp123")
        result = wrap_url("https://www.example.com/item/1", retailer="unknown_shop")
        assert result == "https://www.example.com/item/1"

    def test_amazon_env_var(self, monkeypatch):
        monkeypatch.setenv("AMAZON_ASSOCIATES_TAG", "cf-kiwi-20")
        result = wrap_url("https://www.amazon.com/dp/B001", retailer="amazon")
        assert "tag=cf-kiwi-20" in result


class TestWrapUrlOptOut:
    """get_preference injected — opt-out enforcement."""

    def test_opted_out_returns_plain_url(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "camp123")
        get_pref = _pref_store({"affiliate": {"opt_out": True}})
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id="u1", get_preference=get_pref,
        )
        assert result == "https://www.ebay.com/itm/1"

    def test_opted_in_returns_affiliate_url(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "camp123")
        get_pref = _pref_store({"affiliate": {"opt_out": False}})
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id="u1", get_preference=get_pref,
        )
        assert "campid=camp123" in result

    def test_no_preference_set_defaults_to_opted_in(self, monkeypatch):
        """Missing opt_out key = opted in (default behaviour per design doc)."""
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "camp123")
        get_pref = _pref_store({})
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id="u1", get_preference=get_pref,
        )
        assert "campid=camp123" in result


class TestWrapUrlByok:
    """BYOK affiliate ID takes precedence over CF's ID."""

    def test_byok_id_used_instead_of_cf_id(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "cf-camp")
        get_pref = _pref_store({
            "affiliate": {
                "opt_out": False,
                "byok_ids": {"ebay": "user-own-camp"},
            }
        })
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id="u1", get_preference=get_pref,
        )
        assert "campid=user-own-camp" in result
        assert "cf-camp" not in result

    def test_byok_only_used_when_present(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "cf-camp")
        get_pref = _pref_store({"affiliate": {"opt_out": False, "byok_ids": {}}})
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id="u1", get_preference=get_pref,
        )
        assert "campid=cf-camp" in result

    def test_byok_without_user_id_not_applied(self, monkeypatch):
        """BYOK requires a user_id — anonymous users get CF's ID."""
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "cf-camp")
        get_pref = _pref_store({
            "affiliate": {"opt_out": False, "byok_ids": {"ebay": "user-own-camp"}}
        })
        result = wrap_url(
            "https://www.ebay.com/itm/1", retailer="ebay",
            user_id=None, get_preference=get_pref,
        )
        assert "campid=cf-camp" in result
