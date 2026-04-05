"""Tests for affiliate program registry and URL builders."""
import pytest
from circuitforge_core.affiliates.programs import (
    AffiliateProgram,
    get_program,
    register_program,
    registered_keys,
    _build_ebay_url,
    _build_amazon_url,
)


class TestAffiliateProgram:
    def test_cf_affiliate_id_returns_env_value(self, monkeypatch):
        monkeypatch.setenv("TEST_AFF_ID", "my-id")
        prog = AffiliateProgram(
            name="Test", retailer_key="test",
            env_var="TEST_AFF_ID", build_url=lambda u, i: u
        )
        assert prog.cf_affiliate_id() == "my-id"

    def test_cf_affiliate_id_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_AFF_ID", raising=False)
        prog = AffiliateProgram(
            name="Test", retailer_key="test",
            env_var="TEST_AFF_ID", build_url=lambda u, i: u
        )
        assert prog.cf_affiliate_id() is None

    def test_cf_affiliate_id_returns_none_when_blank(self, monkeypatch):
        monkeypatch.setenv("TEST_AFF_ID", "   ")
        prog = AffiliateProgram(
            name="Test", retailer_key="test",
            env_var="TEST_AFF_ID", build_url=lambda u, i: u
        )
        assert prog.cf_affiliate_id() is None


class TestRegistry:
    def test_builtin_ebay_registered(self):
        assert get_program("ebay") is not None
        assert get_program("ebay").name == "eBay Partner Network"

    def test_builtin_amazon_registered(self):
        assert get_program("amazon") is not None
        assert get_program("amazon").name == "Amazon Associates"

    def test_unknown_key_returns_none(self):
        assert get_program("not_a_retailer") is None

    def test_register_custom_program(self):
        prog = AffiliateProgram(
            name="Custom Shop", retailer_key="customshop",
            env_var="CUSTOM_ID", build_url=lambda u, i: f"{u}?ref={i}"
        )
        register_program(prog)
        assert get_program("customshop") is prog
        assert "customshop" in registered_keys()

    def test_register_overwrites_existing(self):
        prog1 = AffiliateProgram("A", "overwrite_test", "X", lambda u, i: u)
        prog2 = AffiliateProgram("B", "overwrite_test", "Y", lambda u, i: u)
        register_program(prog1)
        register_program(prog2)
        assert get_program("overwrite_test").name == "B"


class TestEbayUrlBuilder:
    def test_appends_params_to_plain_url(self):
        url = _build_ebay_url("https://www.ebay.com/itm/123", "my-campaign")
        assert "campid=my-campaign" in url
        assert "mkcid=1" in url
        assert "mkevt=1" in url
        assert url.startswith("https://www.ebay.com/itm/123?")

    def test_uses_ampersand_when_query_already_present(self):
        url = _build_ebay_url("https://www.ebay.com/itm/123?existing=1", "c1")
        assert url.startswith("https://www.ebay.com/itm/123?existing=1&")
        assert "campid=c1" in url

    def test_does_not_double_encode(self):
        url = _build_ebay_url("https://www.ebay.com/itm/123", "camp-id-1")
        assert "camp-id-1" in url


class TestAmazonUrlBuilder:
    def test_appends_tag_to_plain_url(self):
        url = _build_amazon_url("https://www.amazon.com/dp/B001234567", "cf-kiwi-20")
        assert "tag=cf-kiwi-20" in url

    def test_merges_tag_into_existing_query(self):
        url = _build_amazon_url("https://www.amazon.com/dp/B001234567?ref=sr_1_1", "cf-kiwi-20")
        assert "tag=cf-kiwi-20" in url
        assert "ref=sr_1_1" in url

    def test_replaces_existing_tag(self):
        url = _build_amazon_url("https://www.amazon.com/dp/B001?tag=old-tag-20", "new-tag-20")
        assert "tag=new-tag-20" in url
        assert "old-tag-20" not in url
