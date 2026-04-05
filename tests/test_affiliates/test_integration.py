"""Integration tests — full wrap_url() round-trip through public API."""
import pytest
from circuitforge_core.affiliates import wrap_url, get_disclosure_text


class TestEbayIntegration:
    def test_full_flow_with_env_var(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "cf-snipe-999")
        url = wrap_url("https://www.ebay.com/itm/987654321", retailer="ebay")
        assert "campid=cf-snipe-999" in url
        assert "mkcid=1" in url
        assert "mkevt=1" in url

    def test_full_flow_with_opt_out(self, monkeypatch):
        monkeypatch.setenv("EBAY_AFFILIATE_CAMPAIGN_ID", "cf-snipe-999")

        def get_pref(user_id, path, default=None):
            if path == "affiliate.opt_out":
                return True
            return default

        result = wrap_url(
            "https://www.ebay.com/itm/987654321",
            retailer="ebay",
            user_id="u99",
            get_preference=get_pref,
        )
        assert result == "https://www.ebay.com/itm/987654321"

    def test_disclosure_text_available(self):
        text = get_disclosure_text("ebay")
        assert "eBay" in text
        assert len(text) > 20


class TestAmazonIntegration:
    def test_full_flow_with_env_var(self, monkeypatch):
        monkeypatch.setenv("AMAZON_ASSOCIATES_TAG", "cf-kiwi-20")
        url = wrap_url("https://www.amazon.com/dp/B00TEST1234", retailer="amazon")
        assert "tag=cf-kiwi-20" in url

    def test_preserves_existing_query_params(self, monkeypatch):
        monkeypatch.setenv("AMAZON_ASSOCIATES_TAG", "cf-kiwi-20")
        url = wrap_url(
            "https://www.amazon.com/dp/B00TEST?ref=sr_1_1&keywords=flour",
            retailer="amazon",
        )
        assert "tag=cf-kiwi-20" in url
        assert "ref=sr_1_1" in url
        assert "keywords=flour" in url


class TestNoEnvVar:
    def test_plain_url_returned_when_no_env_var(self, monkeypatch):
        monkeypatch.delenv("EBAY_AFFILIATE_CAMPAIGN_ID", raising=False)
        monkeypatch.delenv("AMAZON_ASSOCIATES_TAG", raising=False)
        ebay_url = "https://www.ebay.com/itm/1"
        amazon_url = "https://www.amazon.com/dp/B001"
        assert wrap_url(ebay_url, retailer="ebay") == ebay_url
        assert wrap_url(amazon_url, retailer="amazon") == amazon_url
