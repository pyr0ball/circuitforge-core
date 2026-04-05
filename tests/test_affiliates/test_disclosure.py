"""Tests for affiliate disclosure text."""
import pytest
from circuitforge_core.affiliates.disclosure import get_disclosure_text, BANNER_COPY


class TestGetDisclosureText:
    def test_returns_string_for_known_retailer(self):
        text = get_disclosure_text("ebay")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_ebay_copy_mentions_ebay(self):
        text = get_disclosure_text("ebay")
        assert "eBay" in text

    def test_amazon_copy_mentions_amazon(self):
        text = get_disclosure_text("amazon")
        assert "Amazon" in text

    def test_unknown_retailer_returns_generic(self):
        text = get_disclosure_text("not_a_retailer")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_banner_copy_has_required_keys(self):
        assert "title" in BANNER_COPY
        assert "body" in BANNER_COPY
        assert "opt_out_label" in BANNER_COPY
        assert "dismiss_label" in BANNER_COPY
