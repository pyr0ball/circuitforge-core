"""
Unit tests for each individual signal function.

Each signal is exercised for: triggered path, not-triggered path, and (where
applicable) the missing-data / no-enrichment path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from circuitforge_core.job_quality.models import JobEnrichment, JobListing
from circuitforge_core.job_quality.signals import (
    ALL_SIGNALS,
    always_open_pattern,
    ats_blackhole,
    high_applicant_count,
    jd_vagueness,
    layoff_news,
    listing_age,
    no_salary_transparency,
    poor_response_history,
    repost_detected,
    requirement_overload,
    staffing_agency,
    weekend_posted,
)

_NOW = datetime.now(tz=timezone.utc)


def _days_ago(n: int) -> datetime:
    return _NOW - timedelta(days=n)


# ---------------------------------------------------------------------------
# listing_age
# ---------------------------------------------------------------------------


class TestListingAge:
    def test_stale_listing_triggers(self):
        listing = JobListing(posted_at=_days_ago(31))
        result = listing_age(listing)
        assert result.triggered is True
        assert result.penalty == result.weight

    def test_fresh_listing_does_not_trigger(self):
        listing = JobListing(posted_at=_days_ago(5))
        result = listing_age(listing)
        assert result.triggered is False
        assert result.penalty == 0.0

    def test_no_posted_at_returns_not_triggered(self):
        result = listing_age(JobListing())
        assert result.triggered is False
        assert result.penalty == 0.0
        assert "No posted_at" in result.detail

    def test_weight_is_0_25(self):
        assert listing_age(JobListing()).weight == 0.25


# ---------------------------------------------------------------------------
# repost_detected
# ---------------------------------------------------------------------------


class TestRepostDetected:
    def test_high_repost_triggers(self):
        result = repost_detected(JobListing(repost_count=3))
        assert result.triggered is True

    def test_low_repost_does_not_trigger(self):
        result = repost_detected(JobListing(repost_count=1))
        assert result.triggered is False

    def test_zero_repost_does_not_trigger(self):
        result = repost_detected(JobListing(repost_count=0))
        assert result.triggered is False

    def test_weight_is_0_25(self):
        assert repost_detected(JobListing()).weight == 0.25


# ---------------------------------------------------------------------------
# no_salary_transparency
# ---------------------------------------------------------------------------


class TestNoSalaryTransparency:
    def test_no_salary_triggers(self):
        result = no_salary_transparency(JobListing(state_code="TX"))
        assert result.triggered is True

    def test_salary_range_prevents_trigger(self):
        result = no_salary_transparency(JobListing(salary_min=80_000, salary_max=120_000))
        assert result.triggered is False

    def test_salary_text_prevents_trigger(self):
        result = no_salary_transparency(JobListing(salary_text="$90k"))
        assert result.triggered is False

    def test_transparency_state_detail(self):
        result = no_salary_transparency(JobListing(state_code="CA"))
        assert "CA" in result.detail or "transparency" in result.detail.lower()

    def test_weight_is_0_20(self):
        assert no_salary_transparency(JobListing()).weight == 0.20


# ---------------------------------------------------------------------------
# always_open_pattern
# ---------------------------------------------------------------------------


class TestAlwaysOpenPattern:
    def test_always_open_triggers(self):
        result = always_open_pattern(JobListing(is_always_open=True))
        assert result.triggered is True

    def test_not_always_open(self):
        result = always_open_pattern(JobListing(is_always_open=False))
        assert result.triggered is False

    def test_weight_is_0_20(self):
        assert always_open_pattern(JobListing()).weight == 0.20


# ---------------------------------------------------------------------------
# staffing_agency
# ---------------------------------------------------------------------------


class TestStaffingAgency:
    def test_agency_triggers(self):
        result = staffing_agency(JobListing(is_staffing_agency=True))
        assert result.triggered is True

    def test_direct_employer_does_not_trigger(self):
        result = staffing_agency(JobListing(is_staffing_agency=False))
        assert result.triggered is False

    def test_weight_is_0_15(self):
        assert staffing_agency(JobListing()).weight == 0.15


# ---------------------------------------------------------------------------
# requirement_overload
# ---------------------------------------------------------------------------


class TestRequirementOverload:
    def test_overloaded_triggers(self):
        result = requirement_overload(JobListing(requirements=["R"] * 13))
        assert result.triggered is True

    def test_reasonable_requirements_do_not_trigger(self):
        result = requirement_overload(JobListing(requirements=["Python", "Go", "SQL"]))
        assert result.triggered is False

    def test_empty_requirements_does_not_trigger(self):
        result = requirement_overload(JobListing())
        assert result.triggered is False

    def test_weight_is_0_12(self):
        assert requirement_overload(JobListing()).weight == 0.12


# ---------------------------------------------------------------------------
# layoff_news
# ---------------------------------------------------------------------------


class TestLayoffNews:
    def test_layoff_news_triggers(self):
        enrichment = JobEnrichment(has_layoff_news=True)
        result = layoff_news(JobListing(), enrichment)
        assert result.triggered is True

    def test_no_layoff_news_does_not_trigger(self):
        enrichment = JobEnrichment(has_layoff_news=False)
        result = layoff_news(JobListing(), enrichment)
        assert result.triggered is False

    def test_no_enrichment_returns_not_triggered(self):
        result = layoff_news(JobListing(), None)
        assert result.triggered is False
        assert "No enrichment" in result.detail

    def test_weight_is_0_12(self):
        assert layoff_news(JobListing()).weight == 0.12


# ---------------------------------------------------------------------------
# jd_vagueness
# ---------------------------------------------------------------------------


class TestJdVagueness:
    def test_short_description_triggers(self):
        result = jd_vagueness(JobListing(description="Short."))
        assert result.triggered is True

    def test_long_description_does_not_trigger(self):
        result = jd_vagueness(JobListing(description="X" * 500))
        assert result.triggered is False

    def test_empty_description_triggers(self):
        result = jd_vagueness(JobListing(description=""))
        assert result.triggered is True

    def test_weight_is_0_10(self):
        assert jd_vagueness(JobListing()).weight == 0.10


# ---------------------------------------------------------------------------
# ats_blackhole
# ---------------------------------------------------------------------------


class TestAtsBlackhole:
    @pytest.mark.parametrize("url", [
        "https://jobs.lever.co/acme/abc",
        "https://boards.greenhouse.io/acme/jobs/123",
        "https://acme.workday.com/en-US/recruiting/job/123",
        "https://acme.icims.com/jobs/123",
        "https://acme.taleo.net/careersection/123",
    ])
    def test_known_ats_triggers(self, url: str):
        result = ats_blackhole(JobListing(ats_url=url))
        assert result.triggered is True

    def test_direct_url_does_not_trigger(self):
        result = ats_blackhole(JobListing(ats_url="https://careers.acme.com/apply/123"))
        assert result.triggered is False

    def test_empty_url_does_not_trigger(self):
        result = ats_blackhole(JobListing(ats_url=""))
        assert result.triggered is False

    def test_weight_is_0_10(self):
        assert ats_blackhole(JobListing()).weight == 0.10


# ---------------------------------------------------------------------------
# high_applicant_count
# ---------------------------------------------------------------------------


class TestHighApplicantCount:
    def test_high_count_triggers(self):
        result = high_applicant_count(JobListing(applicant_count=201))
        assert result.triggered is True

    def test_low_count_does_not_trigger(self):
        result = high_applicant_count(JobListing(applicant_count=10))
        assert result.triggered is False

    def test_none_count_returns_not_triggered(self):
        result = high_applicant_count(JobListing(applicant_count=None))
        assert result.triggered is False
        assert "not available" in result.detail.lower()

    def test_weight_is_0_08(self):
        assert high_applicant_count(JobListing()).weight == 0.08


# ---------------------------------------------------------------------------
# weekend_posted
# ---------------------------------------------------------------------------


class TestWeekendPosted:
    def test_weekend_flag_triggers(self):
        result = weekend_posted(JobListing(weekend_posted=True))
        assert result.triggered is True

    def test_saturday_date_triggers(self):
        # Find next Saturday
        today = _NOW
        days_until_sat = (5 - today.weekday()) % 7
        sat = today - timedelta(days=(today.weekday() - 5) % 7) if today.weekday() > 5 else today + timedelta(days=days_until_sat)
        # Just use a known Saturday: 2026-04-18
        sat = datetime(2026, 4, 18, tzinfo=timezone.utc)  # Saturday
        result = weekend_posted(JobListing(posted_at=sat, weekend_posted=False))
        assert result.triggered is True

    def test_weekday_does_not_trigger(self):
        # 2026-04-20 is Monday
        mon = datetime(2026, 4, 20, tzinfo=timezone.utc)
        result = weekend_posted(JobListing(posted_at=mon, weekend_posted=False))
        assert result.triggered is False

    def test_no_data_returns_not_triggered(self):
        result = weekend_posted(JobListing(posted_at=None, weekend_posted=False))
        assert result.triggered is False

    def test_weight_is_0_04(self):
        assert weekend_posted(JobListing()).weight == 0.04


# ---------------------------------------------------------------------------
# poor_response_history
# ---------------------------------------------------------------------------


class TestPoorResponseHistory:
    def test_high_no_response_rate_triggers(self):
        enrichment = JobEnrichment(no_response_rate=0.75)
        result = poor_response_history(JobListing(), enrichment)
        assert result.triggered is True

    def test_low_no_response_rate_does_not_trigger(self):
        enrichment = JobEnrichment(no_response_rate=0.30)
        result = poor_response_history(JobListing(), enrichment)
        assert result.triggered is False

    def test_none_rate_returns_not_triggered(self):
        enrichment = JobEnrichment(no_response_rate=None)
        result = poor_response_history(JobListing(), enrichment)
        assert result.triggered is False
        assert "available" in result.detail.lower()

    def test_no_enrichment_returns_not_triggered(self):
        result = poor_response_history(JobListing(), None)
        assert result.triggered is False

    def test_weight_is_0_08(self):
        assert poor_response_history(JobListing()).weight == 0.08


# ---------------------------------------------------------------------------
# ALL_SIGNALS registry
# ---------------------------------------------------------------------------


class TestAllSignalsRegistry:
    def test_has_12_signals(self):
        assert len(ALL_SIGNALS) == 12

    def test_all_callable(self):
        for fn in ALL_SIGNALS:
            assert callable(fn)

    def test_all_return_signal_result(self):
        from circuitforge_core.job_quality.models import SignalResult
        listing = JobListing()
        for fn in ALL_SIGNALS:
            result = fn(listing, None)
            assert isinstance(result, SignalResult)

    def test_signal_names_are_unique(self):
        listing = JobListing()
        names = [fn(listing).name for fn in ALL_SIGNALS]
        assert len(names) == len(set(names))
