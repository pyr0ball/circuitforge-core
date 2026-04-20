"""
Tests for score_job() — the aggregating scorer function.

Covers: trust_score math, confidence calculation, clamping,
signal count, enrichment passthrough, edge cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from circuitforge_core.job_quality.models import JobEnrichment, JobListing
from circuitforge_core.job_quality.scorer import score_job
from circuitforge_core.job_quality.signals import ALL_SIGNALS

_NOW = datetime.now(tz=timezone.utc)


def _days_ago(n: int) -> datetime:
    return _NOW - timedelta(days=n)


def _clean_listing() -> JobListing:
    """A listing that should trigger no signals."""
    return JobListing(
        title="Staff Engineer",
        company="Acme Corp",
        state_code="CA",
        salary_min=140_000,
        salary_max=180_000,
        posted_at=_days_ago(3),
        repost_count=0,
        applicant_count=30,
        is_staffing_agency=False,
        is_always_open=False,
        description="X" * 600,
        requirements=["Python", "Go", "SQL"],
        ats_url="https://careers.acme.com/apply/123",
        weekend_posted=False,
    )


def _ghost_listing() -> JobListing:
    """A listing designed to trigger as many signals as possible."""
    return JobListing(
        state_code="",
        posted_at=_days_ago(60),
        repost_count=5,
        is_staffing_agency=True,
        is_always_open=True,
        applicant_count=500,
        requirements=["R"] * 15,
        description="Great opportunity.",
        ats_url="https://jobs.lever.co/ghost/123",
        weekend_posted=True,
    )


class TestScoreJob:
    def test_clean_listing_high_trust(self):
        score = score_job(_clean_listing(), JobEnrichment(has_layoff_news=False, no_response_rate=0.1))
        assert score.trust_score >= 0.85, f"Expected high trust, got {score.trust_score}"

    def test_ghost_listing_low_trust(self):
        score = score_job(_ghost_listing(), JobEnrichment(has_layoff_news=True, no_response_rate=0.9))
        assert score.trust_score <= 0.25, f"Expected low trust, got {score.trust_score}"

    def test_trust_score_clamped_to_1(self):
        score = score_job(JobListing())  # No signals triggered, penalty = 0
        assert score.trust_score <= 1.0

    def test_trust_score_clamped_to_0(self):
        score = score_job(_ghost_listing(), JobEnrichment(has_layoff_news=True, no_response_rate=0.9))
        assert score.trust_score >= 0.0

    def test_returns_all_signals(self):
        score = score_job(JobListing())
        assert len(score.signals) == len(ALL_SIGNALS)

    def test_signal_names_match_registry(self):
        score = score_job(JobListing())
        score_names = {s.name for s in score.signals}
        registry_names = {fn(JobListing()).name for fn in ALL_SIGNALS}
        assert score_names == registry_names

    def test_raw_penalty_equals_sum_of_triggered_weights(self):
        score = score_job(_ghost_listing())
        expected = sum(s.penalty for s in score.signals)
        assert abs(score.raw_penalty - round(expected, 4)) < 1e-6

    def test_trust_score_equals_one_minus_penalty(self):
        score = score_job(_ghost_listing())
        expected = round(max(0.0, 1.0 - score.raw_penalty), 4)
        assert score.trust_score == expected

    def test_confidence_between_0_and_1(self):
        score = score_job(JobListing())
        assert 0.0 <= score.confidence <= 1.0

    def test_no_enrichment_reduces_confidence(self):
        score_no_enrich = score_job(_clean_listing(), None)
        score_with_enrich = score_job(_clean_listing(), JobEnrichment(has_layoff_news=False, no_response_rate=0.1))
        assert score_with_enrich.confidence >= score_no_enrich.confidence

    def test_enrichment_is_passed_to_signals(self):
        enrichment = JobEnrichment(has_layoff_news=True)
        score = score_job(JobListing(), enrichment)
        layoff_signal = next(s for s in score.signals if s.name == "layoff_news")
        assert layoff_signal.triggered is True

    def test_metadata_empty_by_default(self):
        score = score_job(JobListing())
        assert score.metadata == {}

    def test_no_salary_in_transparency_state(self):
        listing = JobListing(state_code="CO", posted_at=_days_ago(1), repost_count=0)
        score = score_job(listing)
        salary_signal = next(s for s in score.signals if s.name == "no_salary_transparency")
        assert salary_signal.triggered is True

    def test_penalty_accumulation_is_additive(self):
        """Each triggered signal adds its weight independently."""
        listing = JobListing(
            is_staffing_agency=True,  # +0.15
            is_always_open=True,      # +0.20
        )
        score = score_job(listing)
        staffing = next(s for s in score.signals if s.name == "staffing_agency")
        always = next(s for s in score.signals if s.name == "always_open_pattern")
        assert staffing.triggered and always.triggered
        assert score.raw_penalty >= staffing.weight + always.weight - 1e-9

    def test_score_is_deterministic(self):
        listing = _ghost_listing()
        enrich = JobEnrichment(has_layoff_news=True, no_response_rate=0.8)
        s1 = score_job(listing, enrich)
        s2 = score_job(listing, enrich)
        assert s1.trust_score == s2.trust_score
        assert s1.raw_penalty == s2.raw_penalty
