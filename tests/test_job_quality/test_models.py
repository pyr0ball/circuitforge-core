"""Tests for job_quality Pydantic models — construction, defaults, and field types."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from circuitforge_core.job_quality.models import (
    JobEnrichment,
    JobListing,
    JobQualityScore,
    SignalResult,
)


class TestJobListing:
    def test_minimal_construction(self):
        listing = JobListing()
        assert listing.title == ""
        assert listing.requirements == []
        assert listing.salary_min is None

    def test_full_construction(self):
        listing = JobListing(
            title="Staff Engineer",
            company="Acme Corp",
            location="Remote",
            state_code="CA",
            salary_min=150_000,
            salary_max=200_000,
            salary_text="$150k–$200k",
            posted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            repost_count=2,
            applicant_count=50,
            is_staffing_agency=False,
            is_always_open=False,
            description="A real job description with meaningful content.",
            requirements=["Python", "Go"],
            ats_url="https://jobs.lever.co/acme/123",
            weekend_posted=False,
        )
        assert listing.salary_min == 150_000
        assert listing.state_code == "CA"
        assert len(listing.requirements) == 2

    def test_repost_count_defaults_zero(self):
        assert JobListing().repost_count == 0

    def test_requirements_is_independent_list(self):
        a = JobListing(requirements=["Python"])
        b = JobListing(requirements=["Go"])
        assert a.requirements != b.requirements


class TestJobEnrichment:
    def test_defaults(self):
        e = JobEnrichment()
        assert e.has_layoff_news is False
        assert e.avg_response_days is None
        assert e.no_response_rate is None

    def test_with_data(self):
        e = JobEnrichment(has_layoff_news=True, no_response_rate=0.75)
        assert e.has_layoff_news is True
        assert e.no_response_rate == 0.75


class TestSignalResult:
    def test_construction(self):
        r = SignalResult(name="listing_age", triggered=True, weight=0.25, penalty=0.25, detail="30 days old.")
        assert r.penalty == 0.25

    def test_not_triggered_zero_penalty(self):
        r = SignalResult(name="staffing_agency", triggered=False, weight=0.15, penalty=0.0)
        assert r.penalty == 0.0

    def test_detail_defaults_empty(self):
        r = SignalResult(name="x", triggered=False, weight=0.1, penalty=0.0)
        assert r.detail == ""


class TestJobQualityScore:
    def _make_signal(self, triggered: bool, weight: float) -> SignalResult:
        return SignalResult(
            name="test",
            triggered=triggered,
            weight=weight,
            penalty=weight if triggered else 0.0,
        )

    def test_construction(self):
        score = JobQualityScore(
            trust_score=0.75,
            confidence=0.9,
            signals=[self._make_signal(True, 0.25)],
            raw_penalty=0.25,
        )
        assert score.trust_score == 0.75
        assert score.confidence == 0.9
        assert score.raw_penalty == 0.25

    def test_metadata_defaults_empty(self):
        score = JobQualityScore(
            trust_score=1.0, confidence=1.0, signals=[], raw_penalty=0.0
        )
        assert score.metadata == {}
