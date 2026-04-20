"""
Pydantic models for the job_quality trust scorer.

MIT licensed — no LLM calls, no network calls, no file I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobListing(BaseModel):
    """Input data sourced directly from a job board scraper or ATS export."""

    # Core identity
    title: str = ""
    company: str = ""
    location: str = ""
    state_code: str = ""  # Two-letter US state code, e.g. "CA"

    # Salary / compensation
    salary_min: float | None = None
    salary_max: float | None = None
    salary_text: str = ""  # Raw salary string from the listing

    # Posting metadata
    posted_at: datetime | None = None
    repost_count: int = 0  # Times the same listing has been reposted
    applicant_count: int | None = None
    is_staffing_agency: bool = False
    is_always_open: bool = False  # Evergreen/always-accepting flag

    # Content
    description: str = ""
    requirements: list[str] = Field(default_factory=list)
    ats_url: str = ""  # ATS apply URL (Greenhouse, Lever, Workday, etc.)

    # Signals from scraper enrichment
    weekend_posted: bool = False  # Posted on Saturday or Sunday


class JobEnrichment(BaseModel):
    """Optional enrichment data gathered outside the listing (news, history, etc.)."""

    has_layoff_news: bool = False  # Recent layoff news for this company
    avg_response_days: float | None = None  # Average recruiter response time (days)
    no_response_rate: float | None = None  # Fraction of applicants with no response (0–1)


class SignalResult(BaseModel):
    """Output of a single signal function."""

    name: str
    triggered: bool
    weight: float
    penalty: float  # weight * triggered (0.0 when not triggered)
    detail: str = ""  # Human-readable explanation


class JobQualityScore(BaseModel):
    """Aggregated trust score for a job listing."""

    trust_score: float  # 0.0 (low trust) – 1.0 (high trust)
    confidence: float  # 0.0 – 1.0: fraction of signals with available evidence
    signals: list[SignalResult]
    raw_penalty: float  # Sum of triggered weights before clamping
    metadata: dict[str, Any] = Field(default_factory=dict)
