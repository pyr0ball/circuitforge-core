"""
Individual signal functions for the job_quality trust scorer.

Each function takes a JobListing and optional JobEnrichment and returns a SignalResult.
All signals are pure functions: no I/O, no LLM calls, no side effects.

MIT licensed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from circuitforge_core.job_quality.models import JobEnrichment, JobListing, SignalResult

# US states with salary transparency laws (as of 2026)
_SALARY_TRANSPARENCY_STATES = {"CO", "CA", "NY", "WA", "IL", "MA"}

# ATS providers whose apply URLs are commonly associated with high ghosting rates
_GHOSTING_ATS_PATTERNS = ("lever.co", "greenhouse.io", "workday.com", "icims.com", "taleo.net")

# Threshold for "always open" detection: repost every N days for M months
_ALWAYS_OPEN_REPOST_THRESHOLD = 3

# Requirement count above which a listing is considered overloaded
_REQUIREMENT_OVERLOAD_COUNT = 12

# Vagueness: description length below this suggests bare-minimum content
_VAGUE_DESCRIPTION_CHARS = 400

# Applicant count above which competition is considered very high
_HIGH_APPLICANT_THRESHOLD = 200

# Listing age above which staleness is likely
_STALE_DAYS = 30

# Response rate above which the role is considered a high-ghosting source
_NO_RESPONSE_RATE_THRESHOLD = 0.60


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# High-weight signals (0.15 – 0.25)
# ---------------------------------------------------------------------------


def listing_age(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Listing older than 30 days is likely stale or already filled."""
    weight = 0.25
    if listing.posted_at is None:
        return SignalResult(name="listing_age", triggered=False, weight=weight, penalty=0.0,
                            detail="No posted_at date available.")
    age_days = (_now() - listing.posted_at.astimezone(timezone.utc)).days
    triggered = age_days > _STALE_DAYS
    return SignalResult(
        name="listing_age",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"Listing is {age_days} days old (threshold: {_STALE_DAYS}).",
    )


def repost_detected(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Listing has been reposted multiple times — a strong ghost-job indicator."""
    weight = 0.25
    triggered = listing.repost_count >= _ALWAYS_OPEN_REPOST_THRESHOLD
    return SignalResult(
        name="repost_detected",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"Repost count: {listing.repost_count} (threshold: {_ALWAYS_OPEN_REPOST_THRESHOLD}).",
    )


def no_salary_transparency(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """No salary info despite being in a transparency-law state, or generally absent."""
    weight = 0.20
    has_range = listing.salary_min is not None or listing.salary_max is not None
    has_text = bool(listing.salary_text.strip())
    has_salary = has_range or has_text
    in_transparency_state = listing.state_code.upper() in _SALARY_TRANSPARENCY_STATES

    if not has_salary:
        if in_transparency_state:
            detail = (f"No salary disclosed despite {listing.state_code} transparency law. "
                      "Possible compliance violation.")
        else:
            detail = "No salary information provided."
        triggered = True
    else:
        triggered = False
        detail = "Salary information present."

    return SignalResult(
        name="no_salary_transparency",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=detail,
    )


def always_open_pattern(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Listing is flagged as always-accepting or evergreen — pipeline filler."""
    weight = 0.20
    triggered = listing.is_always_open
    return SignalResult(
        name="always_open_pattern",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail="Listing marked as always-open/evergreen." if triggered else "Not always-open.",
    )


def staffing_agency(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Posted by a staffing or recruiting agency rather than the hiring company directly."""
    weight = 0.15
    triggered = listing.is_staffing_agency
    return SignalResult(
        name="staffing_agency",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail="Listed by a staffing/recruiting agency." if triggered else "Direct employer listing.",
    )


# ---------------------------------------------------------------------------
# Medium-weight signals (0.08 – 0.12)
# ---------------------------------------------------------------------------


def requirement_overload(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Excessive requirements list suggests a wish-list role or perpetual search."""
    weight = 0.12
    count = len(listing.requirements)
    triggered = count > _REQUIREMENT_OVERLOAD_COUNT
    return SignalResult(
        name="requirement_overload",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"{count} requirements listed (threshold: {_REQUIREMENT_OVERLOAD_COUNT}).",
    )


def layoff_news(listing: JobListing, enrichment: JobEnrichment | None = None) -> SignalResult:
    """Company has recent layoff news — new hires may be at high risk."""
    weight = 0.12
    if enrichment is None:
        return SignalResult(name="layoff_news", triggered=False, weight=weight, penalty=0.0,
                            detail="No enrichment data available.")
    triggered = enrichment.has_layoff_news
    return SignalResult(
        name="layoff_news",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail="Recent layoff news detected for this company." if triggered else "No layoff news found.",
    )


def jd_vagueness(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Job description is suspiciously short — may not represent a real open role."""
    weight = 0.10
    char_count = len(listing.description.strip())
    triggered = char_count < _VAGUE_DESCRIPTION_CHARS
    return SignalResult(
        name="jd_vagueness",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"Description is {char_count} characters (threshold: {_VAGUE_DESCRIPTION_CHARS}).",
    )


def ats_blackhole(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Apply URL routes through a high-volume ATS known for candidate ghosting."""
    weight = 0.10
    url_lower = listing.ats_url.lower()
    matched = next((p for p in _GHOSTING_ATS_PATTERNS if p in url_lower), None)
    triggered = matched is not None
    return SignalResult(
        name="ats_blackhole",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"ATS matches high-ghosting pattern '{matched}'." if triggered else "No high-ghosting ATS detected.",
    )


def high_applicant_count(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Very high applicant count means low odds and possible ghost-collection."""
    weight = 0.08
    if listing.applicant_count is None:
        return SignalResult(name="high_applicant_count", triggered=False, weight=weight, penalty=0.0,
                            detail="Applicant count not available.")
    triggered = listing.applicant_count > _HIGH_APPLICANT_THRESHOLD
    return SignalResult(
        name="high_applicant_count",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"{listing.applicant_count} applicants (threshold: {_HIGH_APPLICANT_THRESHOLD}).",
    )


# ---------------------------------------------------------------------------
# Low-weight signals (0.04 – 0.08)
# ---------------------------------------------------------------------------


def weekend_posted(listing: JobListing, _: JobEnrichment | None = None) -> SignalResult:
    """Posted on a weekend — may indicate bulk/automated ghost-job pipeline posting."""
    weight = 0.04
    if listing.posted_at is None and not listing.weekend_posted:
        return SignalResult(name="weekend_posted", triggered=False, weight=weight, penalty=0.0,
                            detail="No posted_at date available.")
    if listing.weekend_posted:
        triggered = True
    else:
        triggered = listing.posted_at.weekday() >= 5  # type: ignore[union-attr]
    return SignalResult(
        name="weekend_posted",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail="Posted on a weekend." if triggered else "Posted on a weekday.",
    )


def poor_response_history(listing: JobListing, enrichment: JobEnrichment | None = None) -> SignalResult:
    """Company/ATS historically does not respond to applicants."""
    weight = 0.08
    if enrichment is None:
        return SignalResult(name="poor_response_history", triggered=False, weight=weight, penalty=0.0,
                            detail="No enrichment data available.")
    rate = enrichment.no_response_rate
    if rate is None:
        return SignalResult(name="poor_response_history", triggered=False, weight=weight, penalty=0.0,
                            detail="No response rate data available.")
    triggered = rate > _NO_RESPONSE_RATE_THRESHOLD
    return SignalResult(
        name="poor_response_history",
        triggered=triggered,
        weight=weight,
        penalty=weight if triggered else 0.0,
        detail=f"No-response rate: {rate:.0%} (threshold: {_NO_RESPONSE_RATE_THRESHOLD:.0%}).",
    )


# ---------------------------------------------------------------------------
# Signal registry — ordered by weight descending for scorer iteration
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    listing_age,
    repost_detected,
    no_salary_transparency,
    always_open_pattern,
    staffing_agency,
    requirement_overload,
    layoff_news,
    jd_vagueness,
    ats_blackhole,
    high_applicant_count,
    weekend_posted,
    poor_response_history,
]
