"""
score_job: aggregate all signals into a JobQualityScore.

MIT licensed — pure function, no I/O.
"""

from __future__ import annotations

from circuitforge_core.job_quality.models import JobEnrichment, JobListing, JobQualityScore, SignalResult
from circuitforge_core.job_quality.signals import ALL_SIGNALS


def score_job(
    listing: JobListing,
    enrichment: JobEnrichment | None = None,
) -> JobQualityScore:
    """
    Score a job listing for trust/quality.

    Each signal produces a penalty in [0, weight].  The raw penalty is the sum of
    all triggered signal weights.  trust_score = 1 - clamp(raw_penalty, 0, 1).

    confidence reflects what fraction of signals had enough data to evaluate.
    Signals that return triggered=False with a "not available" detail are counted
    as unevaluable — they reduce confidence without adding penalty.
    """
    results: list[SignalResult] = []
    evaluable_count = 0

    for fn in ALL_SIGNALS:
        result = fn(listing, enrichment)
        results.append(result)
        # A signal is evaluable when it either triggered or had data to decide it didn't.
        # Signals that skip due to missing data always set triggered=False AND include
        # "not available" or "No" in their detail.
        if result.triggered or _has_data(result):
            evaluable_count += 1

    raw_penalty = sum(r.penalty for r in results)
    trust_score = max(0.0, min(1.0, 1.0 - raw_penalty))
    confidence = evaluable_count / len(ALL_SIGNALS) if ALL_SIGNALS else 0.0

    return JobQualityScore(
        trust_score=round(trust_score, 4),
        confidence=round(confidence, 4),
        signals=results,
        raw_penalty=round(raw_penalty, 4),
    )


def _has_data(result: SignalResult) -> bool:
    """Return True when the signal's detail indicates it actually evaluated data."""
    skip_phrases = (
        "not available",
        "No enrichment",
        "No posted_at",
        "No response rate",
        "No salary information",
    )
    return not any(phrase.lower() in result.detail.lower() for phrase in skip_phrases)
