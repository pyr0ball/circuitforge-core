"""
circuitforge_core.job_quality — deterministic trust scorer for job listings.

MIT licensed.
"""

from circuitforge_core.job_quality.models import (
    JobEnrichment,
    JobListing,
    JobQualityScore,
    SignalResult,
)
from circuitforge_core.job_quality.scorer import score_job
from circuitforge_core.job_quality.signals import ALL_SIGNALS

__all__ = [
    "JobEnrichment",
    "JobListing",
    "JobQualityScore",
    "SignalResult",
    "score_job",
    "ALL_SIGNALS",
]
