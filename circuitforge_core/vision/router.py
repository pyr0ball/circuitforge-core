"""
Vision model router — stub until v0.2.
Supports: moondream2 (local) and Claude vision API (cloud).
"""
from __future__ import annotations


class VisionRouter:
    """Routes image analysis requests to local or cloud vision models."""

    def analyze(self, image_bytes: bytes, prompt: str) -> str:
        """
        Analyze image_bytes with the given prompt.
        Raises NotImplementedError until vision backends are wired up.
        """
        raise NotImplementedError(
            "VisionRouter is not yet implemented. "
            "Photo analysis requires a Paid tier or local vision model (v0.2+)."
        )
