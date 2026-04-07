# circuitforge_core/vision/router.py — shim
#
# The vision module has been extracted to the standalone cf-vision repo.
# This shim re-exports VisionRouter so existing imports continue to work.
# New code should import directly from cf_vision:
#
#   from cf_vision.router import VisionRouter
#   from cf_vision.models import ImageFrame
#
# Install: pip install -e ../cf-vision
from __future__ import annotations

try:
    from cf_vision.router import VisionRouter  # noqa: F401
    from cf_vision.models import ImageFrame    # noqa: F401
except ImportError:
    # cf-vision not installed — fall back to the stub so products that don't
    # need vision yet don't hard-fail on import.
    class VisionRouter:  # type: ignore[no-redef]
        """Stub — install cf-vision: pip install -e ../cf-vision"""

        def analyze(self, image_bytes: bytes, prompt: str = "", task: str = "document"):
            raise ImportError(
                "cf-vision is not installed. "
                "Run: pip install -e ../cf-vision"
            )
