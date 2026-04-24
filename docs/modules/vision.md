# vision

Vision router base class. **Stub — partially implemented.**

```python
from circuitforge_core.vision import VisionRouter  # base class
```

## Planned design

The vision module mirrors the [LLM router](llm.md) pattern for multimodal inputs. Products subclass `VisionRouter` and configure a fallback chain over vision-capable backends.

**Planned backends:**
- `moondream2` — local, 1.8GB, fast; via the vision service FastAPI sidecar on :8002
- `claude_code` — local wrapper with vision capability
- `anthropic` — cloud, Claude's vision models
- `openai` — cloud, GPT-4o vision

## Current usage

The vision service (`scripts/vision_service/main.py` in Peregrine, and the cf-docuvision path in Kiwi) currently implements vision routing directly without going through this module. This module is being designed to absorb those implementations once the interface stabilizes.

## `VisionRouter` base class

```python
class VisionRouter:
    def analyze(
        self,
        images: list[str],        # base64-encoded
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Run vision inference. Returns text response."""
        raise NotImplementedError
```

## moondream2 specifics

moondream2 is the preferred local vision model — it's small enough for CPU use (1.8GB download) and fast enough for interactive use on GPU. Products using it:

- **Peregrine**: survey screenshot analysis (culture-fit survey assistant)
- **Kiwi**: receipt OCR fast-path, barcode label reading

!!! note "VRAM requirement"
    moondream2 uses ~1.5GB VRAM in 4-bit quantization. Stop the main LLM service before starting the vision service if you're on a card with < 6GB VRAM.
