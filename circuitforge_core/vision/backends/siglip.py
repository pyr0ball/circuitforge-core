# circuitforge_core/vision/backends/siglip.py — SigLIPBackend
#
# Requires: pip install -e "circuitforge-core[vision-siglip]"
# Default model: google/siglip-so400m-patch14-384 (~1.4 GB VRAM)
#
# SigLIP uses sigmoid cross-entropy rather than softmax over labels, so each
# score is an independent 0–1 probability.  This is better than CLIP for
# multi-label classification and document routing.
from __future__ import annotations

import io

from circuitforge_core.vision.backends.base import VisionResult

_DEFAULT_MODEL = "google/siglip-so400m-patch14-384"

# VRAM footprints by model variant (MB, fp16).
_VRAM_TABLE: dict[str, int] = {
    "siglip-so400m-patch14-384": 1440,
    "siglip-so400m-patch14-224": 1440,
    "siglip-base-patch16-224": 340,
    "siglip-large-patch16-256": 690,
}


def _estimate_vram(model_path: str) -> int:
    lower = model_path.lower()
    for key, mb in _VRAM_TABLE.items():
        if key in lower:
            return mb
    return 1500  # conservative default for unknown so400m variants


class SigLIPBackend:
    """
    Image classification + embedding via Google SigLIP.

    classify() returns sigmoid similarity scores for each candidate label —
    independent probabilities, not a softmax distribution.
    embed()    returns the CLS-pool image embedding (normalised).
    caption()  raises NotImplementedError — use VLMBackend for generation.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "SigLIPBackend requires torch and transformers. "
                "Install with: pip install -e 'circuitforge-core[vision-siglip]'"
            ) from exc

        import torch as _torch

        self._device = device
        self._dtype_str = dtype
        self._torch_dtype = (
            _torch.float16 if dtype == "float16"
            else _torch.bfloat16 if dtype == "bfloat16"
            else _torch.float32
        )
        self._model_path = model_path
        self._vram_mb = _estimate_vram(model_path)

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self._torch_dtype,
        ).to(device)
        # Set inference mode (train(False) == model.eval() without grad tracking)
        self._model.train(False)

    # ── VisionBackend Protocol ─────────────────────────────────────────────────

    def classify(self, image: bytes, labels: list[str]) -> VisionResult:
        """Zero-shot sigmoid classification — scores are independent per label."""
        import torch
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self._processor(
            text=labels,
            images=pil_img,
            return_tensors="pt",
            padding="max_length",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # logits_per_image: (1, num_labels) — raw SigLIP logits
            logits = outputs.logits_per_image[0]
            scores = torch.sigmoid(logits).cpu().float().tolist()

        return VisionResult(labels=list(labels), scores=scores, model=self.model_name)

    def embed(self, image: bytes) -> VisionResult:
        """Return normalised image embedding (CLS pool, L2-normalised)."""
        import torch
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            # L2-normalise so dot-product == cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding = image_features[0].cpu().float().tolist()
        return VisionResult(embedding=embedding, model=self.model_name)

    def caption(self, image: bytes, prompt: str = "") -> VisionResult:
        raise NotImplementedError(
            "SigLIPBackend does not support caption generation. "
            "Use backend='vlm' (VLMBackend) for image-to-text generation."
        )

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1]

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    @property
    def supports_embed(self) -> bool:
        return True

    @property
    def supports_caption(self) -> bool:
        return False
