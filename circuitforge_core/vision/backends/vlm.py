# circuitforge_core/vision/backends/vlm.py — VLMBackend
#
# Requires: pip install -e "circuitforge-core[vision-vlm]"
#
# Supports any HuggingFace AutoModelForVision2Seq-compatible VLM.
# Validated models (VRAM fp16):
#   vikhyatk/moondream2           ~2 GB   — fast, lightweight, good for documents
#   llava-hf/llava-1.5-7b-hf     ~14 GB  — strong general VQA
#   Qwen/Qwen2-VL-7B-Instruct    ~16 GB  — multilingual, structured output friendly
#
# VLMBackend implements caption() (generative) and a prompt-based classify()
# that asks the model to pick from a list.  embed() raises NotImplementedError.
from __future__ import annotations

import io

from circuitforge_core.vision.backends.base import VisionResult

# VRAM estimates (MB, fp16) keyed by lowercase model name fragment.
_VRAM_TABLE: dict[str, int] = {
    "moondream2": 2000,
    "moondream": 2000,
    "llava-1.5-7b": 14000,
    "llava-7b": 14000,
    "qwen2-vl-7b": 16000,
    "qwen-vl-7b": 16000,
    "llava-1.5-13b": 26000,
    "phi-3-vision": 8000,
    "phi3-vision": 8000,
    "paligemma": 6000,
    "idefics": 12000,
    "cogvlm": 14000,
}

_CLASSIFY_PROMPT_TMPL = (
    "Choose the single best label for this image from the following options: "
    "{labels}. Reply with ONLY the label text, nothing else."
)


def _estimate_vram(model_path: str) -> int:
    lower = model_path.lower()
    for key, mb in _VRAM_TABLE.items():
        if key in lower:
            return mb
    return 8000  # safe default for unknown 7B-class VLMs


class VLMBackend:
    """
    Generative vision-language model backend.

    caption() generates free-form text from an image + optional prompt.
    classify() prompts the model to select from candidate labels.
    embed() raises NotImplementedError — use SigLIPBackend for embeddings.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as exc:
            raise ImportError(
                "VLMBackend requires torch and transformers. "
                "Install with: pip install -e 'circuitforge-core[vision-vlm]'"
            ) from exc

        import torch as _torch

        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model_path = model_path
        self._vram_mb = _estimate_vram(model_path)

        torch_dtype = (
            _torch.float16 if dtype == "float16"
            else _torch.bfloat16 if dtype == "bfloat16"
            else _torch.float32
        )

        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        # Put model in inference mode — disables dropout/batchnorm training behaviour
        self._model.train(False)

    # ── VisionBackend Protocol ─────────────────────────────────────────────────

    def caption(self, image: bytes, prompt: str = "") -> VisionResult:
        """Generate a text description of the image."""
        import torch
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        effective_prompt = prompt or "Describe this image in detail."

        inputs = self._processor(
            text=effective_prompt,
            images=pil_img,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Strip the input prompt tokens from the generated output
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        text = self._processor.decode(output_ids, skip_special_tokens=True).strip()

        return VisionResult(caption=text, model=self.model_name)

    def classify(self, image: bytes, labels: list[str]) -> VisionResult:
        """
        Prompt-based zero-shot classification.

        Asks the VLM to choose a label from the provided list.  The returned
        scores are binary (1.0 for the selected label, 0.0 for others) since
        VLMs don't expose per-label logits the same way SigLIP does.
        For soft scores, use SigLIPBackend.
        """
        labels_str = ", ".join(f'"{lbl}"' for lbl in labels)
        prompt = _CLASSIFY_PROMPT_TMPL.format(labels=labels_str)
        result = self.caption(image, prompt=prompt)
        raw = (result.caption or "").strip().strip('"').strip("'")

        matched = _match_label(raw, labels)
        scores = [1.0 if lbl == matched else 0.0 for lbl in labels]
        return VisionResult(labels=list(labels), scores=scores, model=self.model_name)

    def embed(self, image: bytes) -> VisionResult:
        raise NotImplementedError(
            "VLMBackend does not support image embeddings. "
            "Use backend='siglip' (SigLIPBackend) for embed()."
        )

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1]

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    @property
    def supports_embed(self) -> bool:
        return False

    @property
    def supports_caption(self) -> bool:
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_label(raw: str, labels: list[str]) -> str:
    """Return the best matching label from the VLM's free-form response."""
    raw_lower = raw.lower()
    for lbl in labels:
        if lbl.lower() == raw_lower:
            return lbl
    for lbl in labels:
        if raw_lower.startswith(lbl.lower()) or lbl.lower().startswith(raw_lower):
            return lbl
    for lbl in labels:
        if lbl.lower() in raw_lower or raw_lower in lbl.lower():
            return lbl
    return labels[0] if labels else raw
