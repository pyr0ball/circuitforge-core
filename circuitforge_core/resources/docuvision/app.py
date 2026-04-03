"""
cf-docuvision — managed document understanding service.

Wraps ByteDance/Dolphin-v2 (Qwen2.5-VL backbone) behind a simple HTTP API.
Managed by cf-orch; started/stopped as a ProcessSpec service.

API
---
GET  /health          → {"status": "ok", "model": "<path>"}
POST /extract         → ExtractResponse

Usage (standalone)::

    python -m circuitforge_core.resources.docuvision.app \\
        --model /Library/Assets/LLM/docuvision/models/dolphin-v2 \\
        --port 8003 --gpu-id 0
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Module-level state — populated by _load_model() on first /extract call
_model: Any = None
_processor: Any = None
_model_path: str = ""
_device: str = "cpu"


# ── lazy loader ───────────────────────────────────────────────────────────────

def _load_model() -> None:
    """Lazy-load Dolphin-v2. Called once on first /extract request."""
    global _model, _processor, _device

    if _model is not None:
        return

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    logger.info("Loading Dolphin-v2 from %s ...", _model_path)
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _processor = AutoProcessor.from_pretrained(
        _model_path,
        trust_remote_code=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        device_map=_device,
    )
    _model.eval()
    logger.info("Dolphin-v2 loaded on %s", _device)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


app = FastAPI(title="cf-docuvision", lifespan=_lifespan)


# ── request / response models ─────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    """
    Either image_b64 (base64-encoded bytes) or image_path (absolute path) must
    be provided. hint guides the extraction mode:
      - "auto"     - Dolphin-v2 detects layout and element types automatically
      - "table"    - optimise for tabular data (receipts, invoices, forms)
      - "text"     - optimise for dense prose (contracts, letters)
      - "form"     - optimise for form field extraction
    """
    image_b64: str | None = None
    image_path: str | None = None
    hint: str = "auto"


class ElementOut(BaseModel):
    type: str          # heading | paragraph | list | table | figure | formula | code
    text: str
    bbox: list[float] | None = None   # [x0, y0, x1, y1] normalised 0-1 if available


class TableOut(BaseModel):
    html: str
    bbox: list[float] | None = None


class ExtractResponse(BaseModel):
    elements: list[ElementOut]
    raw_text: str
    tables: list[TableOut]
    metadata: dict[str, Any]


# ── helpers ───────────────────────────────────────────────────────────────────

_HINT_PROMPTS: dict[str, str] = {
    "auto":  "Parse this document. Extract all elements with their types and text content.",
    "table": "Extract all tables from this document as structured HTML. Also extract any line-item text.",
    "text":  "Extract all text from this document preserving paragraph and heading structure.",
    "form":  "Extract all form fields from this document. Return field labels and their values.",
}


def _image_from_request(req: ExtractRequest):
    """Return a PIL Image from either image_b64 or image_path."""
    from PIL import Image

    if req.image_b64:
        img_bytes = base64.b64decode(req.image_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if req.image_path:
        from pathlib import Path
        p = Path(req.image_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"image_path not found: {req.image_path}")
        return Image.open(p).convert("RGB")

    raise HTTPException(status_code=422, detail="Either image_b64 or image_path must be provided")


def _parse_dolphin_output(raw: str) -> tuple[list[ElementOut], list[TableOut], str]:
    """
    Parse Dolphin-v2's structured output into elements and tables.

    Dolphin-v2 returns a JSON array of element dicts with keys:
      type, text, [html], [bbox]

    Falls back gracefully if the model returns plain text instead.
    """
    elements: list[ElementOut] = []
    tables: list[TableOut] = []

    # Try JSON parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                etype = item.get("type", "paragraph")
                text = item.get("text", "")
                bbox = item.get("bbox")
                if etype == "table":
                    tables.append(TableOut(html=item.get("html", text), bbox=bbox))
                elements.append(ElementOut(type=etype, text=text, bbox=bbox))
            raw_text = "\n".join(e.text for e in elements)
            return elements, tables, raw_text
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain-text fallback: treat entire output as a single paragraph
    elements = [ElementOut(type="paragraph", text=raw.strip())]
    return elements, tables, raw.strip()


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": _model_path}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    _load_model()

    image = _image_from_request(req)
    prompt = _HINT_PROMPTS.get(req.hint, _HINT_PROMPTS["auto"])

    import torch

    inputs = _processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    raw_output = _processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    )

    elements, tables, raw_text = _parse_dolphin_output(raw_output)

    w, h = image.size

    return ExtractResponse(
        elements=elements,
        raw_text=raw_text,
        tables=tables,
        metadata={
            "hint": req.hint,
            "width": w,
            "height": h,
            "model": _model_path,
            "device": _device,
        },
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="cf-docuvision service")
    parser.add_argument("--model", required=True, help="Path to Dolphin-v2 model directory")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    global _model_path
    _model_path = args.model

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
