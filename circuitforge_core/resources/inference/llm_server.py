"""Generic OpenAI-compatible inference server for HuggingFace causal LMs."""
from __future__ import annotations

import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_model: Any = None
_tokenizer: Any = None
_model_id: str = ""
_device: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int | None = 512
    temperature: float | None = 0.7
    stream: bool | None = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": _model_id}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": _model_id, "object": "model", "owned_by": "cf-orch"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(503, detail="Model not loaded")
    if req.stream:
        raise HTTPException(501, detail="Streaming not supported")

    conversation = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        input_ids = _tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(_device)
    except Exception as exc:
        raise HTTPException(500, detail=f"Tokenisation failed: {exc}")

    max_new = req.max_tokens or 512
    temp = req.temperature if req.temperature is not None else 0.7
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "do_sample": temp > 0,
        "pad_token_id": _tokenizer.eos_token_id,
    }
    if temp > 0:
        gen_kwargs["temperature"] = temp

    with torch.inference_mode():
        output_ids = _model.generate(input_ids, **gen_kwargs)

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    reply = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[-1],
            "completion_tokens": len(new_tokens),
            "total_tokens": input_ids.shape[-1] + len(new_tokens),
        },
    }


def _load_model(model_path: str, gpu_id: int) -> None:
    global _model, _tokenizer, _model_id, _device
    _device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    _model_id = model_path
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in _device else torch.float32,
        device_map={"": _device},
        trust_remote_code=True,
    )
    _model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="cf-orch generic LLM inference server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    _load_model(args.model, args.gpu_id)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
