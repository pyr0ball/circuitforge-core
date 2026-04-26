# circuitforge_core/reranker/adapters/qwen3.py — Qwen3-Reranker adapter
#
# Requires: pip install circuitforge-core[reranker-qwen3]
# Tested with: Qwen/Qwen3-Reranker-0.6B, -1.5B, -8B
#
# Scoring mechanism (generative reranker):
#   Rather than generating a full response, we pre-fill the assistant turn with
#   the <think>\n\n</think>\n block and read the logits at the last input token
#   position. The softmax probability of "yes" vs "no" at that position is the
#   relevance score — one forward pass per batch, no generation loop.
#
# Prompt format (Qwen3 chat template):
#   system: "Judge whether the Document meets the requirements based on the
#            Query and the Instruct. Note that the answer can only be 'yes'
#            or 'no'."
#   user:   "<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}"
#   assistant (pre-filled): "<think>\n\n</think>\n\n"
#
# MIT licensed.
from __future__ import annotations

import logging
import threading
from typing import Sequence

from circuitforge_core.reranker.base import TextReranker

logger = logging.getLogger(__name__)

# Optional heavy deps — lazy-imported at load() time.
try:
    import torch as _torch  # type: ignore[import]
except ImportError:
    _torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore[import]
    from transformers import AutoTokenizer as _AutoTokenizer  # type: ignore[import]
except ImportError:
    _AutoModel = None  # type: ignore[assignment]
    _AutoTokenizer = None  # type: ignore[assignment]

# System prompt used for all reranking tasks.
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct. Note that the answer can only be "yes" or "no".'
)

# Default task instruction — products can override via make_reranker(task=...).
_DEFAULT_TASK = "Given a query, retrieve the most relevant document that answers the query."

# The pre-filled assistant turn that puts the model past its thinking block
# so the very next token position scores "yes" vs "no".
_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"


def _requires_deps() -> None:
    if _torch is None:
        raise ImportError(
            "torch is not installed. Run: pip install circuitforge-core[reranker-qwen3]"
        )
    if _AutoModel is None:
        raise ImportError(
            "transformers is not installed. Run: pip install circuitforge-core[reranker-qwen3]"
        )


class Qwen3TextReranker(TextReranker):
    """
    Generative reranker using the Qwen3-Reranker model family.

    Scores candidates by reading yes/no token logits at the last input position
    after pre-filling the assistant thinking block. One forward pass covers an
    entire batch — efficient for ranking large candidate lists.

    Model options (by tier):
        Free:  Qwen/Qwen3-Reranker-0.6B  (~1.2 GB VRAM fp16)
               Qwen/Qwen3-Reranker-1.5B  (~3.0 GB VRAM fp16)
        Paid:  Qwen/Qwen3-Reranker-8B    (~16 GB VRAM fp16, or ~9 GB int8)

    Usage:
        reranker = Qwen3TextReranker("Qwen/Qwen3-Reranker-0.6B")
        results = reranker.rerank("chicken soup recipe", ["recipe 1...", "recipe 2..."])

    With a custom task instruction:
        reranker = Qwen3TextReranker(
            "Qwen/Qwen3-Reranker-1.5B",
            task="Given a job description, retrieve the most relevant resume.",
        )
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        task: str = _DEFAULT_TASK,
        device: str | None = None,
        dtype: str = "float16",
        batch_size: int = 32,
    ) -> None:
        self._model_id = model_id
        self._task = task
        self._device = device  # None = auto-detect at load time
        self._dtype_str = dtype
        self._batch_size = batch_size
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    def load(self) -> None:
        """Explicitly load model weights. Called automatically on first rerank()."""
        _requires_deps()
        with self._lock:
            if self._model is not None:
                return
            device = self._device or ("cuda" if _torch.cuda.is_available() else "cpu")
            dtype_map: dict[str, object] = {
                "float16": _torch.float16,
                "bfloat16": _torch.bfloat16,
                "float32": _torch.float32,
            }
            torch_dtype = dtype_map.get(self._dtype_str, _torch.float16)

            logger.info(
                "Loading Qwen3 reranker: %s (device=%s dtype=%s)",
                self._model_id, device, self._dtype_str,
            )
            tokenizer = _AutoTokenizer.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            model = _AutoModel.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
            model.eval()

            # Resolve the token IDs for "yes" and "no" once at load time.
            # Qwen tokenizers encode single-word tokens without a leading space.
            yes_ids: list[int] = tokenizer.encode("yes", add_special_tokens=False)
            no_ids: list[int] = tokenizer.encode("no", add_special_tokens=False)
            if not yes_ids or not no_ids:
                raise RuntimeError(
                    f"Could not resolve 'yes'/'no' token IDs from tokenizer {self._model_id!r}. "
                    "This model may not be a Qwen3-Reranker variant."
                )

            self._tokenizer = tokenizer
            self._model = model
            self._yes_id = yes_ids[0]
            self._no_id = no_ids[0]

    def unload(self) -> None:
        """Release model weights. Useful for VRAM management between tasks."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._yes_id = None
            self._no_id = None

    def _build_prompt(self, query: str, document: str) -> str:
        """Format a single (query, document) pair as a chat-template prompt."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<Instruct>: {self._task}\n"
                    f"<Query>: {query}\n"
                    f"<Document>: {document}"
                ),
            },
        ]
        # apply_chat_template without tokenization so we can append the prefill.
        text: str = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + _ASSISTANT_PREFILL

    def _score_pairs(self, query: str, candidates: list[str]) -> list[float]:
        if self._model is None:
            self.load()
        return self._score_in_batches(query, candidates)

    def _score_in_batches(self, query: str, candidates: list[str]) -> list[float]:
        """Score all (query, candidate) pairs, splitting into sub-batches."""
        all_scores: list[float] = []
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            all_scores.extend(self._score_batch(query, batch))
        return all_scores

    def _score_batch(self, query: str, candidates: list[str]) -> list[float]:
        """Single forward pass for one sub-batch. Returns a score per candidate."""
        prompts = [self._build_prompt(query, c) for c in candidates]

        # Left-pad so the last token position is consistent across all sequences.
        tokenizer = self._tokenizer  # type: ignore[union-attr]
        original_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
        finally:
            tokenizer.padding_side = original_side

        model = self._model  # type: ignore[union-attr]
        device = next(model.parameters()).device  # type: ignore[union-attr]
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with self._lock:
            with _torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # logits shape: (batch, seq_len, vocab_size)
        # Last position [-1] is the token the model would output next.
        last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        yes_logits = last_logits[:, self._yes_id]   # (batch,)
        no_logits = last_logits[:, self._no_id]     # (batch,)

        # Softmax over yes/no only — score = P(yes | query, doc).
        stacked = _torch.stack([yes_logits, no_logits], dim=-1)  # (batch, 2)
        probs = _torch.softmax(stacked, dim=-1)
        scores: list[float] = probs[:, 0].float().cpu().tolist()
        return scores
