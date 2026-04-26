"""Tests for Qwen3TextReranker with mocked transformers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from circuitforge_core.reranker.adapters.qwen3 import Qwen3TextReranker, _ASSISTANT_PREFILL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_model(yes_logit: float = 5.0, no_logit: float = 1.0, batch_size: int = 1):
    """Return a mock AutoModelForCausalLM that outputs fixed yes/no logits."""
    import torch

    model = MagicMock()
    # Simulate logits: (batch, seq_len=1, vocab_size=32000)
    # yes token id = 9693, no token id = 2201 (Qwen tokenizer typical values)
    vocab_size = 32000
    logits = torch.zeros(batch_size, 1, vocab_size)
    logits[:, :, 9693] = yes_logit   # "yes" token
    logits[:, :, 2201] = no_logit    # "no" token
    output = MagicMock()
    output.logits = logits
    model.return_value = output

    # next(model.parameters()).device
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    return model


def _make_mock_tokenizer(yes_id: int = 9693, no_id: int = 2201):
    """Return a mock AutoTokenizer."""
    import torch

    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, **kw: (
        [yes_id] if text == "yes" else [no_id]
    )
    tokenizer.apply_chat_template.return_value = "<prompt>"
    tokenizer.padding_side = "right"

    # Return simple fixed-length tensors from __call__
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tokenizer


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_load_raises_without_torch():
    reranker = Qwen3TextReranker("Qwen/Qwen3-Reranker-0.6B")
    with patch("circuitforge_core.reranker.adapters.qwen3._torch", None):
        with pytest.raises(ImportError, match="torch"):
            reranker.load()


def test_load_raises_without_transformers():
    reranker = Qwen3TextReranker("Qwen/Qwen3-Reranker-0.6B")
    with patch("circuitforge_core.reranker.adapters.qwen3._AutoModel", None):
        with pytest.raises(ImportError, match="transformers"):
            reranker.load()


def test_model_id():
    assert Qwen3TextReranker("Qwen/Qwen3-Reranker-1.5B").model_id == "Qwen/Qwen3-Reranker-1.5B"


def test_unload_clears_state():
    reranker = Qwen3TextReranker()
    reranker._model = MagicMock()
    reranker._tokenizer = MagicMock()
    reranker._yes_id = 1
    reranker._no_id = 2
    reranker.unload()
    assert reranker._model is None
    assert reranker._tokenizer is None
    assert reranker._yes_id is None
    assert reranker._no_id is None


def test_build_prompt_includes_prefill():
    reranker = Qwen3TextReranker()
    reranker._tokenizer = _make_mock_tokenizer()
    prompt = reranker._build_prompt("what is chicken soup", "a hearty recipe")
    assert _ASSISTANT_PREFILL in prompt


def test_score_batch_returns_yes_probability():
    """Higher yes_logit → score closer to 1.0."""
    import torch

    reranker = Qwen3TextReranker()
    reranker._tokenizer = _make_mock_tokenizer()
    reranker._model = _make_mock_model(yes_logit=10.0, no_logit=0.0)
    reranker._yes_id = 9693
    reranker._no_id = 2201

    scores = reranker._score_batch("query", ["candidate"])
    assert len(scores) == 1
    assert scores[0] > 0.99  # softmax(10, 0)[0] ≈ 0.9999


def test_score_batch_low_yes_logit():
    """Lower yes_logit → score closer to 0.0."""
    reranker = Qwen3TextReranker()
    reranker._tokenizer = _make_mock_tokenizer()
    reranker._model = _make_mock_model(yes_logit=0.0, no_logit=10.0)
    reranker._yes_id = 9693
    reranker._no_id = 2201

    scores = reranker._score_batch("query", ["irrelevant candidate"])
    assert scores[0] < 0.01


def test_rerank_sorts_by_score():
    """Integration through rerank() — highest yes-logit candidate should rank first."""
    import torch

    reranker = Qwen3TextReranker(batch_size=10)
    tokenizer = _make_mock_tokenizer()
    # Return different-length tensors per call to simulate multi-candidate batch
    call_count = [0]

    def tokenize_side_effect(prompts, **kw):
        n = len(prompts)
        return {
            "input_ids": torch.zeros(n, 10, dtype=torch.long),
            "attention_mask": torch.ones(n, 10, dtype=torch.long),
        }

    tokenizer.side_effect = tokenize_side_effect
    tokenizer.return_value = None  # disable default return
    reranker._tokenizer = tokenizer

    # Simulate two candidates: first gets low yes logit, second gets high
    import torch as _torch
    vocab_size = 32000
    batch_logits = _torch.zeros(2, 1, vocab_size)
    batch_logits[0, 0, 9693] = 1.0   # candidate 0: low relevance
    batch_logits[0, 0, 2201] = 5.0
    batch_logits[1, 0, 9693] = 5.0   # candidate 1: high relevance
    batch_logits[1, 0, 2201] = 1.0

    output = MagicMock()
    output.logits = batch_logits
    model = MagicMock()
    model.return_value = output
    param = MagicMock()
    param.device = _torch.device("cpu")
    model.parameters.return_value = iter([param])

    reranker._model = model
    reranker._yes_id = 9693
    reranker._no_id = 2201

    results = reranker.rerank("query", ["low relevance doc", "high relevance doc"])
    assert results[0].candidate == "high relevance doc"
    assert results[0].rank == 0


def test_score_in_batches_splits_correctly():
    """Verify that large candidate lists are split into sub-batches."""
    reranker = Qwen3TextReranker(batch_size=2)
    reranker._tokenizer = _make_mock_tokenizer()
    reranker._yes_id = 9693
    reranker._no_id = 2201

    batch_results: list[list[float]] = []

    def fake_score_batch(query, cands):
        batch_results.append(cands)
        return [0.5] * len(cands)

    reranker._score_batch = fake_score_batch  # type: ignore[method-assign]
    scores = reranker._score_in_batches("q", ["a", "b", "c", "d", "e"])
    assert len(scores) == 5
    # 5 candidates with batch_size=2 → 3 sub-batches: [a,b], [c,d], [e]
    assert len(batch_results) == 3
    assert batch_results[2] == ["e"]
