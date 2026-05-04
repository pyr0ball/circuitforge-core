from unittest.mock import MagicMock, patch
import pytest
from circuitforge_core.llm import LLMRouter


def _make_router(config: dict) -> LLMRouter:
    """Build a router from an in-memory config dict (bypass file loading)."""
    router = object.__new__(LLMRouter)
    router.config = config
    return router


def test_complete_uses_first_reachable_backend():
    router = _make_router(
        {
            "fallback_order": ["local"],
            "backends": {
                "local": {
                    "type": "openai_compat",
                    "base_url": "http://localhost:11434/v1",
                    "model": "llama3",
                    "supports_images": False,
                }
            },
        }
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))]
    )
    with (
        patch.object(router, "_is_reachable", return_value=True),
        patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client),
    ):
        result = router.complete("say hello")
    assert result == "hello"


def test_complete_falls_back_on_unreachable_backend():
    router = _make_router(
        {
            "fallback_order": ["unreachable", "working"],
            "backends": {
                "unreachable": {
                    "type": "openai_compat",
                    "base_url": "http://nowhere:1/v1",
                    "model": "x",
                    "supports_images": False,
                },
                "working": {
                    "type": "openai_compat",
                    "base_url": "http://localhost:11434/v1",
                    "model": "llama3",
                    "supports_images": False,
                },
            },
        }
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="fallback"))]
    )

    def reachable(url):
        return "nowhere" not in url

    with (
        patch.object(router, "_is_reachable", side_effect=reachable),
        patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client),
    ):
        result = router.complete("test")
    assert result == "fallback"


def test_complete_raises_when_all_backends_exhausted():
    router = _make_router(
        {
            "fallback_order": ["dead"],
            "backends": {
                "dead": {
                    "type": "openai_compat",
                    "base_url": "http://nowhere:1/v1",
                    "model": "x",
                    "supports_images": False,
                }
            },
        }
    )
    with patch.object(router, "_is_reachable", return_value=False):
        with pytest.raises(RuntimeError, match="exhausted"):
            router.complete("test")


def test_try_cf_orch_alloc_import_path():
    """Verify lazy import points to circuitforge_orch, not circuitforge_core.resources."""
    import inspect
    from circuitforge_core.llm import router as router_module

    src = inspect.getsource(router_module.LLMRouter._try_cf_orch_alloc)
    assert "circuitforge_orch.client" in src
    assert "circuitforge_core.resources.client" not in src


def test_embed_returns_vectors_from_openai_compat_backend():
    router = _make_router(
        {
            "fallback_order": ["ollama"],
            "backends": {
                "ollama": {
                    "type": "openai_compat",
                    "base_url": "http://localhost:11434/v1",
                    "model": "mistral:7b",
                    "embedding_model": "nomic-embed-text",
                    "supports_images": False,
                }
            },
        }
    )
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
    )
    with (
        patch.object(router, "_is_reachable", return_value=True),
        patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client),
    ):
        result = router.embed(["hello world", "fireball rules"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client.embeddings.create.assert_called_once_with(
        model="nomic-embed-text",
        input=["hello world", "fireball rules"],
    )


def test_embed_uses_chat_model_when_no_embedding_model_configured():
    router = _make_router(
        {
            "fallback_order": ["ollama"],
            "backends": {
                "ollama": {
                    "type": "openai_compat",
                    "base_url": "http://localhost:11434/v1",
                    "model": "llama3",
                    "supports_images": False,
                }
            },
        }
    )
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.9, 0.8])]
    )
    with (
        patch.object(router, "_is_reachable", return_value=True),
        patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client),
    ):
        router.embed(["test"])

    call_kwargs = mock_client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "llama3"


def test_embed_skips_non_openai_compat_backends():
    router = _make_router(
        {
            "fallback_order": ["anthropic", "ollama"],
            "backends": {
                "anthropic": {
                    "type": "anthropic",
                    "enabled": True,
                    "model": "claude-haiku-4-5-20251001",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "supports_images": True,
                },
                "ollama": {
                    "type": "openai_compat",
                    "base_url": "http://localhost:11434/v1",
                    "model": "nomic-embed-text",
                    "supports_images": False,
                },
            },
        }
    )
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1])]
    )
    mock_openai = MagicMock(return_value=mock_client)
    with (
        patch.object(router, "_is_reachable", return_value=True),
        patch("circuitforge_core.llm.router.OpenAI", mock_openai),
    ):
        result = router.embed(["hello"])

    assert result == [[0.1]]
    # Only ollama reached the OpenAI constructor; anthropic was skipped by type check
    mock_openai.assert_called_once()


def test_embed_raises_when_all_backends_exhausted():
    router = _make_router(
        {
            "fallback_order": ["dead"],
            "backends": {
                "dead": {
                    "type": "openai_compat",
                    "base_url": "http://nowhere:1/v1",
                    "model": "x",
                    "supports_images": False,
                }
            },
        }
    )
    with patch.object(router, "_is_reachable", return_value=False):
        with pytest.raises(RuntimeError, match="exhausted"):
            router.embed(["test"])
