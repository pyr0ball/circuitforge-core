from unittest.mock import MagicMock, patch
import pytest
from circuitforge_core.llm import LLMRouter


def _make_router(config: dict) -> LLMRouter:
    """Build a router from an in-memory config dict (bypass file loading)."""
    router = object.__new__(LLMRouter)
    router.config = config
    return router


def test_complete_uses_first_reachable_backend():
    router = _make_router({
        "fallback_order": ["local"],
        "backends": {
            "local": {
                "type": "openai_compat",
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
                "supports_images": False,
            }
        }
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))]
    )
    with patch.object(router, "_is_reachable", return_value=True), \
         patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client):
        result = router.complete("say hello")
    assert result == "hello"


def test_complete_falls_back_on_unreachable_backend():
    router = _make_router({
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
            }
        }
    })
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="fallback"))]
    )
    def reachable(url):
        return "nowhere" not in url
    with patch.object(router, "_is_reachable", side_effect=reachable), \
         patch("circuitforge_core.llm.router.OpenAI", return_value=mock_client):
        result = router.complete("test")
    assert result == "fallback"


def test_complete_raises_when_all_backends_exhausted():
    router = _make_router({
        "fallback_order": ["dead"],
        "backends": {
            "dead": {
                "type": "openai_compat",
                "base_url": "http://nowhere:1/v1",
                "model": "x",
                "supports_images": False,
            }
        }
    })
    with patch.object(router, "_is_reachable", return_value=False):
        with pytest.raises(RuntimeError, match="exhausted"):
            router.complete("test")
