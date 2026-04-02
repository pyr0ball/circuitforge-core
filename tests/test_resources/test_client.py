import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpretty
from circuitforge_core.resources.client import CFOrchClient, Allocation

_ALLOC_BODY = (
    '{"allocation_id":"abc123","service":"vllm","node_id":"heimdall",'
    '"gpu_id":0,"model":"Ouro-1.4B","url":"http://heimdall:8000","started":false,"warm":true}'
)


@httpretty.activate
def test_sync_allocate_returns_allocation():
    httpretty.register_uri(
        httpretty.POST, "http://orch:7700/api/services/vllm/allocate",
        body=_ALLOC_BODY, content_type="application/json",
    )
    httpretty.register_uri(
        httpretty.DELETE, "http://orch:7700/api/services/vllm/allocations/abc123",
        body='{"released":true}', content_type="application/json",
    )
    client = CFOrchClient("http://orch:7700")
    with client.allocate("vllm", model_candidates=["Ouro-1.4B"], caller="test") as alloc:
        assert isinstance(alloc, Allocation)
        assert alloc.url == "http://heimdall:8000"
        assert alloc.model == "Ouro-1.4B"
        assert alloc.allocation_id == "abc123"
    assert httpretty.last_request().method == "DELETE"


@httpretty.activate
def test_sync_allocate_ignores_404_on_release():
    httpretty.register_uri(
        httpretty.POST, "http://orch:7700/api/services/vllm/allocate",
        body='{"allocation_id":"xyz","service":"vllm","node_id":"a","gpu_id":0,'
             '"model":"m","url":"http://a:8000","started":false,"warm":false}',
        content_type="application/json",
    )
    httpretty.register_uri(
        httpretty.DELETE, "http://orch:7700/api/services/vllm/allocations/xyz",
        status=404, body='{"detail":"not found"}', content_type="application/json",
    )
    client = CFOrchClient("http://orch:7700")
    with client.allocate("vllm", model_candidates=["m"]) as alloc:
        assert alloc.url == "http://a:8000"
    # No exception raised — 404 on release is silently ignored


@httpretty.activate
def test_sync_allocate_raises_on_503():
    httpretty.register_uri(
        httpretty.POST, "http://orch:7700/api/services/vllm/allocate",
        status=503, body='{"detail":"no capacity"}', content_type="application/json",
    )
    client = CFOrchClient("http://orch:7700")
    with pytest.raises(RuntimeError, match="cf-orch allocation failed"):
        with client.allocate("vllm", model_candidates=["m"]):
            pass


async def test_async_allocate_works():
    # httpretty only patches stdlib sockets; httpx async uses anyio sockets so
    # we mock httpx.AsyncClient directly instead.
    alloc_data = {
        "allocation_id": "a1", "service": "vllm", "node_id": "n",
        "gpu_id": 0, "model": "m", "url": "http://n:8000",
        "started": False, "warm": False,
    }
    release_data = {"released": True}

    def _make_response(data, status_code=200):
        resp = MagicMock()
        resp.is_success = status_code < 400
        resp.status_code = status_code
        resp.json.return_value = data
        return resp

    mock_post = AsyncMock(return_value=_make_response(alloc_data))
    mock_delete = AsyncMock(return_value=_make_response(release_data))

    mock_async_client = MagicMock()
    mock_async_client.post = mock_post
    mock_async_client.delete = mock_delete
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        client = CFOrchClient("http://orch:7700")
        async with client.allocate_async("vllm", model_candidates=["m"]) as alloc:
            assert alloc.url == "http://n:8000"
            assert alloc.allocation_id == "a1"
    mock_delete.assert_called_once()
