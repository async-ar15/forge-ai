"""``POST /v1/infer`` SSE contract and gRPC failure modes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest
from forgeai.gateway.tenant_context import TenantContext


def _parse_sse(body: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


@pytest.fixture
def authed_client(gateway_client):
    """Client with auth patched open."""

    client, state = gateway_client
    tid = __import__("uuid").uuid4()
    ctx = TenantContext(tenant_id=tid, tenant_tier="free", auth_degraded=False)

    async def _auth(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return ctx, True

    with patch(
        "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
        new=_auth,
    ):
        yield client, state


def test_full_sse_stream(authed_client) -> None:
    """Chunks carry ``request_id``; final chunk includes cost."""

    client, state = authed_client
    resp = client.post(
        "/v1/infer",
        json={"query": "hello", "stream": True},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert len(events) >= 2
    rid = resp.headers.get("X-Request-Id")
    assert rid
    for ev in events:
        if "error" in ev:
            continue
        assert ev.get("request_id") == rid
    final = events[-1]
    assert final.get("is_final") is True
    assert final.get("cost_microdollars") == 99
    state.decision_logger.log_decision.assert_awaited_once()
    state.reward_pipeline.compute_and_record.assert_awaited_once()
    state.response_store.store.assert_awaited_once()


def test_query_type_unknown_default_action(authed_client) -> None:
    """``query_type`` is forwarded without inference (unknown is valid)."""

    client, state = authed_client
    client.post(
        "/v1/infer",
        json={"query": "hello", "query_type": "unknown"},
        headers={"X-API-Key": "k"},
    )
    call = state.policy_stub.Decide.await_args
    assert call is not None
    req = call[0][0]
    qt = int(req.routing_state_features.query_type)
    from forgeai.proto import common_pb2

    assert qt == int(common_pb2.QUERY_TYPE_UNKNOWN)


def test_policy_service_503(authed_client) -> None:
    """Policy gRPC failure becomes HTTP 503 SSE error."""

    import grpc as _grpc

    client, state = authed_client
    state.policy_stub.Decide = AsyncMock(
        side_effect=grpc.aio.AioRpcError(
            _grpc.StatusCode.UNAVAILABLE,
            None,
            None,
            "nu",
            None,
        ),
    )
    resp = client.post(
        "/v1/infer",
        json={"query": "hello"},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 503
    ev = _parse_sse(resp.text)[0]
    assert ev.get("error") == "policy_unavailable"


def test_execution_service_503(authed_client) -> None:
    """Execution stream RPC failure yields error SSE."""

    client, state = authed_client

    def _boom(*_a: object, **_k: object) -> None:
        raise grpc.aio.AioRpcError(
            grpc.StatusCode.UNAVAILABLE,
            None,
            None,
            "nu",
            None,
        )

    state.execution_stub.Execute = MagicMock(side_effect=_boom)
    resp = client.post(
        "/v1/infer",
        json={"query": "hello"},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    ev = _parse_sse(resp.text)[0]
    assert ev.get("error") == "execution_unavailable"


def test_query_exceeds_max_returns_422(authed_client) -> None:
    """Oversized ``query`` fails validation."""

    client, _state = authed_client
    resp = client.post(
        "/v1/infer",
        json={"query": "x" * 32769},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 422


def test_decision_logger_once_and_r_online(authed_client) -> None:
    """Exactly one persistence call; reward scalar is finite."""

    client, state = authed_client
    client.post(
        "/v1/infer",
        json={"query": "hello"},
        headers={"X-API-Key": "k"},
    )
    state.decision_logger.log_decision.assert_awaited_once()
    state.reward_pipeline.compute_and_record.assert_awaited_once()
    arg = state.decision_logger.log_decision.await_args[0][0]
    assert hasattr(arg, "r_online")
    assert float(arg.r_online) == float(arg.r_online)


def test_stream_false_single_buffer(authed_client) -> None:
    """Non-streaming still uses SSE with a buffered token event."""

    client, _state = authed_client
    resp = client.post(
        "/v1/infer",
        json={"query": "hello", "stream": False},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    texts = [e.get("token") for e in _parse_sse(resp.text) if "token" in e]
    assert "hi" in texts
