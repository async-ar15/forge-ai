"""OpenAI-compatible gateway contract tests."""

from __future__ import annotations

import json
import uuid
from typing import cast
from unittest.mock import patch

import httpx
import pytest
from forgeai.gateway.app import create_test_app
from forgeai.gateway.openai_translate import (
    infer_query_type_heuristic,
    messages_to_chatml_prompt,
)
from forgeai.gateway.schemas import OpenAIChatMessage
from forgeai.gateway.tenant_context import TenantContext
from forgeai.proto import common_pb2
from openai import OpenAI
from starlette.testclient import TestClient


@pytest.fixture
def authed_openai_client(mock_gateway_state):
    app = create_test_app(mock_gateway_state.settings, mock_gateway_state)
    tid = uuid.uuid4()
    ctx = TenantContext(tenant_id=tid, tenant_tier="free", auth_degraded=False)

    async def _auth(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return ctx, True

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        TestClient(app) as client,
    ):
        yield client, mock_gateway_state


def _parse_sse_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("data: ")]


def test_chat_completions_non_stream_shape(authed_openai_client) -> None:
    client, _state = authed_openai_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "forgeai-routed"
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"].keys()) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }


def test_chat_completions_streaming_sse_contract(authed_openai_client) -> None:
    client, _state = authed_openai_client
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    lines = _parse_sse_lines(resp.text)
    assert len(lines) >= 2
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"].get("content") is not None
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert lines[-1] == "data: [DONE]"


def test_models_endpoint_shape(authed_openai_client) -> None:
    client, _state = authed_openai_client
    resp = client.get("/v1/models", headers={"X-API-Key": "k"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [row["id"] for row in body["data"]]
    assert ids == ["forgeai-auto", "forgeai-small", "forgeai-medium", "forgeai-large"]


def test_chat_response_headers_present(authed_openai_client) -> None:
    client, _state = authed_openai_client
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "anything", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    for key in (
        "x-forgeai-model",
        "x-forgeai-precision",
        "x-forgeai-retrieval",
        "x-forgeai-cost",
        "x-forgeai-latency",
        "x-forgeai-policy-version",
        "x-forgeai-request-id",
    ):
        assert key in resp.headers


def test_headers_present_on_stream_and_infer(authed_openai_client) -> None:
    client, _state = authed_openai_client
    stream_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"X-API-Key": "k"},
    )
    infer_resp = client.post(
        "/v1/infer",
        json={"query": "hello"},
        headers={"X-API-Key": "k"},
    )
    for res in (stream_resp, infer_resp):
        assert "x-forgeai-model" in res.headers
        assert "x-forgeai-precision" in res.headers
        assert "x-forgeai-retrieval" in res.headers
        assert "x-forgeai-cost" in res.headers
        assert "x-forgeai-latency" in res.headers
        assert "x-forgeai-policy-version" in res.headers
        assert "x-forgeai-request-id" in res.headers


def test_unknown_and_gpt4o_model_names_accepted(authed_openai_client) -> None:
    client, _state = authed_openai_client
    unknown = client.post(
        "/v1/chat/completions",
        json={
            "model": "some-new-model",
            "messages": [{"role": "user", "content": "x"}],
        },
        headers={"X-API-Key": "k"},
    )
    known = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        headers={"X-API-Key": "k"},
    )
    assert unknown.status_code == 200
    assert known.status_code == 200


def test_query_type_heuristics_from_messages(authed_openai_client) -> None:
    client, state = authed_openai_client
    client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "```python\nprint(1)\n```"}],
        },
        headers={"X-API-Key": "k"},
    )
    req_code = state.policy_stub.Decide.await_args_list[-1][0][0]
    assert int(req_code.routing_state_features.query_type) == int(
        common_pb2.QUERY_TYPE_CODE
    )
    client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
        },
        headers={"X-API-Key": "k"},
    )
    req_chat = state.policy_stub.Decide.await_args_list[-1][0][0]
    assert int(req_chat.routing_state_features.query_type) == int(
        common_pb2.QUERY_TYPE_CHAT
    )


def test_pure_chatml_and_heuristic_helpers() -> None:
    msgs = [
        OpenAIChatMessage(role="system", content="You write code."),
        OpenAIChatMessage(role="user", content="```python\nprint(1)\n```"),
    ]
    qtype = infer_query_type_heuristic(msgs)
    prompt = messages_to_chatml_prompt(msgs)
    assert qtype == "code"
    assert "<|im_start|>system\nYou write code.<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_openai_sdk_drop_in_call(authed_openai_client) -> None:
    client, _state = authed_openai_client

    def _handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        query = f"?{req.url.query.decode()}" if req.url.query else ""
        resp = client.request(
            req.method,
            f"{path}{query}",
            headers=dict(req.headers),
            content=req.content,
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
            request=req,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(_handler))
    sdk = OpenAI(base_url="http://testserver/v1", api_key="k", http_client=http_client)
    out = sdk.chat.completions.create(
        model="gpt-4o",
        messages=cast(list[dict[str, str]], [{"role": "user", "content": "hello"}]),
    )
    assert out.choices[0].message.content is not None
