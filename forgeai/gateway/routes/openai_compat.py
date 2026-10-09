"""OpenAI-compatible additive gateway endpoints."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, cast

import grpc.aio
import httpx
from fastapi import APIRouter, Request
from forgeai.gateway.decision_build import (
    build_request_outcome,
    decision_from_infer_completion,
)
from forgeai.gateway.forge_headers import routing_headers
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.infer_pipeline import (
    build_execution_request,
    call_policy_decide,
    extract_feature_vector,
)
from forgeai.gateway.middleware.auth_and_limit import get_tenant_context
from forgeai.gateway.openai_constants import (
    OPENAI_DONE_TOKEN,
    OPENAI_ENDPOINT_CHAT_COMPLETIONS,
    OPENAI_MODEL_ROUTED,
    OPENAI_OBJECT_CHAT_COMPLETION,
    OPENAI_OBJECT_CHAT_COMPLETION_CHUNK,
    OPENAI_OBJECT_LIST,
    OPENAI_OBJECT_MODEL,
    OPENAI_ROLE_ASSISTANT,
)
from forgeai.gateway.openai_translate import (
    estimate_token_count,
    infer_query_type_heuristic,
    last_user_message,
    messages_to_chatml_prompt,
    split_stream_deltas,
)
from forgeai.gateway.schemas import InferRequest, OpenAIChatCompletionsRequest
from forgeai.gateway.sse import format_sse_event
from forgeai.gateway.tenant_context import TenantContext
from forgeai.policy.bandit_actions import ALL_ACTIONS, canonical_action_key
from forgeai.policy.features import FeatureVector
from forgeai.proto import common_pb2, execution_service_pb2, policy_service_pb2
from starlette.responses import JSONResponse, Response, StreamingResponse

router = APIRouter(tags=["openai-compat"])
_OLLAMA_TIMEOUT_SECONDS: Final[float] = 30.0
_OLLAMA_DEMO_MAX_TOKENS: Final[int] = 50
_MODEL_IDS: Final[tuple[str, ...]] = (
    "forgeai-auto",
    "forgeai-small",
    "forgeai-medium",
    "forgeai-large",
)


@dataclass(frozen=True, slots=True)
class ExecutionCollectResult:
    text: str
    latency_ms: int
    cost_micro: int | None
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class RoutedExecutionResult:
    execution: ExecutionCollectResult
    policy: policy_service_pb2.PolicyDecideResponse
    reward: float
    cost_usd: float
    latency_ms: int


@router.get("/v1/models")
@router.get("/models")
async def list_models() -> JSONResponse:
    created = int(datetime.now(UTC).timestamp())
    data = [
        {
            "id": model_id,
            "object": OPENAI_OBJECT_MODEL,
            "created": created,
            "owned_by": "forgeai",
        }
        for model_id in _MODEL_IDS
    ]
    return JSONResponse({"object": OPENAI_OBJECT_LIST, "data": data}, status_code=200)


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(
    request: Request, body: OpenAIChatCompletionsRequest
) -> Response:
    gw: GatewayState = request.app.state.gateway
    tenant = get_tenant_context(request)
    rid = uuid.UUID(str(request.state.request_id))
    infer_body, prompt, query = _prepare_chat_request(body)
    request.state.query_type_log = infer_body.query_type
    features = await extract_feature_vector(gw, infer_body, tenant)
    routed: RoutedExecutionResult | None
    if gw.settings.forgeai_demo_mode:
        routed = await _demo_mode_result(
            gw,
            rid,
            tenant.tenant_id,
            prompt,
            features,
            body,
        )
    else:
        routed = await _normal_mode_result(
            gw,
            rid,
            tenant,
            infer_body,
            prompt,
            features,
            body,
        )
    if routed is None:
        return JSONResponse(
            {"error": "upstream_unavailable"},
            status_code=503,
            headers=routing_headers(
                request_id=rid,
                policy=None,
                cost_usd=0.0,
                latency_ms=0,
            ),
        )
    request.state.r_online_log = routed.reward
    await gw.response_store.store(rid, query, routed.execution.text, "")
    headers = routing_headers(
        request_id=rid,
        policy=routed.policy,
        cost_usd=routed.cost_usd,
        latency_ms=routed.latency_ms,
    )
    if not body.stream:
        payload = _non_stream_payload(rid, routed.execution.text, prompt)
        return JSONResponse(payload, status_code=200, headers=headers)
    chunks = _stream_chunks(rid, routed.execution.text)
    return StreamingResponse(chunks, media_type="text/event-stream", headers=headers)


def _prepare_chat_request(
    body: OpenAIChatCompletionsRequest,
) -> tuple[InferRequest, str, str]:
    qtype = infer_query_type_heuristic(body.messages)
    prompt = messages_to_chatml_prompt(body.messages)
    query = last_user_message(body.messages)
    infer_body = InferRequest(
        query=query,
        query_type=qtype,
        token_budget=body.max_tokens or 256,
        stream=False,
    )
    return infer_body, prompt, query


def _non_stream_payload(
    request_id: uuid.UUID, text: str, prompt: str
) -> dict[str, object]:
    created = int(datetime.now(UTC).timestamp())
    prompt_tokens = estimate_token_count(prompt)
    completion_tokens = estimate_token_count(text)
    return {
        "id": f"chatcmpl-{request_id}",
        "object": OPENAI_OBJECT_CHAT_COMPLETION,
        "created": created,
        "model": OPENAI_MODEL_ROUTED,
        "choices": [
            {
                "index": 0,
                "message": {"role": OPENAI_ROLE_ASSISTANT, "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _safe_policy(
    gw: GatewayState,
    rid: uuid.UUID,
    tenant: TenantContext,
    features: FeatureVector,
) -> policy_service_pb2.PolicyDecideResponse | None:
    try:
        return await call_policy_decide(
            gw,
            request_id=rid,
            tenant=tenant,
            features=features,
        )
    except grpc.aio.AioRpcError:
        return None


async def _execute_collect(
    gw: GatewayState,
    request_id: uuid.UUID,
    tenant_id: uuid.UUID,
    policy: policy_service_pb2.PolicyDecideResponse,
    prompt: str,
    infer_body: InferRequest,
) -> ExecutionCollectResult:
    req = build_execution_request(
        request_id=request_id,
        tenant_id=tenant_id,
        action=policy.chosen_action,
        prompt=prompt,
        max_tokens=infer_body.token_budget,
        session_id=None,
    )
    t0 = time.perf_counter()
    text_parts: list[str] = []
    cost_micro: int | None = None
    fallback_used = False
    stream = gw.execution_stub.Execute(
        req,
        metadata=(("x-request-id", str(request_id)),),
    )
    ait = cast(
        AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk],
        stream,
    )
    async for chunk in ait:
        if chunk.generated_text_delta_utf8:
            text_parts.append(chunk.generated_text_delta_utf8)
        if chunk.HasField("gateway_fallback_used_hint"):
            fallback_used = bool(chunk.gateway_fallback_used_hint)
        if not chunk.stream_finished:
            continue
        if chunk.HasField("stream_final_cost_microdollars"):
            cost_micro = int(
                chunk.stream_final_cost_microdollars.amount_microdollars,
            )
        break
    return ExecutionCollectResult(
        text="".join(text_parts),
        latency_ms=int((time.perf_counter() - t0) * 1000.0),
        cost_micro=cost_micro,
        fallback_used=fallback_used,
    )


async def _normal_mode_result(
    gw: GatewayState,
    rid: uuid.UUID,
    tenant: TenantContext,
    infer_body: InferRequest,
    prompt: str,
    features: FeatureVector,
    body: OpenAIChatCompletionsRequest,
) -> RoutedExecutionResult | None:
    policy = await _safe_policy(gw, rid, tenant, features)
    if policy is None:
        return None
    try:
        result = await _execute_collect(
            gw,
            rid,
            tenant.tenant_id,
            policy,
            prompt,
            infer_body,
        )
    except grpc.aio.AioRpcError:
        return None
    outcome = build_request_outcome(
        final_latency_ms=float(result.latency_ms),
        cost_microdollars=result.cost_micro,
        feature_vector=features,
        fallback_used=bool(result.fallback_used),
    )
    decision = decision_from_infer_completion(
        request_id=rid,
        tenant_id=tenant.tenant_id,
        deployment_version=gw.settings.gateway_execution_deployment_version,
        features=features,
        policy_resp=policy,
        outcome=outcome,
        tenant_policy=gw.tenant_policy,
        requested_model_hint=body.model,
        endpoint=OPENAI_ENDPOINT_CHAT_COMPLETIONS,
    )
    await gw.decision_logger.log_decision(decision)
    reward = await gw.reward_pipeline.compute_and_record(decision, outcome)
    return RoutedExecutionResult(
        execution=result,
        policy=policy,
        reward=float(reward),
        cost_usd=float(result.cost_micro or 0) / 1_000_000.0,
        latency_ms=result.latency_ms,
    )


async def _demo_mode_result(
    gw: GatewayState,
    rid: uuid.UUID,
    tenant_id: uuid.UUID,
    prompt: str,
    features: FeatureVector,
    body: OpenAIChatCompletionsRequest,
) -> RoutedExecutionResult:
    policy = _demo_policy_response()
    result = await _demo_collect(gw, prompt)
    outcome = build_request_outcome(
        final_latency_ms=float(result.latency_ms),
        cost_microdollars=result.cost_micro,
        feature_vector=features,
        fallback_used=False,
    )
    decision = decision_from_infer_completion(
        request_id=rid,
        tenant_id=tenant_id,
        deployment_version=gw.settings.gateway_execution_deployment_version,
        features=features,
        policy_resp=policy,
        outcome=outcome,
        tenant_policy=gw.tenant_policy,
        requested_model_hint=body.model,
        endpoint=OPENAI_ENDPOINT_CHAT_COMPLETIONS,
    )
    await gw.decision_logger.log_decision(decision)
    reward = await gw.reward_pipeline.compute_and_record(decision, outcome)
    return RoutedExecutionResult(
        execution=result,
        policy=policy,
        reward=float(reward),
        cost_usd=float(result.cost_micro or 0) / 1_000_000.0,
        latency_ms=result.latency_ms,
    )


async def _demo_collect(gw: GatewayState, prompt: str) -> ExecutionCollectResult:
    t0 = time.perf_counter()
    text = await _ollama_text(
        gw.settings.ollama_url,
        gw.settings.ollama_model,
        prompt,
        max_tokens=_OLLAMA_DEMO_MAX_TOKENS,
    )
    if text is None:
        text = (
            "ForgeAI demo mode: ollama not available. "
            "Start with make dev-up to enable real inference."
        )
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    tokens = max(1, estimate_token_count(text))
    cost_usd = Decimal(tokens) * Decimal("0.0000005")
    cost_micro = int((cost_usd * Decimal("1000000")).quantize(Decimal("1")))
    return ExecutionCollectResult(
        text=text,
        latency_ms=latency_ms,
        cost_micro=cost_micro,
        fallback_used=False,
    )


def _demo_policy_response() -> policy_service_pb2.PolicyDecideResponse:
    chosen = common_pb2.ActionSpec(
        model_tier=common_pb2.MODEL_TIER_SMALL,
        precision=common_pb2.PRECISION_FP16,
        retrieval_mode=common_pb2.RETRIEVAL_MODE_CACHE_ONLY,
        output_budget=common_pb2.OUTPUT_BUDGET_MEDIUM,
    )
    scores = {canonical_action_key(a): 0.0 for a in ALL_ACTIONS}
    return policy_service_pb2.PolicyDecideResponse(
        chosen_action=chosen,
        exploration_flag=False,
        exploration_type=common_pb2.EXPLORATION_TYPE_NONE,
        greedy_action=chosen,
        action_scores=scores,
        policy_version="a3f9c2",
    )


async def _ollama_text(
    url: str, model: str, prompt: str, *, max_tokens: int = _OLLAMA_DEMO_MAX_TOKENS
) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{url}/api/generate", json=payload)
            response.raise_for_status()
            body = cast(dict[str, Any], response.json())
    except Exception:
        return None
    text = body.get("response")
    return str(text) if isinstance(text, str) and text.strip() else None


async def _stream_chunks(request_id: uuid.UUID, text: str) -> AsyncIterator[str]:
    created = int(datetime.now(UTC).timestamp())
    for delta in split_stream_deltas(text):
        body = {
            "id": f"chatcmpl-{request_id}",
            "object": OPENAI_OBJECT_CHAT_COMPLETION_CHUNK,
            "created": created,
            "model": OPENAI_MODEL_ROUTED,
            "choices": [
                {"index": 0, "delta": {"content": delta}, "finish_reason": None}
            ],
        }
        yield format_sse_event(body)
    final = {
        "id": f"chatcmpl-{request_id}",
        "object": OPENAI_OBJECT_CHAT_COMPLETION_CHUNK,
        "created": created,
        "model": OPENAI_MODEL_ROUTED,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield format_sse_event(final)
    yield f"data: {OPENAI_DONE_TOKEN}\n\n"
