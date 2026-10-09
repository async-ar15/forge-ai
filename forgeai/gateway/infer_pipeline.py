"""Inference hot path: features → policy → execution stream → decision log."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from datetime import UTC, datetime
from typing import Final, cast

import grpc.aio
from google.protobuf.timestamp_pb2 import Timestamp

from forgeai.gateway.decision_build import (
    build_request_outcome,
    decision_from_infer_completion,
)
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.schemas import InferRequest
from forgeai.gateway.sse import sse_error_chunk, sse_infer_chunk
from forgeai.gateway.tenant_context import TenantContext
from forgeai.policy.features import FeatureExtractionInput, FeatureVector
from forgeai.proto import common_pb2, execution_service_pb2, policy_service_pb2

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def _ingress_ts() -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(UTC))
    return ts


def build_policy_request(
    *,
    request_id: uuid.UUID,
    tenant_id: uuid.UUID,
    state_vec: common_pb2.StateVector,
    deployment_version: str,
) -> policy_service_pb2.PolicyDecideRequest:
    return policy_service_pb2.PolicyDecideRequest(
        correlation_request_id=str(request_id),
        tenant_id=str(tenant_id),
        routing_state_features=state_vec,
        execution_deployment_version=deployment_version,
        gateway_ingress_timestamp_utc=_ingress_ts(),
    )


def build_execution_request(
    *,
    request_id: uuid.UUID,
    tenant_id: uuid.UUID,
    action: common_pb2.ActionSpec,
    prompt: str,
    max_tokens: int,
    session_id: str | None,
) -> execution_service_pb2.ExecutionExecuteRequest:
    req = execution_service_pb2.ExecutionExecuteRequest(
        correlation_request_id=str(request_id),
        tenant_id=str(tenant_id),
        routing_action_spec=action,
        user_prompt_utf8=prompt,
        max_output_token_limit=max_tokens,
        execution_start_timestamp_utc=_ingress_ts(),
    )
    if session_id:
        req.broker_session_id_utf8 = session_id
    return req


async def extract_feature_vector(
    gw: GatewayState,
    body: InferRequest,
    tenant: TenantContext,
) -> FeatureVector:
    inp = FeatureExtractionInput(
        query_text=body.query,
        token_budget=body.token_budget,
        query_type_raw=body.query_type,
        latency_slo_ms_raw=body.latency_slo_ms,
        tenant_tier_raw=tenant.tenant_tier,
    )
    return await gw.feature_extractor.extract(inp)


async def call_policy_decide(
    gw: GatewayState,
    *,
    request_id: uuid.UUID,
    tenant: TenantContext,
    features: FeatureVector,
) -> policy_service_pb2.PolicyDecideResponse:
    pr = build_policy_request(
        request_id=request_id,
        tenant_id=tenant.tenant_id,
        state_vec=features.to_proto(),
        deployment_version=gw.settings.gateway_execution_deployment_version,
    )
    md = (("x-request-id", str(request_id)),)
    call = gw.policy_stub.Decide(pr, metadata=md)
    return await cast(
        Awaitable[policy_service_pb2.PolicyDecideResponse],
        call,
    )


async def log_and_sink_r_online(
    gw: GatewayState,
    *,
    request_id: uuid.UUID,
    tenant: TenantContext,
    features: FeatureVector,
    policy_resp: policy_service_pb2.PolicyDecideResponse,
    lat_ms: float,
    cost_micro: int | None,
    fallback_used: bool,
    sink: dict[str, float],
) -> None:
    outcome = build_request_outcome(
        final_latency_ms=lat_ms,
        cost_microdollars=cost_micro,
        feature_vector=features,
        fallback_used=fallback_used,
    )
    decision = decision_from_infer_completion(
        request_id=request_id,
        tenant_id=tenant.tenant_id,
        deployment_version=gw.settings.gateway_execution_deployment_version,
        features=features,
        policy_resp=policy_resp,
        outcome=outcome,
        tenant_policy=gw.tenant_policy,
    )
    await gw.decision_logger.log_decision(decision)
    reward = await gw.reward_pipeline.compute_and_record(decision, outcome)
    sink["r_online"] = float(reward)


async def execution_sse_chunks(
    gw: GatewayState,
    *,
    request_id: uuid.UUID,
    tenant: TenantContext,
    body: InferRequest,
    features: FeatureVector,
    policy_resp: policy_service_pb2.PolicyDecideResponse,
    sink: dict[str, float],
) -> AsyncIterator[str]:
    """Stream tokens; always logs exactly one decision after the stream ends."""

    rid_s = str(request_id)
    t0 = time.perf_counter()
    exec_req = build_execution_request(
        request_id=request_id,
        tenant_id=tenant.tenant_id,
        action=policy_resp.chosen_action,
        prompt=body.query,
        max_tokens=body.token_budget,
        session_id=body.session_id,
    )
    md = (("x-request-id", rid_s),)
    text_buf: list[str] = []
    full_text_parts: list[str] = []
    cost_micro: int | None = None
    fallback_hint = False
    err_text: str | None = None
    saw_terminal = False
    try:
        stream = gw.execution_stub.Execute(exec_req, metadata=md)
        ait = cast(
            AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk],
            stream,
        )
        async for chunk in ait:
            if chunk.generated_text_delta_utf8:
                full_text_parts.append(chunk.generated_text_delta_utf8)
            if body.stream and chunk.generated_text_delta_utf8:
                yield sse_infer_chunk(
                    token=chunk.generated_text_delta_utf8,
                    is_final=False,
                    cost_microdollars=None,
                    request_id=rid_s,
                )
            if not body.stream:
                text_buf.append(chunk.generated_text_delta_utf8)
            if chunk.HasField("gateway_fallback_used_hint"):
                fallback_hint = bool(chunk.gateway_fallback_used_hint)
            if not chunk.stream_finished:
                continue
            saw_terminal = True
            if chunk.HasField("stream_final_cost_microdollars"):
                cost_micro = int(
                    chunk.stream_final_cost_microdollars.amount_microdollars,
                )
            if chunk.stream_error_message_utf8:
                err_text = chunk.stream_error_message_utf8
            break
    except grpc.aio.AioRpcError as exc:
        _LOG.warning("execution_grpc_failed code=%s", exc.code())
        yield sse_error_chunk(error="execution_unavailable", request_id=rid_s)
        return
    if not saw_terminal:
        err_text = err_text or "execution_stream_incomplete"
    lat_ms = (time.perf_counter() - t0) * 1000.0
    await log_and_sink_r_online(
        gw,
        request_id=request_id,
        tenant=tenant,
        features=features,
        policy_resp=policy_resp,
        lat_ms=lat_ms,
        cost_micro=cost_micro,
        fallback_used=fallback_hint,
        sink=sink,
    )
    if err_text is not None:
        yield sse_error_chunk(error=err_text, request_id=rid_s)
        return
    _schedule_response_store_write(
        gw,
        request_id=request_id,
        query=body.query,
        response_text="".join(full_text_parts),
    )
    if not body.stream and text_buf:
        yield sse_infer_chunk(
            token="".join(text_buf),
            is_final=False,
            cost_microdollars=None,
            request_id=rid_s,
        )
    yield sse_infer_chunk(
        token="",
        is_final=True,
        cost_microdollars=cost_micro,
        request_id=rid_s,
    )


def _schedule_response_store_write(
    gw: GatewayState,
    *,
    request_id: uuid.UUID,
    query: str,
    response_text: str,
) -> None:
    task = asyncio.create_task(
        gw.response_store.store(
            request_id=request_id,
            query=query,
            response_text=response_text,
            retrieved_context="",
        )
    )
    task.add_done_callback(
        lambda t: _on_response_store_done(t, request_id=request_id),
    )


def _on_response_store_done(task: asyncio.Task[None], *, request_id: uuid.UUID) -> None:
    try:
        task.result()
    except Exception:
        _LOG.error("response_store_background_write_failed request_id=%s", request_id)


__all__ = [
    "build_execution_request",
    "build_policy_request",
    "call_policy_decide",
    "execution_sse_chunks",
    "extract_feature_vector",
]
