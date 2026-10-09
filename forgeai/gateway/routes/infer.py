"""Public inference entrypoint (SSE)."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

import grpc.aio
from fastapi import APIRouter, HTTPException, Request
from forgeai.gateway.constants import INFER_PATH
from forgeai.gateway.forge_headers import routing_headers
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.infer_observability import (
    record_infer_terminal,
    wrap_infer_sse_observability,
)
from forgeai.gateway.infer_pipeline import (
    call_policy_decide,
    execution_sse_chunks,
    extract_feature_vector,
)
from forgeai.gateway.middleware.auth_and_limit import get_tenant_context
from forgeai.gateway.schemas import InferRequest
from forgeai.gateway.sse import sse_error_chunk
from starlette.responses import Response, StreamingResponse

router = APIRouter(tags=["infer"])


@router.post(INFER_PATH, response_model=None)
async def post_infer(
    request: Request,
    body: InferRequest,
) -> Response | StreamingResponse:
    """Stream tokens as SSE; policy/execution hard failures use HTTP 503."""

    gw: GatewayState = request.app.state.gateway
    request.state.query_type_log = body.query_type
    tenant = get_tenant_context(request)
    rid = uuid.UUID(str(request.state.request_id))
    rid_s = str(rid)
    sink: dict[str, float] = {}
    t_infer = time.perf_counter()
    try:
        features = await extract_feature_vector(gw, body, tenant)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        policy_resp = await call_policy_decide(
            gw,
            request_id=rid,
            tenant=tenant,
            features=features,
        )
    except grpc.aio.AioRpcError:
        record_infer_terminal(
            request,
            status_code=503,
            latency_seconds=time.perf_counter() - t_infer,
        )
        body503 = sse_error_chunk(error="policy_unavailable", request_id=rid_s)
        return Response(
            content=body503,
            status_code=503,
            media_type="text/event-stream",
            headers=routing_headers(
                request_id=rid,
                policy=None,
                cost_usd=0.0,
                latency_ms=0,
            ),
        )

    async def _core() -> AsyncIterator[str]:
        async for part in execution_sse_chunks(
            gw,
            request_id=rid,
            tenant=tenant,
            body=body,
            features=features,
            policy_resp=policy_resp,
            sink=sink,
        ):
            yield part
        request.state.r_online_log = sink.get("r_online")

    async def _observed() -> AsyncIterator[str]:
        async for x in wrap_infer_sse_observability(
            request,
            _core(),
            http_status=200,
        ):
            yield x

    return StreamingResponse(
        _observed(),
        media_type="text/event-stream",
        status_code=200,
        headers=routing_headers(
            request_id=rid,
            policy=policy_resp,
            cost_usd=0.0,
            latency_ms=0,
        ),
    )


__all__ = ["post_infer", "router"]
