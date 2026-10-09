"""End-of-stream metrics and access logs for ``POST /v1/infer`` (SSE)."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Final

from starlette.requests import Request

from forgeai.gateway.log_context import reset_request_id, set_request_id
from forgeai.observability.metrics import (
    GATEWAY_REQUEST_DURATION_SECONDS,
    GATEWAY_REQUESTS_TOTAL,
)

_LOG: Final[logging.Logger] = logging.getLogger("forgeai.gateway.access")


def record_infer_terminal(
    request: Request,
    *,
    status_code: int,
    latency_seconds: float,
) -> None:
    """Prometheus + access log for infer paths that return without an SSE body."""

    rid = str(getattr(request.state, "request_id", ""))
    tier = getattr(request.state, "tenant_tier_log", "unknown")
    qtype = getattr(request.state, "query_type_log", "unknown")
    tenant = getattr(request.state, "tenant_id_log", None)
    GATEWAY_REQUEST_DURATION_SECONDS.labels(
        query_type=str(qtype),
        tenant_tier=str(tier),
    ).observe(latency_seconds)
    GATEWAY_REQUESTS_TOTAL.labels(
        query_type=str(qtype),
        tenant_tier=str(tier),
        status_code=str(status_code),
    ).inc()
    _LOG.info(
        "gateway_request request_id=%s tenant_id=%s query_type=%s "
        "latency_ms=%.3f status_code=%s r_online=%s",
        rid,
        tenant,
        qtype,
        latency_seconds * 1000.0,
        status_code,
        None,
    )


async def wrap_infer_sse_observability(
    request: Request,
    inner: AsyncIterator[str],
    *,
    http_status: int,
) -> AsyncIterator[str]:
    """Observe wall time until the SSE body completes; emit one access log line."""

    rid = str(getattr(request.state, "request_id", ""))
    token = set_request_id(rid)
    t0 = time.perf_counter()
    code = http_status
    try:
        async for chunk in inner:
            yield chunk
    finally:
        dt = time.perf_counter() - t0
        tier = getattr(request.state, "tenant_tier_log", "unknown")
        qtype = getattr(request.state, "query_type_log", "unknown")
        r_on = getattr(request.state, "r_online_log", None)
        tenant = getattr(request.state, "tenant_id_log", None)
        GATEWAY_REQUEST_DURATION_SECONDS.labels(
            query_type=str(qtype),
            tenant_tier=str(tier),
        ).observe(dt)
        GATEWAY_REQUESTS_TOTAL.labels(
            query_type=str(qtype),
            tenant_tier=str(tier),
            status_code=str(code),
        ).inc()
        _LOG.info(
            "gateway_request request_id=%s tenant_id=%s query_type=%s "
            "latency_ms=%.3f status_code=%s r_online=%s",
            rid,
            tenant,
            qtype,
            dt * 1000.0,
            code,
            r_on,
        )
        reset_request_id(token)


__all__ = ["record_infer_terminal", "wrap_infer_sse_observability"]
