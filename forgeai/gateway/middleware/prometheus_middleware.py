"""Prometheus latency histogram and request counter for HTTP traffic."""

from __future__ import annotations

import time
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from forgeai.gateway.constants import INFER_PATH
from forgeai.observability.metrics import (
    GATEWAY_REQUEST_DURATION_SECONDS,
    GATEWAY_REQUESTS_TOTAL,
)

_DEFAULT_LABEL: Final[str] = "unknown"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Observe duration after the response is fully generated (streaming included)."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == INFER_PATH:
            return await call_next(request)
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        qtype = getattr(request.state, "query_type_log", _DEFAULT_LABEL)
        tier = getattr(request.state, "tenant_tier_log", _DEFAULT_LABEL)
        code = str(response.status_code)
        GATEWAY_REQUEST_DURATION_SECONDS.labels(
            query_type=str(qtype),
            tenant_tier=str(tier),
        ).observe(dt)
        GATEWAY_REQUESTS_TOTAL.labels(
            query_type=str(qtype),
            tenant_tier=str(tier),
            status_code=code,
        ).inc()
        return response


__all__ = ["PrometheusMiddleware"]
