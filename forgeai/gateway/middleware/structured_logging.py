"""INFO access log with ``request_id`` bound via contextvars."""

from __future__ import annotations

import logging
import time
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from forgeai.gateway.constants import INFER_PATH
from forgeai.gateway.log_context import reset_request_id, set_request_id

_LOG: Final[logging.Logger] = logging.getLogger("forgeai.gateway.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log one line per request; ``request_id`` comes from logging filter context."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == INFER_PATH:
            return await call_next(request)
        rid = str(getattr(request.state, "request_id", ""))
        token = set_request_id(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = int(response.status_code)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            tenant = getattr(request.state, "tenant_id_log", None)
            qtype = getattr(request.state, "query_type_log", None)
            r_on = getattr(request.state, "r_online_log", None)
            _LOG.info(
                "gateway_request request_id=%s tenant_id=%s query_type=%s "
                "latency_ms=%.3f status_code=%s r_online=%s",
                rid,
                tenant,
                qtype,
                elapsed_ms,
                status,
                r_on,
            )
            reset_request_id(token)


__all__ = ["StructuredLoggingMiddleware"]
