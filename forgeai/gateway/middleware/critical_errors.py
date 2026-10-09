"""Catch unhandled errors; SSE on infer, JSON elsewhere (no stack traces)."""

from __future__ import annotations

import logging
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from forgeai.gateway.constants import INFER_PATH
from forgeai.gateway.sse import sse_error_chunk

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class CriticalExceptionMiddleware(BaseHTTPMiddleware):
    """Last-resort handler; logs CRITICAL with request context."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:
            rid = str(getattr(request.state, "request_id", ""))
            tenant = getattr(request.state, "tenant_id_log", None)
            _LOG.critical(
                "gateway_unhandled_exception request_id=%s tenant_id=%s path=%s",
                rid,
                tenant,
                request.url.path,
                exc_info=True,
            )
            if request.url.path == INFER_PATH:
                body = sse_error_chunk(error="internal_error", request_id=rid)
                return Response(
                    content=body,
                    status_code=500,
                    media_type="text/event-stream",
                )
            return JSONResponse({"detail": "internal_error"}, status_code=500)


__all__ = ["CriticalExceptionMiddleware"]
