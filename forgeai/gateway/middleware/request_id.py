"""Attach a UUID ``request_id`` to each inbound HTTP request."""

from __future__ import annotations

import uuid
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER: Final[str] = "X-Request-Id"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate ``request_id``, store on ``request.state``, echo in response headers."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        rid = str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


__all__ = ["REQUEST_ID_HEADER", "RequestIDMiddleware"]
