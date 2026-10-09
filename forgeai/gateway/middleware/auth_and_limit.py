"""API key authentication and Redis sliding-window rate limits."""

from __future__ import annotations

import logging
import uuid
from typing import Final, cast

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from forgeai.gateway.auth_core import resolve_tenant_context
from forgeai.gateway.constants import (
    INFER_PATH,
    OPENAI_CHAT_COMPLETIONS_PATH,
    OPENAI_MODELS_PATH,
)
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.rate_limit import (
    requests_per_minute_for_tier,
    sliding_window_allow,
)
from forgeai.gateway.tenant_context import TenantContext
from forgeai.observability.metrics import GATEWAY_RATE_LIMIT_REJECTIONS_TOTAL

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_UNAUTHORIZED_BODY: Final[dict[str, str]] = {"detail": "unauthorized"}
_AUTH_PATHS: Final[frozenset[str]] = frozenset(
    {
        INFER_PATH,
        OPENAI_CHAT_COMPLETIONS_PATH,
        OPENAI_MODELS_PATH,
        "/chat/completions",
        "/models",
    }
)
_DEMO_TENANT_ID: Final[uuid.UUID] = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _api_key_header(request: Request) -> str | None:
    return request.headers.get("X-API-Key")


class AuthAndRateLimitMiddleware(BaseHTTPMiddleware):
    """Validate ``X-API-Key`` and enforce per-tenant RPM on API entrypoints."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path not in _AUTH_PATHS:
            return await call_next(request)
        gw: GatewayState = request.app.state.gateway
        if gw.settings.forgeai_demo_mode:
            demo_ctx = TenantContext(
                tenant_id=_DEMO_TENANT_ID,
                tenant_tier="free",
                auth_degraded=False,
            )
            request.state.tenant_context = demo_ctx
            request.state.tenant_id_log = str(demo_ctx.tenant_id)
            request.state.tenant_tier_log = demo_ctx.tenant_tier
            return await call_next(request)
        raw = _api_key_header(request)
        ctx, ok = await resolve_tenant_context(
            raw_api_key=raw,
            redis=gw.redis_client,
            session_factory=gw.session_factory,
        )
        if not ok or ctx is None:
            return JSONResponse(_UNAUTHORIZED_BODY, status_code=401)
        allowed, retry_after = await sliding_window_allow(
            gw.redis_client,
            tenant_id=ctx.tenant_id,
            limit=requests_per_minute_for_tier(ctx.tenant_tier),
        )
        if not allowed:
            GATEWAY_RATE_LIMIT_REJECTIONS_TOTAL.labels(
                tenant_tier=ctx.tenant_tier
            ).inc()
            return JSONResponse(
                {"detail": "rate_limit_exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        request.state.tenant_context = ctx
        request.state.tenant_id_log = str(ctx.tenant_id)
        request.state.tenant_tier_log = ctx.tenant_tier
        if ctx.auth_degraded:
            _LOG.warning(
                "auth_fail_open_request_allowed tenant_tier=%s",
                ctx.tenant_tier,
            )
        return await call_next(request)


def get_tenant_context(request: Request) -> TenantContext:
    """Return the authenticated tenant (infer route only, after auth middleware)."""

    return cast(TenantContext, request.state.tenant_context)


__all__ = ["AuthAndRateLimitMiddleware", "get_tenant_context"]
