"""FastAPI application assembly (routes + middleware + lifespan)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI
from sqlalchemy import text

from forgeai.config import Settings, get_settings
from forgeai.gateway.bootstrap import bootstrap_gateway_state, shutdown_gateway_state
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.log_context import RequestIdFilter
from forgeai.gateway.middleware import (
    AuthAndRateLimitMiddleware,
    CriticalExceptionMiddleware,
    PrometheusMiddleware,
    RequestIDMiddleware,
    StructuredLoggingMiddleware,
)
from forgeai.gateway.routes.health import router as health_router
from forgeai.gateway.routes.infer import router as infer_router
from forgeai.gateway.routes.metrics_proxy import router as metrics_router
from forgeai.gateway.routes.openai_compat import router as openai_router

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def _install_request_id_filter() -> None:
    flt = RequestIdFilter()
    _loggers = (
        "forgeai.gateway",
        "forgeai.gateway.access",
        "forgeai.gateway.features",
    )
    for name in _loggers:
        logging.getLogger(name).addFilter(flt)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _install_request_id_filter()
    state = await bootstrap_gateway_state(settings)
    app.state.gateway = state
    try:
        async with state.registry_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        _LOG.warning("gateway_postgres_probe_failed", exc_info=True)
    _LOG.info(
        "gateway_started policy_version_boot=%s",
        state.policy_version_boot,
    )
    yield
    await shutdown_gateway_state(state)


def create_app() -> FastAPI:
    """Build the production gateway (reads ``Settings`` from the environment)."""

    application = FastAPI(
        title="ForgeAI Inference Gateway",
        version="0.1.0",
        lifespan=_lifespan,
    )
    application.add_middleware(CriticalExceptionMiddleware)
    application.add_middleware(AuthAndRateLimitMiddleware)
    application.add_middleware(PrometheusMiddleware)
    application.add_middleware(StructuredLoggingMiddleware)
    application.add_middleware(RequestIDMiddleware)
    application.include_router(infer_router)
    application.include_router(openai_router)
    application.include_router(health_router)
    application.include_router(metrics_router)
    return application


def create_test_app(settings: Settings, state: GatewayState) -> FastAPI:
    """Minimal app for tests (pre-built ``GatewayState`` on ``app.state.gateway``)."""

    @asynccontextmanager
    async def _empty_lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    application = FastAPI(lifespan=_empty_lifespan)
    application.state.gateway = state
    application.add_middleware(CriticalExceptionMiddleware)
    application.add_middleware(AuthAndRateLimitMiddleware)
    application.add_middleware(PrometheusMiddleware)
    application.add_middleware(StructuredLoggingMiddleware)
    application.add_middleware(RequestIDMiddleware)
    application.include_router(infer_router)
    application.include_router(openai_router)
    application.include_router(health_router)
    application.include_router(metrics_router)
    return application


__all__ = ["create_app", "create_test_app"]
