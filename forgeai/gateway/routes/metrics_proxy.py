"""Proxy to the process Prometheus scrape endpoint (internal token)."""

from __future__ import annotations

import hmac
import logging
from typing import Final

import httpx
from fastapi import APIRouter, Request, Response
from forgeai.gateway.constants import METRICS_PATH, METRICS_PROXY_TIMEOUT_SECONDS
from forgeai.gateway.gateway_state import GatewayState

router = APIRouter(tags=["metrics"])
_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_INTERNAL_HEADER = "X-Internal-Metrics-Token"


@router.get(METRICS_PATH)
async def get_metrics(request: Request) -> Response:
    """Forward ``/metrics`` from the configured Prometheus listener port."""

    gw: GatewayState = request.app.state.gateway
    expected = gw.settings.gateway_internal_metrics_token
    got = request.headers.get(_INTERNAL_HEADER) or ""
    if not expected or not hmac.compare_digest(got, expected):
        return Response(content="forbidden", status_code=403)
    url = f"http://127.0.0.1:{gw.settings.prometheus_metrics_port}/metrics"
    try:
        async with httpx.AsyncClient(timeout=METRICS_PROXY_TIMEOUT_SECONDS) as client:
            upstream = await client.get(url)
    except Exception:
        _LOG.warning("metrics_proxy_upstream_failed", exc_info=True)
        return Response(content="# metrics unavailable\n", media_type="text/plain")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="text/plain",
    )


__all__ = ["get_metrics", "router"]
