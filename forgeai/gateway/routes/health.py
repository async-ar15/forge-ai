"""Liveness and dependency hints for orchestrators (never HTTP 5xx)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from forgeai.gateway.constants import HEALTH_PATH
from forgeai.gateway.gateway_state import GatewayState

router = APIRouter(tags=["health"])


@router.get(HEALTH_PATH)
async def get_health(request: Request) -> dict[str, object]:
    """Return process-local health; failures surface as degraded fields only."""

    gw: GatewayState = request.app.state.gateway
    degraded = gw.policy_version_boot == "unavailable"
    status = "degraded" if degraded else "ok"
    return {
        "status": status,
        "policy_version": gw.policy_version_boot,
        "loaded_models": list(gw.loaded_models_boot),
    }


__all__ = ["get_health", "router"]
