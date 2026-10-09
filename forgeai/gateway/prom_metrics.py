"""Gateway metric aliases sourced from observability metrics module."""

from __future__ import annotations

from forgeai.observability.metrics import (
    GATEWAY_REQUEST_DURATION_SECONDS,
    GATEWAY_REQUESTS_TOTAL,
)

__all__ = [
    "GATEWAY_REQUEST_DURATION_SECONDS",
    "GATEWAY_REQUESTS_TOTAL",
]
