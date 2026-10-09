"""Tenant identity resolved from API keys (or fail-open defaults)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Request-scoped tenant after auth (or synthetic fail-open row)."""

    tenant_id: uuid.UUID
    tenant_tier: str
    auth_degraded: bool


def fail_open_context() -> TenantContext:
    """Synthetic free-tier identity when auth backends are unavailable."""

    return TenantContext(
        tenant_id=uuid.UUID(int=0),
        tenant_tier="free",
        auth_degraded=True,
    )


__all__ = ["TenantContext", "fail_open_context"]
