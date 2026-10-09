"""Tenant-scoped reward coefficients (mirrors ``tenant_policies`` columns, no ORM).

Does not own: SQLAlchemy persistence, admin APIs, or coefficient learning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantPolicy:
    """Online reward coefficients; names match ``tenant_policies`` columns."""

    coeff_latency: float
    coeff_cost: float
    coeff_slo_violation: float
    coeff_fallback: float
    coeff_retry: float
    coeff_cache_hit: float


__all__ = ["TenantPolicy"]
