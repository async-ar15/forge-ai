"""Owns: ORM mapping for tenant_policies (Section 4).

Does not own: admin UI for per-tenant coefficient edits (deferred v2)
or cache invalidation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.tenant import Tenant


class TenantPolicy(Base):
    """Per-tenant reward coefficient vector (v1 uses shared defaults)."""

    __tablename__ = "tenant_policies"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id", ondelete="CASCADE"),
        primary_key=True,
    )
    coeff_latency: Mapped[float] = mapped_column(Float, nullable=False)
    coeff_cost: Mapped[float] = mapped_column(Float, nullable=False)
    coeff_slo_violation: Mapped[float] = mapped_column(Float, nullable=False)
    coeff_fallback: Mapped[float] = mapped_column(Float, nullable=False)
    coeff_retry: Mapped[float] = mapped_column(Float, nullable=False)
    coeff_cache_hit: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    tenant: Mapped[Tenant] = relationship("Tenant", back_populates="policies")

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"TenantPolicy(tenant_id={self.tenant_id!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "tenant_id": str(self.tenant_id),
            "coeff_latency": self.coeff_latency,
            "coeff_cost": self.coeff_cost,
            "coeff_slo_violation": self.coeff_slo_violation,
            "coeff_fallback": self.coeff_fallback,
            "coeff_retry": self.coeff_retry,
            "coeff_cache_hit": self.coeff_cache_hit,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["TenantPolicy"]
