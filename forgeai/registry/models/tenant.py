"""Owns: ORM mapping for the tenants anchor table (Section 4 FK target).

Does not own: auth identity linkage, billing metadata, or row-level security policies.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.routing_decision import RoutingDecision
    from forgeai.registry.models.tenant_policy import TenantPolicy


class Tenant(Base):
    """Tenant row implied by Section 4 foreign keys.

    Columns beyond PK are repository extensions.
    """

    __tablename__ = "tenants"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
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
    api_key_sha256_hex: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
    )
    api_key_bcrypt_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_tier: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="free",
    )

    policies: Mapped[list[TenantPolicy]] = relationship(
        "TenantPolicy",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    routing_decisions: Mapped[list[RoutingDecision]] = relationship(
        "RoutingDecision",
        back_populates="tenant",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"Tenant(tenant_id={self.tenant_id!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "tenant_id": str(self.tenant_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tenant_tier": self.tenant_tier,
        }


__all__ = ["Tenant"]
