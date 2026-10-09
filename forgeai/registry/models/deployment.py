"""Owns: ORM mapping for deployments (Section 4).

Does not own: Kubernetes ReplicaSet controllers, Istio virtual services,
or drain orchestration.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import DeploymentEnvironment, DeploymentStatus
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.model_version import ModelVersion


def _environment_check() -> CheckConstraint:
    """CHECK for deployments.environment."""

    values = ", ".join(repr(m.value) for m in DeploymentEnvironment)
    return CheckConstraint(
        f"environment IN ({values})",
        name="ck_deployments_environment",
    )


def _status_check() -> CheckConstraint:
    """CHECK for deployments.status."""

    values = ", ".join(repr(m.value) for m in DeploymentStatus)
    return CheckConstraint(f"status IN ({values})", name="ck_deployments_status")


class Deployment(Base):
    """Serving deployment row for a promoted model version."""

    __tablename__ = "deployments"
    __table_args__ = (_environment_check(), _status_check())

    deployment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_versions.version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    environment: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    replicas: Mapped[int] = mapped_column(Integer, nullable=False)
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    model_version: Mapped[ModelVersion] = relationship(
        "ModelVersion",
        back_populates="deployments",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return (
            f"Deployment(deployment_id={self.deployment_id!r}, "
            f"environment={self.environment!r}, status={self.status!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "deployment_id": str(self.deployment_id),
            "model_version_id": str(self.model_version_id),
            "environment": self.environment,
            "status": self.status,
            "replicas": self.replicas,
            "deployed_at": self.deployed_at.isoformat(),
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["Deployment"]
