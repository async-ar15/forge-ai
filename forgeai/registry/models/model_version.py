"""Owns: ORM mapping for model_versions (Section 4).

Does not own: promotion workflows, canary traffic rules, or artifact download ACLs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.artifact import Artifact
    from forgeai.registry.models.deployment import Deployment
    from forgeai.registry.models.eval_run import EvalRun
    from forgeai.registry.models.ml_model import RegistryModel
    from forgeai.registry.models.quant_profile import QuantProfile
    from forgeai.registry.models.training_run import TrainingRun


class ModelVersion(Base):
    """Versioned model artifact binding and optional quantization profile link."""

    __tablename__ = "model_versions"

    version_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_tag: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("artifacts.artifact_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    quant_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("quant_profiles.profile_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("training_runs.training_run_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
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
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    model: Mapped[RegistryModel] = relationship(
        "RegistryModel",
        back_populates="versions",
    )
    artifact: Mapped[Artifact] = relationship(
        "Artifact",
        back_populates="model_versions",
    )
    active_quant_profile: Mapped[QuantProfile | None] = relationship(
        "QuantProfile",
        foreign_keys=[quant_profile_id],
        back_populates="model_versions_pointing_here",
        uselist=False,
    )
    quant_profiles: Mapped[list[QuantProfile]] = relationship(
        "QuantProfile",
        back_populates="model_version",
        foreign_keys="QuantProfile.model_version_id",
    )
    eval_runs: Mapped[list[EvalRun]] = relationship(
        "EvalRun",
        back_populates="model_version",
    )
    deployments: Mapped[list[Deployment]] = relationship(
        "Deployment",
        back_populates="model_version",
    )
    training_run: Mapped[TrainingRun | None] = relationship(
        "TrainingRun",
        back_populates="produced_model_versions",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"ModelVersion(version_id={self.version_id!r}, tag={self.version_tag!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "version_id": str(self.version_id),
            "model_id": str(self.model_id),
            "version_tag": self.version_tag,
            "artifact_id": str(self.artifact_id),
            "quant_profile_id": (
                str(self.quant_profile_id) if self.quant_profile_id else None
            ),
            "training_run_id": (
                str(self.training_run_id) if self.training_run_id else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
        }


__all__ = ["ModelVersion"]
