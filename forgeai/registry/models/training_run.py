"""Owns: ORM mapping for training_runs orchestration state."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import TrainingRunStatus
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.checkpoint import Checkpoint
    from forgeai.registry.models.ml_model import RegistryModel
    from forgeai.registry.models.model_version import ModelVersion


def _status_check() -> CheckConstraint:
    """CHECK for training_runs.status."""

    values = ", ".join(repr(m.value) for m in TrainingRunStatus)
    return CheckConstraint(f"status IN ({values})", name="ck_training_runs_status")


class TrainingRun(Base):
    """Training run metadata for async orchestration and checkpoint lineage."""

    __tablename__ = "training_runs"
    __table_args__ = (_status_check(),)

    training_run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
    )
    base_model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    current_epoch: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    current_loss: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0"
    )
    best_checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("checkpoints.checkpoint_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)

    produced_model_versions: Mapped[list[ModelVersion]] = relationship(
        "ModelVersion",
        back_populates="training_run",
    )
    base_model: Mapped[RegistryModel] = relationship("RegistryModel")
    checkpoints: Mapped[list[Checkpoint]] = relationship(
        "Checkpoint",
        foreign_keys="Checkpoint.training_run_id",
        back_populates="training_run",
    )
    best_checkpoint: Mapped[Checkpoint | None] = relationship(
        "Checkpoint",
        foreign_keys=[best_checkpoint_id],
        back_populates="best_for_training_run",
        uselist=False,
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return (
            f"TrainingRun(training_run_id={self.training_run_id!r}, "
            f"job_id={self.job_id!r}, status={self.status!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "training_run_id": str(self.training_run_id),
            "job_id": str(self.job_id),
            "base_model_id": str(self.base_model_id),
            "config_snapshot": dict(self.config_snapshot),
            "status": self.status,
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "best_checkpoint_id": (
                str(self.best_checkpoint_id) if self.best_checkpoint_id else None
            ),
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error_message": self.error_message,
            "seed": self.seed,
        }


__all__ = ["TrainingRun"]
