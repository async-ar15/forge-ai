"""Owns: ORM mapping for checkpoints linked to training runs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.training_run import TrainingRun


class Checkpoint(Base):
    """Persisted checkpoint metadata and verification status."""

    __tablename__ = "checkpoints"

    checkpoint_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("training_runs.training_run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    s3_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    loss_at_step: Mapped[float] = mapped_column(Float, nullable=False)
    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    training_run: Mapped[TrainingRun] = relationship(
        "TrainingRun",
        foreign_keys=[training_run_id],
        back_populates="checkpoints",
    )
    best_for_training_run: Mapped[TrainingRun | None] = relationship(
        "TrainingRun",
        foreign_keys="TrainingRun.best_checkpoint_id",
        back_populates="best_checkpoint",
        uselist=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": str(self.checkpoint_id),
            "training_run_id": str(self.training_run_id),
            "step": self.step,
            "s3_path": self.s3_path,
            "sha256_hash": self.sha256_hash,
            "loss_at_step": self.loss_at_step,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
        }


__all__ = ["Checkpoint"]
