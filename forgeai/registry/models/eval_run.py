"""Owns: ORM mapping for eval_runs (Section 4).

Does not own: judge prompt assembly, dataset row storage formats,
or score aggregation jobs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import EvalType
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.dataset import Dataset
    from forgeai.registry.models.model_version import ModelVersion
    from forgeai.registry.models.quant_profile import QuantProfile


def _eval_type_check() -> CheckConstraint:
    """CHECK for eval_runs.eval_type."""

    values = ", ".join(repr(m.value) for m in EvalType)
    return CheckConstraint(f"eval_type IN ({values})", name="ck_eval_runs_eval_type")


class EvalRun(Base):
    """Evaluation execution record for a specific model version and dataset."""

    __tablename__ = "eval_runs"
    __table_args__ = (_eval_type_check(),)

    eval_run_id: Mapped[uuid.UUID] = mapped_column(
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
    eval_type: Mapped[str] = mapped_column(String(32), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("datasets.dataset_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
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
        back_populates="eval_runs",
    )
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="eval_runs")
    quant_profiles: Mapped[list[QuantProfile]] = relationship(
        "QuantProfile",
        back_populates="eval_run",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return (
            f"EvalRun(eval_run_id={self.eval_run_id!r}, "
            f"eval_type={self.eval_type!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "eval_run_id": str(self.eval_run_id),
            "model_version_id": str(self.model_version_id),
            "eval_type": self.eval_type,
            "judge_model": self.judge_model,
            "dataset_id": str(self.dataset_id),
            "score": self.score,
            "score_breakdown": dict(self.score_breakdown),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["EvalRun"]
