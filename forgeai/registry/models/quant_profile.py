"""Owns: ORM mapping for quant_profiles (Section 4).

Does not own: AWQ/GPTQ kernels, per-layer calibration jobs, or GPU kernel autotuning.
"""

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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import GlobalPrecision, QuantMethod
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.eval_run import EvalRun
    from forgeai.registry.models.model_version import ModelVersion


def _method_check() -> CheckConstraint:
    """CHECK for quant_profiles.method."""

    values = ", ".join(repr(m.value) for m in QuantMethod)
    return CheckConstraint(f"method IN ({values})", name="ck_quant_profiles_method")


def _precision_check() -> CheckConstraint:
    """CHECK for quant_profiles.global_precision."""

    values = ", ".join(repr(m.value) for m in GlobalPrecision)
    return CheckConstraint(
        f"global_precision IN ({values})",
        name="ck_quant_profiles_global_precision",
    )


class QuantProfile(Base):
    """Quantization profile bound to a model version and validating eval run."""

    __tablename__ = "quant_profiles"
    __table_args__ = (_method_check(), _precision_check())

    profile_id: Mapped[uuid.UUID] = mapped_column(
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
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    global_precision: Mapped[str] = mapped_column(String(16), nullable=False)
    layer_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    accuracy_delta: Mapped[float] = mapped_column(Float, nullable=False)
    latency_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    memory_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("eval_runs.eval_run_id", ondelete="RESTRICT"),
        nullable=False,
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

    model_version: Mapped[ModelVersion] = relationship(
        "ModelVersion",
        back_populates="quant_profiles",
        foreign_keys=[model_version_id],
    )
    eval_run: Mapped[EvalRun] = relationship("EvalRun", back_populates="quant_profiles")
    model_versions_pointing_here: Mapped[list[ModelVersion]] = relationship(
        "ModelVersion",
        back_populates="active_quant_profile",
        foreign_keys="ModelVersion.quant_profile_id",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"QuantProfile(profile_id={self.profile_id!r}, method={self.method!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "profile_id": str(self.profile_id),
            "model_version_id": str(self.model_version_id),
            "method": self.method,
            "global_precision": self.global_precision,
            "layer_config": dict(self.layer_config),
            "accuracy_delta": self.accuracy_delta,
            "latency_improvement": self.latency_improvement,
            "memory_mb": self.memory_mb,
            "eval_run_id": str(self.eval_run_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["QuantProfile"]
