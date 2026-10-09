"""Owns: ORM mapping for the datasets anchor table required by eval_runs.dataset_id FK.

Does not own: dataset file manifests, PII classification, or cross-region replication.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.eval_run import EvalRun


class Dataset(Base):
    """Minimal eval dataset registry row.

    Section 4 references dataset_id without defining a parent table.
    """

    __tablename__ = "datasets"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
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

    eval_runs: Mapped[list[EvalRun]] = relationship(
        "EvalRun",
        back_populates="dataset",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"Dataset(dataset_id={self.dataset_id!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "dataset_id": str(self.dataset_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["Dataset"]
