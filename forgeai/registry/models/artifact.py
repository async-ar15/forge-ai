"""Owns: ORM mapping for the artifacts table (S3 path anchor for model_versions).

Does not own: presigned URL issuance, multipart upload state machines,
or virus scanning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.model_version import ModelVersion


class Artifact(Base):
    """Binary artifact metadata pointing at object storage (Section 4 note: S3 path)."""

    __tablename__ = "artifacts"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
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

    model_versions: Mapped[list[ModelVersion]] = relationship(
        "ModelVersion",
        back_populates="artifact",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"Artifact(artifact_id={self.artifact_id!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "artifact_id": str(self.artifact_id),
            "storage_uri": self.storage_uri,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["Artifact"]
