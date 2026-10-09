"""Owns: ORM mapping for the models table (Section 4).

Does not own: Hugging Face hub sync jobs, tokenizer assets, or GPU placement rules.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, CheckConstraint, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import ModelProvider
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.model_version import ModelVersion


def _provider_in_check() -> CheckConstraint:
    """Build CHECK constraint for models.provider from registry constants."""

    values = ", ".join(repr(m.value) for m in ModelProvider)
    return CheckConstraint(f"provider IN ({values})", name="ck_models_provider")


class RegistryModel(Base):
    """Registered base model metadata (table name `models`)."""

    __tablename__ = "models"
    __table_args__ = (_provider_in_check(),)

    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    base_architecture: Mapped[str] = mapped_column(String(128), nullable=False)
    parameter_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
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

    versions: Mapped[list[ModelVersion]] = relationship(
        "ModelVersion",
        back_populates="model",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"RegistryModel(model_id={self.model_id!r}, name={self.name!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "model_id": str(self.model_id),
            "name": self.name,
            "provider": self.provider,
            "base_architecture": self.base_architecture,
            "parameter_count": self.parameter_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["RegistryModel"]
