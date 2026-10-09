"""Owns: async SQLAlchemy engine factory aimed at Postgres-backed registry tables.

Does not own: Alembic migration graph definitions or synchronous admin tooling.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine as sa_create_async_engine,
)

from forgeai.config import Settings


def create_registry_engine(settings: Settings) -> AsyncEngine:
    """Create a pooled async engine for application use."""

    return sa_create_async_engine(settings.database_url, future=True)


__all__ = ["create_registry_engine"]
