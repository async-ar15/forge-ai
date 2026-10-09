"""Owns: SQLAlchemy DeclarativeBase for all registry ORM models.

Does not own: Alembic env wiring, async session factories, or migration revision bodies.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for ForgeAI registry tables."""

    pass


__all__ = ["Base"]
