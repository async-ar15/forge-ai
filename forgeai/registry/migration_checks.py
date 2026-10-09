"""Owns: CHECK constraint text builders derived from registry StrEnum classes.

Does not own: Alembic revision ordering, index naming, or FK ON DELETE policies.
"""

from __future__ import annotations

from enum import StrEnum

import sqlalchemy as sa


def enum_in(column_sql: str, enum_cls: type[StrEnum], name: str) -> sa.CheckConstraint:
    """Return a CheckConstraint: `column_sql IN (...)` using enum member values."""

    values = ", ".join(repr(m.value) for m in enum_cls)
    return sa.CheckConstraint(sa.text(f"{column_sql} IN ({values})"), name=name)


def enum_in_or_null(
    column_sql: str,
    enum_cls: type[StrEnum],
    name: str,
) -> sa.CheckConstraint:
    """Return CheckConstraint allowing NULL or one of the enum values."""

    values = ", ".join(repr(m.value) for m in enum_cls)
    return sa.CheckConstraint(
        sa.text(f"({column_sql} IS NULL) OR ({column_sql} IN ({values}))"),
        name=name,
    )


__all__ = ["enum_in", "enum_in_or_null"]
