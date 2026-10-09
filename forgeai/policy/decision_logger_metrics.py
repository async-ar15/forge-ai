"""Prometheus counters for routing decision durability (Section 4).

Does not own: SQLAlchemy or filesystem fallback paths.
"""

from __future__ import annotations

from enum import StrEnum

from forgeai.observability.metrics import (
    DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL,
    DECISIONS_FALLBACK_TOTAL,
    DECISIONS_LABEL_MISSING_TOTAL,
    DECISIONS_LOGGED_TOTAL,
)


class DecisionFallbackReason(StrEnum):
    """Prometheus ``reason`` label for ``forgeai_decisions_fallback_total``."""

    DB_ERROR = "db_error"
    SERIALIZATION_ERROR = "serialization_error"


__all__ = [
    "DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL",
    "DECISIONS_FALLBACK_TOTAL",
    "DECISIONS_LABEL_MISSING_TOTAL",
    "DECISIONS_LOGGED_TOTAL",
    "DecisionFallbackReason",
]
