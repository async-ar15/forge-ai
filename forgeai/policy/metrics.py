"""Policy metric helpers sourced from the shared observability module."""

from __future__ import annotations

from forgeai.observability.metrics import FEATURE_EXTRACTION_FALLBACK_TOTAL


def record_feature_extraction_fallback(*, feature_name: str, reason: str) -> None:
    """Increment the fallback counter with fixed low-cardinality labels."""

    FEATURE_EXTRACTION_FALLBACK_TOTAL.labels(
        feature_name=feature_name,
        reason=reason,
    ).inc()


__all__ = [
    "FEATURE_EXTRACTION_FALLBACK_TOTAL",
    "record_feature_extraction_fallback",
]
