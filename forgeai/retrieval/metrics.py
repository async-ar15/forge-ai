"""Retrieval metric aliases sourced from observability metrics module."""

from __future__ import annotations

from forgeai.observability.metrics import (
    RETRIEVAL_CACHE_HIT_TOTAL,
    RETRIEVAL_PREFETCH_ERRORS_TOTAL,
    RETRIEVAL_QDRANT_ERRORS_TOTAL,
    RETRIEVAL_REQUEST_DURATION_SECONDS,
)

__all__ = [
    "RETRIEVAL_CACHE_HIT_TOTAL",
    "RETRIEVAL_PREFETCH_ERRORS_TOTAL",
    "RETRIEVAL_QDRANT_ERRORS_TOTAL",
    "RETRIEVAL_REQUEST_DURATION_SECONDS",
]
