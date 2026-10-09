"""Owns: shared domain enumerations referenced across ForgeAI services.

Does not own: persistence encoding, protobuf enum wire values, or client-specific DTOs.
"""

from __future__ import annotations

from enum import StrEnum


class QueryType(StrEnum):
    """Client-supplied query classification for v1 routing."""

    RAG = "rag"
    CODE = "code"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    UNKNOWN = "unknown"


class TenantTier(StrEnum):
    """Commercial tier used in state features and static cache-hit fallbacks."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ModelTier(StrEnum):
    """Model capacity class selected by the policy engine."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Precision(StrEnum):
    """Numeric precision tier for quantized execution."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class RetrievalMode(StrEnum):
    """Retrieval strategy arm of the contextual bandit."""

    OFF = "off"
    CACHE_ONLY = "cache_only"
    FULL = "full"


class OutputBudget(StrEnum):
    """Bounded generation length class."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ExplorationType(StrEnum):
    """How exploration was applied when logging routing decisions."""

    NONE = "none"
    EPSILON = "epsilon"
    UCB = "ucb"
    FORCED = "forced"


class MetricName(StrEnum):
    """Prometheus metric base names owned by observability exporters."""

    REQUEST_LATENCY_SECONDS = "forgeai_request_latency_seconds"
    REQUEST_COST_USD = "forgeai_request_cost_usd"
    CACHE_HIT_TOTAL = "forgeai_cache_hit_total"


__all__ = [
    "ExplorationType",
    "MetricName",
    "ModelTier",
    "OutputBudget",
    "Precision",
    "QueryType",
    "RetrievalMode",
    "TenantTier",
]
