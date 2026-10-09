"""Policy-engine constants, wire enums for feature extraction, and static fallbacks.

Does not own: protobuf wire values, reward coefficients, or execution SLOs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from forgeai.policy.tenant_policy import TenantPolicy

DEFAULT_EXPLORATION_EPSILON: Final[str] = "0.05"


class PolicyConfigKey(StrEnum):
    """Environment keys that tune the Policy Engine (no literals in logic)."""

    EPSILON = "POLICY_EPSILON"


class LinUCBConfigKey(StrEnum):
    """Environment keys for disjoint LinUCB hyperparameters (Section 3)."""

    ALPHA = "LINUCB_ALPHA"
    EXPLORATION_EPSILON = "LINUCB_EXPLORATION_EPSILON"


class DecisionLoggerConfigKey(StrEnum):
    """Environment keys for routing decision durability (Section 4)."""

    DECISION_FALLBACK_PATH = "FORGEAI_DECISION_FALLBACK_PATH"


# Default JSONL sink when ``FORGEAI_DECISION_FALLBACK_PATH`` is unset
# (single named default).
DEFAULT_DECISION_FALLBACK_PATH: Final[str] = "/tmp/forgeai_decisions_fallback.jsonl"


class FeatureQueryType(StrEnum):
    """Normalized query classification on the policy hot path (Section 3)."""

    RAG = "rag"
    CODE = "code"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    UNKNOWN = "unknown"


class FeatureTenantTier(StrEnum):
    """Tenant tier from auth context.

    INTERNAL is reserved for ForgeAI system calls - judge, health checks,
    internal tooling. Always routes to conservative default action regardless of
    bandit scores.
    """

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


# --- Feature min-max bounds (Section 3, v1 fixed; never learned on the hot path) ---

FEATURE_QUERY_LEN_MIN: Final[int] = 0
FEATURE_QUERY_LEN_MAX: Final[int] = 32768
FEATURE_TOKEN_BUDGET_MIN: Final[int] = 0
FEATURE_TOKEN_BUDGET_MAX: Final[int] = 32768
FEATURE_LATENCY_SLO_MS_MIN: Final[int] = 0
FEATURE_LATENCY_SLO_MS_MAX: Final[int] = 10000
FEATURE_QUEUE_DEPTH_MIN: Final[int] = 0
FEATURE_QUEUE_DEPTH_MAX: Final[int] = 1000
FEATURE_GPU_LOAD_MIN: Final[float] = 0.0
FEATURE_GPU_LOAD_MAX: Final[float] = 1.0
FEATURE_CACHE_HIT_PROB_MIN: Final[float] = 0.0
FEATURE_CACHE_HIT_PROB_MAX: Final[float] = 1.0

_FEATURE_CONTINUOUS: Final[int] = 6
_FEATURE_QUERY_CARDINALITY: Final[int] = len(FeatureQueryType)
_FEATURE_TENANT_CARDINALITY: Final[int] = len(FeatureTenantTier)

FEATURE_DIM: Final[int] = (
    _FEATURE_CONTINUOUS + _FEATURE_QUERY_CARDINALITY + _FEATURE_TENANT_CARDINALITY
)

assert (
    FEATURE_DIM == 15
), "FEATURE_DIM must match Section 3 encoding; update bandit if enums change."

DEFAULT_LINUCB_ALPHA_STR: Final[str] = "1.0"
DEFAULT_LINUCB_EXPLORATION_EPSILON_STR: Final[str] = "0.1"

# Default tenant policy when no DB row exists (scalar reward magnitudes only).
DEFAULT_TENANT_POLICY: Final[TenantPolicy] = TenantPolicy(
    coeff_latency=0.001,
    coeff_cost=1.0,
    coeff_slo_violation=10.0,
    coeff_fallback=5.0,
    coeff_retry=2.0,
    coeff_cache_hit=0.5,
)


# --- Latency SLO defaults when latency_slo_ms is omitted (named constants) ---

LATENCY_SLO_DEFAULT_MS_FREE: Final[int] = 2000
LATENCY_SLO_DEFAULT_MS_PRO: Final[int] = 1000
LATENCY_SLO_DEFAULT_MS_ENTERPRISE: Final[int] = 500

# --- Token budget validation (Section 3 output cap class inputs) ---

TOKEN_BUDGET_MIN_VALID: Final[int] = 1
TOKEN_BUDGET_MAX_VALID_INCLUSIVE: Final[int] = 32768

# --- Performance contracts (microseconds / seconds) ---

FEATURE_EXTRACTION_TARGET_MS: Final[int] = 5
FEATURE_EXTRACTION_TARGET_US: Final[int] = FEATURE_EXTRACTION_TARGET_MS * 1000
CACHE_HIT_REDIS_TIMEOUT_SECONDS: Final[float] = 0.001
MICROSECONDS_PER_SECOND: Final[int] = 1_000_000
NANOSECONDS_PER_MICROSECOND: Final[int] = 1000

# --- tiktoken ---

TIKTOKEN_ENCODING_CL100K_BASE: Final[str] = "cl100k_base"

# --- Static cache-hit prior when Redis Bloom is skipped (Section 3 lookup table) ---

_FQT = FeatureQueryType
_FTT = FeatureTenantTier

CACHE_HIT_PROB_FALLBACK_TABLE: Final[
    dict[tuple[FeatureQueryType, FeatureTenantTier], float]
] = {
    (_FQT.RAG, _FTT.FREE): 0.08,
    (_FQT.RAG, _FTT.PRO): 0.12,
    (_FQT.RAG, _FTT.ENTERPRISE): 0.18,
    (_FQT.RAG, _FTT.INTERNAL): 0.0,
    (_FQT.CODE, _FTT.FREE): 0.05,
    (_FQT.CODE, _FTT.PRO): 0.09,
    (_FQT.CODE, _FTT.ENTERPRISE): 0.14,
    (_FQT.CODE, _FTT.INTERNAL): 0.0,
    (_FQT.CHAT, _FTT.FREE): 0.10,
    (_FQT.CHAT, _FTT.PRO): 0.15,
    (_FQT.CHAT, _FTT.ENTERPRISE): 0.20,
    (_FQT.CHAT, _FTT.INTERNAL): 0.0,
    (_FQT.SUMMARIZE, _FTT.FREE): 0.06,
    (_FQT.SUMMARIZE, _FTT.PRO): 0.11,
    (_FQT.SUMMARIZE, _FTT.ENTERPRISE): 0.16,
    (_FQT.SUMMARIZE, _FTT.INTERNAL): 0.0,
    (_FQT.UNKNOWN, _FTT.FREE): 0.04,
    (_FQT.UNKNOWN, _FTT.PRO): 0.07,
    (_FQT.UNKNOWN, _FTT.ENTERPRISE): 0.10,
    (_FQT.UNKNOWN, _FTT.INTERNAL): 0.0,
}

__all__ = [
    "CACHE_HIT_PROB_FALLBACK_TABLE",
    "CACHE_HIT_REDIS_TIMEOUT_SECONDS",
    "DEFAULT_EXPLORATION_EPSILON",
    "DEFAULT_DECISION_FALLBACK_PATH",
    "DEFAULT_LINUCB_ALPHA_STR",
    "DEFAULT_LINUCB_EXPLORATION_EPSILON_STR",
    "DEFAULT_TENANT_POLICY",
    "DecisionLoggerConfigKey",
    "FEATURE_CACHE_HIT_PROB_MAX",
    "FEATURE_CACHE_HIT_PROB_MIN",
    "FEATURE_DIM",
    "FEATURE_EXTRACTION_TARGET_MS",
    "FEATURE_EXTRACTION_TARGET_US",
    "FEATURE_GPU_LOAD_MAX",
    "FEATURE_GPU_LOAD_MIN",
    "FEATURE_LATENCY_SLO_MS_MAX",
    "FEATURE_LATENCY_SLO_MS_MIN",
    "FEATURE_QUERY_LEN_MAX",
    "FEATURE_QUERY_LEN_MIN",
    "FEATURE_QUEUE_DEPTH_MAX",
    "FEATURE_QUEUE_DEPTH_MIN",
    "FEATURE_TOKEN_BUDGET_MAX",
    "FEATURE_TOKEN_BUDGET_MIN",
    "FeatureQueryType",
    "FeatureTenantTier",
    "LATENCY_SLO_DEFAULT_MS_ENTERPRISE",
    "LATENCY_SLO_DEFAULT_MS_FREE",
    "LATENCY_SLO_DEFAULT_MS_PRO",
    "LinUCBConfigKey",
    "MICROSECONDS_PER_SECOND",
    "NANOSECONDS_PER_MICROSECOND",
    "PolicyConfigKey",
    "TIKTOKEN_ENCODING_CL100K_BASE",
    "TOKEN_BUDGET_MAX_VALID_INCLUSIVE",
    "TOKEN_BUDGET_MIN_VALID",
]
