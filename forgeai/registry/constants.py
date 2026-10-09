"""Owns: registry domain enumerations used by ORM models and Alembic CHECK constraints.

Does not own: protobuf enums, client-supplied query_type strings before
validation, or feature flags.
"""

from __future__ import annotations

from enum import StrEnum


class ArtifactStatus(StrEnum):
    """High-level lifecycle states for registry artifacts (operational extension)."""

    PENDING = "pending"
    AVAILABLE = "available"


class ModelProvider(StrEnum):
    """Provider values for models.provider (Section 4)."""

    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    OPENAI = "openai"


class QuantMethod(StrEnum):
    """Quantization method values for quant_profiles.method (Section 4)."""

    AWQ = "awq"
    GPTQ = "gptq"
    BNB = "bnb"
    SLIDER_QUANT = "slider_quant"


class GlobalPrecision(StrEnum):
    """Precision values for quant_profiles.global_precision and routing actions.

    (Section 4.)
    """

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class EvalType(StrEnum):
    """eval_runs.eval_type allowed values (Section 4)."""

    RUBRIC = "rubric"
    TASK_SUCCESS = "task_success"
    LATENCY = "latency"
    COST = "cost"


class DeploymentEnvironment(StrEnum):
    """deployments.environment values (Section 4)."""

    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(StrEnum):
    """deployments.status values (Section 4)."""

    PENDING = "pending"
    ACTIVE = "active"
    DRAINING = "draining"
    RETIRED = "retired"


class TrainingRunStatus(StrEnum):
    """training_runs.status placeholder lifecycle (registry extension)."""

    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryType(StrEnum):
    """routing_decisions.query_type values (Section 4)."""

    RAG = "rag"
    CODE = "code"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    UNKNOWN = "unknown"


class TenantTier(StrEnum):
    """routing_decisions.tenant_tier values (Section 4)."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class ModelTier(StrEnum):
    """routing_decisions.model_tier action values (Section 4)."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RetrievalMode(StrEnum):
    """routing_decisions.retrieval_mode action values (Section 4)."""

    OFF = "off"
    CACHE_ONLY = "cache_only"
    FULL = "full"


class OutputBudget(StrEnum):
    """routing_decisions.output_budget action values (Section 4)."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ExplorationType(StrEnum):
    """routing_decisions.exploration_type values when not NULL (Section 4)."""

    EPSILON = "epsilon"
    UCB = "ucb"
    FORCED = "forced"


class LabelSource(StrEnum):
    """routing_decisions.label_source values when not NULL (Section 4)."""

    JUDGE = "judge"
    HUMAN = "human"
    AUTO = "auto"


def sql_in_values(members: type[StrEnum]) -> tuple[str, ...]:
    """Return SQL IN tuple literals for CHECK constraints from a StrEnum."""

    return tuple(m.value for m in members)


__all__ = [
    "ArtifactStatus",
    "DeploymentEnvironment",
    "DeploymentStatus",
    "EvalType",
    "TrainingRunStatus",
    "ExplorationType",
    "GlobalPrecision",
    "LabelSource",
    "ModelProvider",
    "ModelTier",
    "OutputBudget",
    "QuantMethod",
    "QueryType",
    "RetrievalMode",
    "TenantTier",
    "sql_in_values",
]
