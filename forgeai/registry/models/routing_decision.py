"""Owns: ORM mapping for routing_decisions (Section 4).

Does not own: bandit feature extraction, reward arithmetic,
or eval SQL view definitions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from forgeai.registry.constants import (
    ExplorationType,
    GlobalPrecision,
    LabelSource,
    ModelTier,
    OutputBudget,
    QueryType,
    RetrievalMode,
    TenantTier,
)
from forgeai.registry.models.base import Base

if TYPE_CHECKING:
    from forgeai.registry.models.tenant import Tenant


def _vals(enum_cls: type[StrEnum]) -> str:
    """Format IN list from a StrEnum class."""

    return ", ".join(repr(m.value) for m in enum_cls)


class RoutingDecision(Base):
    """Single routed request: append-only log with one permitted async follow-up.

    Insert once for the hot path; ``labeled_at`` is set only when offline labels
    (e.g. ``q_offline``) are written. No ``updated_at`` — arbitrary row mutation
    is out of contract.
    """

    __tablename__ = "routing_decisions"
    __table_args__ = (
        CheckConstraint(
            f"query_type IN ({_vals(QueryType)})",
            name="ck_routing_decisions_query_type",
        ),
        CheckConstraint(
            f"tenant_tier IN ({_vals(TenantTier)})",
            name="ck_routing_decisions_tenant_tier",
        ),
        CheckConstraint(
            f"model_tier IN ({_vals(ModelTier)})",
            name="ck_routing_decisions_model_tier",
        ),
        CheckConstraint(
            f'"precision" IN ({_vals(GlobalPrecision)})',
            name="ck_routing_decisions_precision",
        ),
        CheckConstraint(
            f"retrieval_mode IN ({_vals(RetrievalMode)})",
            name="ck_routing_decisions_retrieval_mode",
        ),
        CheckConstraint(
            f"output_budget IN ({_vals(OutputBudget)})",
            name="ck_routing_decisions_output_budget",
        ),
        CheckConstraint(
            "(exploration_type IS NULL) OR (exploration_type IN ("
            f"{_vals(ExplorationType)}))",
            name="ck_routing_decisions_exploration_type",
        ),
        CheckConstraint(
            f"(label_source IS NULL) OR (label_source IN ({_vals(LabelSource)}))",
            name="ck_routing_decisions_label_source",
        ),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    timestamp: Mapped[datetime] = mapped_column(
        "timestamp",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    policy_version: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    deployment_version: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    query_len: Mapped[int] = mapped_column(Integer, nullable=False)
    token_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    query_type: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_slo_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    queue_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_load: Mapped[float] = mapped_column(Float, nullable=False)
    cache_hit_prob: Mapped[float] = mapped_column(Float, nullable=False)
    state_vector: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    precision: Mapped[str] = mapped_column(String(16), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    output_budget: Mapped[str] = mapped_column(String(16), nullable=False)
    exploration_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    exploration_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    greedy_action: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    action_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    final_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    final_cost_usd: Mapped[Decimal] = mapped_column(Numeric(24, 12), nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    slo_violation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    r_online: Mapped[float] = mapped_column(Float, nullable=False)
    r_components: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    q_offline: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    requested_model_hint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    tenant: Mapped[Tenant] = relationship(
        "Tenant",
        back_populates="routing_decisions",
    )

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return (
            f"RoutingDecision(request_id={self.request_id!r}, "
            f"tenant_id={self.tenant_id!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar columns to JSON-friendly primitives."""

        return {
            "request_id": str(self.request_id),
            "timestamp": self.timestamp.isoformat(),
            "policy_version": self.policy_version,
            "deployment_version": self.deployment_version,
            "tenant_id": str(self.tenant_id),
            "query_len": self.query_len,
            "token_budget": self.token_budget,
            "query_type": self.query_type,
            "tenant_tier": self.tenant_tier,
            "latency_slo_ms": self.latency_slo_ms,
            "queue_depth": self.queue_depth,
            "gpu_load": self.gpu_load,
            "cache_hit_prob": self.cache_hit_prob,
            "state_vector": dict(self.state_vector),
            "model_tier": self.model_tier,
            "precision": self.precision,
            "retrieval_mode": self.retrieval_mode,
            "output_budget": self.output_budget,
            "exploration_flag": self.exploration_flag,
            "exploration_type": self.exploration_type,
            "greedy_action": dict(self.greedy_action),
            "action_scores": dict(self.action_scores),
            "final_latency_ms": self.final_latency_ms,
            "final_cost_usd": float(self.final_cost_usd),
            "fallback_used": self.fallback_used,
            "retry_count": self.retry_count,
            "cache_hit": self.cache_hit,
            "slo_violation": self.slo_violation,
            "r_online": self.r_online,
            "r_components": dict(self.r_components),
            "q_offline": self.q_offline,
            "judge_score": self.judge_score,
            "groundedness_score": self.groundedness_score,
            "label_source": self.label_source,
            "requested_model_hint": self.requested_model_hint,
            "endpoint": self.endpoint,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
        }


__all__ = ["RoutingDecision"]
