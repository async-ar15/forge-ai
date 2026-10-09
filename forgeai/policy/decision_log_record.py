"""Immutable phase-1 routing decision payload for ``routing_decisions`` (Section 4).

Does not own: SQLAlchemy sessions, Prometheus, or JSONL fallback I/O.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from forgeai.policy.bandit_actions import ActionSpec

# Keys produced by ``FeatureVector.to_log_dict()`` — must stay aligned with that method.
STATE_VECTOR_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "query_len",
        "token_budget",
        "query_type",
        "tenant_tier",
        "latency_slo_ms",
        "queue_depth",
        "gpu_load",
        "cache_hit_prob",
    }
)

# Per-term online reward decomposition written to ``r_components`` JSONB.
R_COMPONENT_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "latency",
        "cost",
        "slo_violation",
        "fallback",
        "retry",
        "cache_hit",
    }
)


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """Append-only snapshot for one routed request; every field is phase-1 mandatory.

    ``exploration_type`` uses wire strings: ``none``, ``epsilon``, ``ucb``, ``forced``.
    ``action_scores`` keys are canonical arm keys from ``bandit_actions``.
    ``state_vector`` must equal ``FeatureVector.to_log_dict()`` shape (8 keys).
    """

    request_id: uuid.UUID
    timestamp: datetime
    policy_version: str
    deployment_version: str
    tenant_id: uuid.UUID
    query_len: int
    token_budget: int
    query_type: str
    tenant_tier: str
    latency_slo_ms: int
    queue_depth: int
    gpu_load: float
    cache_hit_prob: float
    state_vector: dict[str, Any]
    chosen_action: ActionSpec
    model_tier: str
    precision: str
    retrieval_mode: str
    output_budget: str
    exploration_flag: bool
    exploration_type: str
    greedy_action: ActionSpec
    action_scores: dict[str, float]
    final_latency_ms: int
    final_cost_usd: float
    fallback_used: bool
    retry_count: int
    cache_hit: bool
    slo_violation: bool
    r_online: float
    r_components: dict[str, float]
    requested_model_hint: str | None = None
    endpoint: str | None = None


def decision_log_from_bandit_outcome(
    *,
    request_id: uuid.UUID,
    timestamp: datetime,
    policy_version: str,
    deployment_version: str,
    tenant_id: uuid.UUID,
    state_vector: dict[str, Any],
    chosen_action: ActionSpec,
    greedy_action: ActionSpec,
    exploration_flag: bool,
    exploration_type_wire: str,
    action_scores: dict[str, float],
    final_latency_ms: int,
    final_cost_usd: float,
    fallback_used: bool,
    retry_count: int,
    cache_hit: bool,
    slo_violation: bool,
    r_online: float,
    r_components: dict[str, float],
    requested_model_hint: str | None = None,
    endpoint: str | None = None,
) -> DecisionLog:
    """Build ``DecisionLog``; denormalized action columns match ``chosen_action``."""

    return DecisionLog(
        request_id=request_id,
        timestamp=timestamp,
        policy_version=policy_version,
        deployment_version=deployment_version,
        tenant_id=tenant_id,
        query_len=int(state_vector["query_len"]),
        token_budget=int(state_vector["token_budget"]),
        query_type=str(state_vector["query_type"]),
        tenant_tier=str(state_vector["tenant_tier"]),
        latency_slo_ms=int(state_vector["latency_slo_ms"]),
        queue_depth=int(state_vector["queue_depth"]),
        gpu_load=float(state_vector["gpu_load"]),
        cache_hit_prob=float(state_vector["cache_hit_prob"]),
        state_vector=dict(state_vector),
        chosen_action=chosen_action,
        model_tier=chosen_action.model_tier.value,
        precision=chosen_action.precision.value,
        retrieval_mode=chosen_action.retrieval_mode.value,
        output_budget=chosen_action.output_budget.value,
        exploration_flag=exploration_flag,
        exploration_type=exploration_type_wire,
        greedy_action=greedy_action,
        action_scores=dict(action_scores),
        final_latency_ms=final_latency_ms,
        final_cost_usd=final_cost_usd,
        fallback_used=fallback_used,
        retry_count=retry_count,
        cache_hit=cache_hit,
        slo_violation=slo_violation,
        r_online=r_online,
        r_components=dict(r_components),
        requested_model_hint=requested_model_hint,
        endpoint=endpoint,
    )


__all__ = [
    "DecisionLog",
    "R_COMPONENT_FIELD_NAMES",
    "STATE_VECTOR_FIELD_NAMES",
    "decision_log_from_bandit_outcome",
]
