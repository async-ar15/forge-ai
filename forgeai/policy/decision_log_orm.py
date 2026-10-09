"""Pure mapping ``DecisionLog`` → ``RoutingDecision`` ORM instance (Section 4).

Does not own: transactions, retries, or JSONL durability.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final

from forgeai.policy.bandit_actions import ActionSpec
from forgeai.policy.decision_log_record import DecisionLog
from forgeai.registry.models.routing_decision import RoutingDecision

_EXPLORATION_NONE_TOKENS: Final[frozenset[str]] = frozenset(("", "none"))


def action_spec_to_json_mapping(spec: ActionSpec) -> dict[str, str]:
    """JSONB payload for ``greedy_action`` / nested action mirrors."""

    return {
        "model_tier": spec.model_tier.value,
        "precision": spec.precision.value,
        "retrieval_mode": spec.retrieval_mode.value,
        "output_budget": spec.output_budget.value,
    }


def exploration_type_for_orm(wire: str) -> str | None:
    """Map wire exploration label to nullable ``routing_decisions.exploration_type``."""

    if wire.strip().lower() in _EXPLORATION_NONE_TOKENS:
        return None
    return wire


def _assign_scalars(rd: RoutingDecision, decision: DecisionLog) -> None:
    """Copy scalar and JSONB columns from ``decision`` onto ``rd``."""

    rd.request_id = decision.request_id
    rd.timestamp = decision.timestamp
    rd.policy_version = decision.policy_version
    rd.deployment_version = decision.deployment_version
    rd.tenant_id = decision.tenant_id
    rd.query_len = decision.query_len
    rd.token_budget = decision.token_budget
    rd.query_type = decision.query_type
    rd.tenant_tier = decision.tenant_tier
    rd.latency_slo_ms = decision.latency_slo_ms
    rd.queue_depth = decision.queue_depth
    rd.gpu_load = decision.gpu_load
    rd.cache_hit_prob = decision.cache_hit_prob
    rd.state_vector = dict(decision.state_vector)
    rd.model_tier = decision.model_tier
    rd.precision = decision.precision
    rd.retrieval_mode = decision.retrieval_mode
    rd.output_budget = decision.output_budget
    rd.exploration_flag = decision.exploration_flag
    rd.exploration_type = exploration_type_for_orm(decision.exploration_type)
    rd.final_latency_ms = decision.final_latency_ms
    rd.final_cost_usd = Decimal(str(decision.final_cost_usd))
    rd.fallback_used = decision.fallback_used
    rd.retry_count = decision.retry_count
    rd.cache_hit = decision.cache_hit
    rd.slo_violation = decision.slo_violation
    rd.r_online = decision.r_online


def _assign_action_json(rd: RoutingDecision, decision: DecisionLog) -> None:
    """Attach greedy action, score map, and reward breakdown JSONB."""

    rd.greedy_action = dict(action_spec_to_json_mapping(decision.greedy_action))
    scores: dict[str, Any] = {k: float(v) for k, v in decision.action_scores.items()}
    rd.action_scores = scores
    rd.r_components = {k: float(v) for k, v in decision.r_components.items()}
    rd.requested_model_hint = decision.requested_model_hint
    rd.endpoint = decision.endpoint


def build_orm_record(decision: DecisionLog) -> RoutingDecision:
    """Materialize a detached ``RoutingDecision`` row from a frozen ``DecisionLog``."""

    rd = RoutingDecision()
    _assign_scalars(rd, decision)
    _assign_action_json(rd, decision)
    return rd


__all__ = [
    "action_spec_to_json_mapping",
    "build_orm_record",
    "exploration_type_for_orm",
]
