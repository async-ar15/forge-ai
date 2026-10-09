"""JSON-safe views of ``DecisionLog`` for CRITICAL logs and JSONL fallback.

Does not own: database sessions or business validation.
"""

from __future__ import annotations

import json
from typing import Any

from forgeai.policy.decision_log_orm import action_spec_to_json_mapping
from forgeai.policy.decision_log_record import DecisionLog


def decision_log_to_jsonable(decision: DecisionLog) -> dict[str, Any]:
    """Recursive dict suitable for ``json.dumps`` (UUID/datetime as strings)."""

    return {
        "request_id": str(decision.request_id),
        "timestamp": decision.timestamp.isoformat(),
        "policy_version": decision.policy_version,
        "deployment_version": decision.deployment_version,
        "tenant_id": str(decision.tenant_id),
        "query_len": decision.query_len,
        "token_budget": decision.token_budget,
        "query_type": decision.query_type,
        "tenant_tier": decision.tenant_tier,
        "latency_slo_ms": decision.latency_slo_ms,
        "queue_depth": decision.queue_depth,
        "gpu_load": decision.gpu_load,
        "cache_hit_prob": decision.cache_hit_prob,
        "state_vector": dict(decision.state_vector),
        "chosen_action": action_spec_to_json_mapping(decision.chosen_action),
        "model_tier": decision.model_tier,
        "precision": decision.precision,
        "retrieval_mode": decision.retrieval_mode,
        "output_budget": decision.output_budget,
        "exploration_flag": decision.exploration_flag,
        "exploration_type": decision.exploration_type,
        "greedy_action": action_spec_to_json_mapping(decision.greedy_action),
        "action_scores": dict(decision.action_scores),
        "final_latency_ms": decision.final_latency_ms,
        "final_cost_usd": decision.final_cost_usd,
        "fallback_used": decision.fallback_used,
        "retry_count": decision.retry_count,
        "cache_hit": decision.cache_hit,
        "slo_violation": decision.slo_violation,
        "r_online": decision.r_online,
        "r_components": dict(decision.r_components),
    }


def decision_log_json_dumps(decision: DecisionLog) -> str:
    """Serialize ``DecisionLog`` to a compact JSON string."""

    return json.dumps(decision_log_to_jsonable(decision), sort_keys=True)


__all__ = ["decision_log_json_dumps", "decision_log_to_jsonable"]
