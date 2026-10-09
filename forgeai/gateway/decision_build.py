"""Assemble ``DecisionLog`` rows after execution completes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from forgeai.gateway.exploration_wire import exploration_type_int_to_wire
from forgeai.policy.bandit_actions import action_spec_from_proto
from forgeai.policy.decision_log_record import (
    DecisionLog,
    decision_log_from_bandit_outcome,
)
from forgeai.policy.features import FeatureVector
from forgeai.policy.reward import (
    RequestOutcome,
    compute_online_reward,
    reward_components_from_outcome,
)
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import policy_service_pb2


def build_request_outcome(
    *,
    final_latency_ms: float,
    cost_microdollars: int | None,
    feature_vector: FeatureVector,
    fallback_used: bool,
) -> RequestOutcome:
    """Map execution telemetry into the reward helper input."""

    usd = float(cost_microdollars or 0) / 1_000_000.0
    slo = final_latency_ms > float(feature_vector.latency_slo_ms)
    cache_hit = feature_vector.cache_hit_prob >= 1.0
    return RequestOutcome(
        final_latency_ms=final_latency_ms,
        final_cost_usd=usd,
        slo_violated=slo,
        fallback_used=fallback_used,
        retry_count=0,
        cache_hit=cache_hit,
        latency_slo_ms=float(feature_vector.latency_slo_ms),
    )


def decision_from_infer_completion(
    *,
    request_id: uuid.UUID,
    tenant_id: uuid.UUID,
    deployment_version: str,
    features: FeatureVector,
    policy_resp: policy_service_pb2.PolicyDecideResponse,
    outcome: RequestOutcome,
    tenant_policy: TenantPolicy,
    requested_model_hint: str | None = None,
    endpoint: str = "/v1/infer",
) -> DecisionLog:
    """Return a fully populated ``DecisionLog``."""

    state_vector = features.to_log_dict()
    chosen = action_spec_from_proto(policy_resp.chosen_action)
    greedy = action_spec_from_proto(policy_resp.greedy_action)
    et = exploration_type_int_to_wire(int(policy_resp.exploration_type))
    r_online = compute_online_reward(outcome, tenant_policy)
    r_comp = reward_components_from_outcome(outcome, tenant_policy)
    return decision_log_from_bandit_outcome(
        request_id=request_id,
        timestamp=datetime.now(UTC),
        policy_version=policy_resp.policy_version,
        deployment_version=deployment_version,
        tenant_id=tenant_id,
        state_vector=state_vector,
        chosen_action=chosen,
        greedy_action=greedy,
        exploration_flag=bool(policy_resp.exploration_flag),
        exploration_type_wire=et,
        action_scores=dict(policy_resp.action_scores),
        final_latency_ms=int(outcome.final_latency_ms),
        final_cost_usd=outcome.final_cost_usd,
        fallback_used=outcome.fallback_used,
        retry_count=outcome.retry_count,
        cache_hit=outcome.cache_hit,
        slo_violation=outcome.slo_violated,
        r_online=r_online,
        r_components=r_comp,
        requested_model_hint=requested_model_hint,
        endpoint=endpoint,
    )


__all__ = ["build_request_outcome", "decision_from_infer_completion"]
