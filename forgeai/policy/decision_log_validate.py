"""Structural invariants for ``DecisionLog`` before persistence (Section 4).

Does not own: ORM mapping, Prometheus, or filesystem fallback.
"""

from __future__ import annotations

from forgeai.policy.bandit_actions import ACTION_SPACE_SIZE
from forgeai.policy.decision_log_record import (
    R_COMPONENT_FIELD_NAMES,
    STATE_VECTOR_FIELD_NAMES,
    DecisionLog,
)


def list_consistency_violations(decision: DecisionLog) -> list[str]:
    """Return human-readable violation codes; empty means structurally sound."""

    out: list[str] = []
    greedy_mismatch = decision.greedy_action != decision.chosen_action
    if not decision.exploration_flag and greedy_mismatch:
        out.append("greedy_ne_chosen_when_exploration_flag_false")
    if _action_score_cardinality_invalid(decision):
        out.append("action_scores_cardinality_not_81")
    if frozenset(decision.r_components.keys()) != R_COMPONENT_FIELD_NAMES:
        out.append("r_components_keys_mismatch")
    if frozenset(decision.state_vector.keys()) != STATE_VECTOR_FIELD_NAMES:
        out.append("state_vector_keys_mismatch")
    if decision.chosen_action.model_tier.value != decision.model_tier:
        out.append("chosen_action_model_tier_column_mismatch")
    if decision.chosen_action.precision.value != decision.precision:
        out.append("chosen_action_precision_column_mismatch")
    if decision.chosen_action.retrieval_mode.value != decision.retrieval_mode:
        out.append("chosen_action_retrieval_mode_column_mismatch")
    if decision.chosen_action.output_budget.value != decision.output_budget:
        out.append("chosen_action_output_budget_column_mismatch")
    return out


def _action_score_cardinality_invalid(decision: DecisionLog) -> bool:
    if len(decision.action_scores) == ACTION_SPACE_SIZE:
        return False
    if _is_internal_forced_system_decision(decision):
        return len(decision.action_scores) != 0
    return True


def _is_internal_forced_system_decision(decision: DecisionLog) -> bool:
    return (
        decision.tenant_tier == "internal"
        and decision.exploration_type == "forced"
        and decision.exploration_flag is False
    )


__all__ = ["list_consistency_violations"]
