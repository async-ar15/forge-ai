"""Online reward R_online for LinUCB updates (pure function, Section 3).

Does not own: Postgres ``tenant_policies`` loading, batch replay, or bandit algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.policy.tenant_policy import TenantPolicy


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    """Post-request signals fed into R_online (all scalars, no side effects)."""

    final_latency_ms: float
    final_cost_usd: float
    slo_violated: bool
    fallback_used: bool
    retry_count: int
    cache_hit: bool
    latency_slo_ms: float


def reward_components_from_outcome(
    outcome: RequestOutcome,
    policy: TenantPolicy,
) -> dict[str, float]:
    """Per-term breakdown aligned with ``compute_online_reward`` (six JSONB keys)."""

    lat = float(outcome.final_latency_ms)
    cost = float(outcome.final_cost_usd)
    slo = 1.0 if outcome.slo_violated else 0.0
    fb = 1.0 if outcome.fallback_used else 0.0
    retries = float(outcome.retry_count)
    hit = 1.0 if outcome.cache_hit else 0.0
    return {
        "latency": -policy.coeff_latency * lat,
        "cost": -policy.coeff_cost * cost,
        "slo_violation": -policy.coeff_slo_violation * slo,
        "fallback": -policy.coeff_fallback * fb,
        "retry": -policy.coeff_retry * retries,
        "cache_hit": policy.coeff_cache_hit * hit,
    }


def compute_online_reward(outcome: RequestOutcome, policy: TenantPolicy) -> float:
    """Return R_online = -a·lat - b·cost - c·I_slo - d·I_fb - e·retries + f·I_hit.

    Pure: no logging, no I/O, no globals. O(1) time and memory.
    Coefficient names mirror ``tenant_policies`` columns on ``TenantPolicy``.
    """

    lat = float(outcome.final_latency_ms)
    cost = float(outcome.final_cost_usd)
    slo = 1.0 if outcome.slo_violated else 0.0
    fb = 1.0 if outcome.fallback_used else 0.0
    retries = float(outcome.retry_count)
    hit = 1.0 if outcome.cache_hit else 0.0
    return (
        -policy.coeff_latency * lat
        - policy.coeff_cost * cost
        - policy.coeff_slo_violation * slo
        - policy.coeff_fallback * fb
        - policy.coeff_retry * retries
        + policy.coeff_cache_hit * hit
    )


__all__ = ["RequestOutcome", "compute_online_reward", "reward_components_from_outcome"]
