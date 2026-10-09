"""Bandit metric helpers sourced from the shared observability module."""

from __future__ import annotations

from forgeai.observability.metrics import (
    BANDIT_UPDATES_TOTAL,
    POLICY_RPC_DURATION_SECONDS,
)


def record_bandit_update(*, action_key: str) -> None:
    """Increment updates for the arm key (low-cardinality: 81 fixed strings)."""

    BANDIT_UPDATES_TOTAL.labels(action_key=action_key).inc()


__all__ = [
    "BANDIT_UPDATES_TOTAL",
    "POLICY_RPC_DURATION_SECONDS",
    "record_bandit_update",
]
