"""A narrow façade over vLLM-oriented execution concerns for later wiring.

Does not own: ElephantBroker internals, retrieval prefetch, or streaming transports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from forgeai.enums import ModelTier, OutputBudget, Precision, RetrievalMode
from forgeai.execution.constants import DEFAULT_OUTPUT_TOKEN_CAP, ExecutionConfigKey


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Action parameters forwarded from the Policy Engine into execution."""

    model_tier: ModelTier
    precision: Precision
    retrieval_mode: RetrievalMode
    output_budget: OutputBudget


class ExecutionRuntime:
    """Placeholder runtime that validates specs against environment-derived caps."""

    def __init__(self) -> None:
        """Construct runtime state without eagerly loading model weights."""

        self._token_cap = _read_output_token_cap()

    def validate_spec(self, spec: ExecutionSpec) -> None:
        """Ensure the execution spec respects configured output budgets."""

        _ = spec
        if self._token_cap < 1:
            msg = "EXECUTION_OUTPUT_TOKEN_CAP must be a positive integer."
            raise ValueError(msg)


def _read_output_token_cap() -> int:
    """Parse the output token ceiling from the process environment."""

    raw = os.environ.get(
        ExecutionConfigKey.OUTPUT_TOKEN_CAP.value,
        DEFAULT_OUTPUT_TOKEN_CAP,
    )
    return int(raw)


__all__ = ["ExecutionRuntime", "ExecutionSpec"]
