"""Execution-layer tuning: sampling caps, billing rates, cache bounds (Sections 2/5).

Does not own: Postgres registry rows, S3 replication, or autoscaler setpoints.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Final

from forgeai.enums import ModelTier, Precision

# --- Output token caps from ActionSpec.output_budget ---
# (maps to vLLM SamplingParams.max_tokens)
# These are v1 defaults — per-request override is a v2 feature.
VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS: Final[int] = 256
VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS: Final[int] = 1024
VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS: Final[int] = 4096
VLLM_OUTPUT_BUDGET_DEFAULT_TEMPERATURE: Final[float] = 0.7

DEFAULT_OUTPUT_TOKEN_CAP: Final[str] = "8192"


class ExecutionConfigKey(StrEnum):
    """Environment keys for execution engine tuning."""

    OUTPUT_TOKEN_CAP = "EXECUTION_OUTPUT_TOKEN_CAP"
    MAX_LOADED_MODELS = "MAX_LOADED_MODELS"
    SERVING_ENVIRONMENT = "EXECUTION_SERVING_ENVIRONMENT"


# MAX_LOADED_MODELS: read at ``ModelLoader`` construction (tests may patch env first).

# --- Billing: microdollars charged per generated token (output side) ---
# Source: lab v1 internal transfer-price sheet (2026-Q1); not fetched dynamically in v1.
# Update these when model costs change — they are not fetched dynamically in v1.
_COST_TOKEN_MICRO_USD_SMALL_FP16: Final[int] = 8
_COST_TOKEN_MICRO_USD_SMALL_INT8: Final[int] = 5
_COST_TOKEN_MICRO_USD_SMALL_INT4: Final[int] = 3
_COST_TOKEN_MICRO_USD_MEDIUM_FP16: Final[int] = 14
_COST_TOKEN_MICRO_USD_MEDIUM_INT8: Final[int] = 9
_COST_TOKEN_MICRO_USD_MEDIUM_INT4: Final[int] = 6
_COST_TOKEN_MICRO_USD_LARGE_FP16: Final[int] = 22
_COST_TOKEN_MICRO_USD_LARGE_INT8: Final[int] = 15
_COST_TOKEN_MICRO_USD_LARGE_INT4: Final[int] = 10

COST_PER_OUTPUT_TOKEN_MICRODOLLARS: Final[dict[tuple[ModelTier, Precision], int]] = {
    (ModelTier.SMALL, Precision.FP16): _COST_TOKEN_MICRO_USD_SMALL_FP16,
    (ModelTier.SMALL, Precision.INT8): _COST_TOKEN_MICRO_USD_SMALL_INT8,
    (ModelTier.SMALL, Precision.INT4): _COST_TOKEN_MICRO_USD_SMALL_INT4,
    (ModelTier.MEDIUM, Precision.FP16): _COST_TOKEN_MICRO_USD_MEDIUM_FP16,
    (ModelTier.MEDIUM, Precision.INT8): _COST_TOKEN_MICRO_USD_MEDIUM_INT8,
    (ModelTier.MEDIUM, Precision.INT4): _COST_TOKEN_MICRO_USD_MEDIUM_INT4,
    (ModelTier.LARGE, Precision.FP16): _COST_TOKEN_MICRO_USD_LARGE_FP16,
    (ModelTier.LARGE, Precision.INT8): _COST_TOKEN_MICRO_USD_LARGE_INT8,
    (ModelTier.LARGE, Precision.INT4): _COST_TOKEN_MICRO_USD_LARGE_INT4,
}


def cost_per_output_token_microdollars(tier: ModelTier, precision: Precision) -> int:
    """Return the v1 static microdollar rate for one output token."""

    return COST_PER_OUTPUT_TOKEN_MICRODOLLARS[(tier, precision)]


def read_output_token_cap() -> int:
    """Parse the hard ceiling from the process environment."""

    raw = os.environ.get(
        ExecutionConfigKey.OUTPUT_TOKEN_CAP.value,
        DEFAULT_OUTPUT_TOKEN_CAP,
    )
    return int(raw)


__all__ = [
    "COST_PER_OUTPUT_TOKEN_MICRODOLLARS",
    "DEFAULT_OUTPUT_TOKEN_CAP",
    "ExecutionConfigKey",
    "VLLM_OUTPUT_BUDGET_DEFAULT_TEMPERATURE",
    "VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS",
    "VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS",
    "VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS",
    "cost_per_output_token_microdollars",
    "read_output_token_cap",
]
