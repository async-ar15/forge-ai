"""vLLM ``SamplingParams`` construction from ``ActionSpec.output_budget``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forgeai.enums import OutputBudget
from forgeai.execution.constants import (
    VLLM_OUTPUT_BUDGET_DEFAULT_TEMPERATURE,
    VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS,
    VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS,
    VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS,
)


@dataclass(frozen=True, slots=True)
class ForgeSamplingParams:
    """Subset of vLLM sampling knobs used by ForgeAI v1."""

    max_tokens: int
    temperature: float

    def as_vllm_kwargs(self) -> dict[str, Any]:
        """Keyword args for ``vllm.SamplingParams`` when the dependency is present."""

        return {"max_tokens": self.max_tokens, "temperature": self.temperature}


def base_max_tokens_for_output_budget(budget: OutputBudget) -> int:
    """Map output budget class to v1 default max new tokens (documented caps)."""

    if budget is OutputBudget.SHORT:
        return VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS
    if budget is OutputBudget.MEDIUM:
        return VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS
    if budget is OutputBudget.LONG:
        return VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS
    return VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS


def build_sampling_params(
    output_budget: OutputBudget,
    *,
    max_output_token_limit: int,
) -> ForgeSamplingParams:
    """Clamp registry defaults by the per-request hard cap from the gateway."""

    base = base_max_tokens_for_output_budget(output_budget)
    capped = min(base, max(1, max_output_token_limit))
    return ForgeSamplingParams(
        max_tokens=capped,
        temperature=VLLM_OUTPUT_BUDGET_DEFAULT_TEMPERATURE,
    )


__all__ = [
    "ForgeSamplingParams",
    "base_max_tokens_for_output_budget",
    "build_sampling_params",
]
