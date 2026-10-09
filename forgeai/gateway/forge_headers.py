"""ForgeAI routing proof headers for infer and OpenAI-compatible endpoints."""

from __future__ import annotations

import uuid
from typing import Final

from forgeai.gateway.openai_constants import (
    OPENAI_HEADER_COST,
    OPENAI_HEADER_LATENCY,
    OPENAI_HEADER_MODEL,
    OPENAI_HEADER_POLICY_VERSION,
    OPENAI_HEADER_PRECISION,
    OPENAI_HEADER_REQUEST_ID,
    OPENAI_HEADER_RETRIEVAL,
)
from forgeai.proto import common_pb2, policy_service_pb2

_PREFIX_MODEL: Final[str] = "MODEL_TIER_"
_PREFIX_PRECISION: Final[str] = "PRECISION_"
_PREFIX_RETRIEVAL: Final[str] = "RETRIEVAL_MODE_"


def routing_headers(
    *,
    request_id: uuid.UUID,
    policy: policy_service_pb2.PolicyDecideResponse | None,
    cost_usd: float,
    latency_ms: int,
) -> dict[str, str]:
    model = _enum_value(
        (
            common_pb2.ModelTier.Name(policy.chosen_action.model_tier)
            if policy is not None
            else ""
        ),
        _PREFIX_MODEL,
    )
    precision = _enum_value(
        (
            common_pb2.Precision.Name(policy.chosen_action.precision)
            if policy is not None
            else ""
        ),
        _PREFIX_PRECISION,
    )
    retrieval = _enum_value(
        (
            common_pb2.RetrievalMode.Name(policy.chosen_action.retrieval_mode)
            if policy is not None
            else ""
        ),
        _PREFIX_RETRIEVAL,
    )
    version = policy.policy_version if policy is not None else "unknown"
    return {
        OPENAI_HEADER_MODEL: model,
        OPENAI_HEADER_PRECISION: precision,
        OPENAI_HEADER_RETRIEVAL: retrieval,
        OPENAI_HEADER_COST: f"{cost_usd:.6f}",
        OPENAI_HEADER_LATENCY: str(int(latency_ms)),
        OPENAI_HEADER_POLICY_VERSION: version,
        OPENAI_HEADER_REQUEST_ID: str(request_id),
    }


def _enum_value(raw: str, prefix: str) -> str:
    if raw == "":
        return "unknown"
    return raw.replace(prefix, "", 1).lower()
