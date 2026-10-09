"""Kafka event dataclasses with deterministic JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import UUID

from forgeai.kafka.constants import (
    TOPIC_DECISIONS,
    TOPIC_EVAL_LABELS,
    TOPIC_MODEL_LOADS,
    TOPIC_REGRESSIONS,
    TOPIC_REWARDS,
)
from forgeai.policy.decision_log_record import DecisionLog


def _iso8601_utc(value: datetime) -> str:
    dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _to_jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return _iso8601_utc(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        data = asdict(cast(Any, value))
        return {k: _to_jsonable(v) for k, v in data.items()}
    return value


@dataclass(frozen=True, slots=True)
class RewardEvent:
    request_id: UUID
    timestamp: datetime
    tenant_id: UUID
    model_tier: str
    precision: str
    retrieval_mode: str
    r_online: float
    r_components: dict[str, float]
    final_latency_ms: int
    final_cost_usd: Decimal
    slo_violation: bool
    cache_hit: bool
    exploration_flag: bool
    policy_version: str
    topic: str = TOPIC_REWARDS

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(asdict(self)), separators=(",", ":"))

    @classmethod
    def from_decision_log(
        cls,
        *,
        decision_log: DecisionLog,
        r_online: float,
        r_components: dict[str, float],
    ) -> RewardEvent:
        return cls(
            request_id=decision_log.request_id,
            timestamp=decision_log.timestamp,
            tenant_id=decision_log.tenant_id,
            model_tier=decision_log.model_tier,
            precision=decision_log.precision,
            retrieval_mode=decision_log.retrieval_mode,
            r_online=float(r_online),
            r_components={k: float(v) for k, v in r_components.items()},
            final_latency_ms=int(decision_log.final_latency_ms),
            final_cost_usd=Decimal(str(decision_log.final_cost_usd)),
            slo_violation=bool(decision_log.slo_violation),
            cache_hit=bool(decision_log.cache_hit),
            exploration_flag=bool(decision_log.exploration_flag),
            policy_version=str(decision_log.policy_version),
        )


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    request_id: UUID
    timestamp: datetime
    tenant_id: UUID
    chosen_action: dict[str, str]
    exploration_flag: bool
    exploration_type: str
    policy_version: str
    topic: str = TOPIC_DECISIONS

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(asdict(self)), separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ModelLoadEvent:
    request_id: UUID
    timestamp: datetime
    model_tier: str
    precision: str
    load_latency_ms: int
    memory_mb: int
    success: bool
    error_message: str | None
    topic: str = TOPIC_MODEL_LOADS

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(asdict(self)), separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class EvalLabelEvent:
    request_id: UUID
    timestamp: datetime
    q_offline: float
    judge_score: float
    groundedness_score: float
    label_source: str
    topic: str = TOPIC_EVAL_LABELS

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(asdict(self)), separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RegressionEvent:
    detected_at: datetime
    metric_name: str
    previous_value: float
    current_value: float
    delta: float
    policy_version_old: str
    policy_version_new: str
    severity: str
    topic: str = TOPIC_REGRESSIONS

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(asdict(self)), separators=(",", ":"))


__all__ = [
    "DecisionEvent",
    "EvalLabelEvent",
    "ModelLoadEvent",
    "RegressionEvent",
    "RewardEvent",
]
