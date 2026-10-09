"""Async retraining trigger based on accumulated offline labels."""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from sqlalchemy.engine import Engine

from forgeai.enums import ModelTier, OutputBudget, Precision, RetrievalMode
from forgeai.eval.constants import (
    RETRAINING_LABEL_THRESHOLD,
    RETRAINING_REWARD_BLEND_ALPHA,
)
from forgeai.kafka.constants import TOPIC_EVAL_LABELS
from forgeai.kafka.consumer import ForgeKafkaConsumer
from forgeai.observability.eval_queries import get_retraining_rows
from forgeai.observability.metrics import POLICY_RETRAINING_TOTAL
from forgeai.policy.bandit import LinUCBBandit
from forgeai.policy.bandit_actions import ActionSpec
from forgeai.policy.constants import FeatureQueryType, FeatureTenantTier
from forgeai.policy.features import FeatureVector
from forgeai.proto import policy_service_pb2, policy_service_pb2_grpc

_LOG = logging.getLogger(__name__)


class RetrainingTrigger:
    """Trigger policy retraining when enough async labels accumulate."""

    __slots__ = (
        "_alpha",
        "_consumer",
        "_engine",
        "_labels_since_retrain",
        "_policy_stub",
        "_threshold",
    )

    def __init__(
        self,
        *,
        consumer: ForgeKafkaConsumer,
        engine: Engine,
        policy_stub: policy_service_pb2_grpc.PolicyServiceStub,
        threshold: int = RETRAINING_LABEL_THRESHOLD,
        alpha: float = RETRAINING_REWARD_BLEND_ALPHA,
    ) -> None:
        self._consumer = consumer
        self._engine = engine
        self._policy_stub = policy_stub
        self._threshold = int(threshold)
        self._alpha = float(alpha)
        self._labels_since_retrain = 0

    async def run(self) -> None:
        await self._consumer.consume(TOPIC_EVAL_LABELS, self._handle_eval_label)

    async def _handle_eval_label(self, _event: dict[str, object]) -> None:
        self._labels_since_retrain += 1
        if self._labels_since_retrain < self._threshold:
            return
        ok = await self._retrain_and_hot_swap()
        if ok:
            self._labels_since_retrain = 0

    async def _retrain_and_hot_swap(self) -> bool:
        rows = get_retraining_rows(self._engine)
        if rows.empty:
            return False
        bandit, stats = _fit_bandit(rows, alpha=self._alpha)
        blob = _serialize_bandit_bytes(bandit)
        req = policy_service_pb2.LoadPolicyRequest(serialized_policy_bytes=blob)
        call = self._policy_stub.LoadPolicy(req)
        if inspect.isawaitable(call):
            resp = await call
        else:
            resp = cast(policy_service_pb2.LoadPolicyResponse, call)
        if not bool(resp.acknowledged):
            return False
        POLICY_RETRAINING_TOTAL.inc()
        _LOG.info(
            "policy_retrained rows=%d combined_reward_mean=%.6f "
            "combined_reward_min=%.6f "
            "combined_reward_max=%.6f old_policy_version=%s new_policy_version=%s",
            int(stats["rows"]),
            float(stats["mean"]),
            float(stats["min"]),
            float(stats["max"]),
            "unknown",
            resp.new_policy_version,
        )
        return True


def _fit_bandit(rows: object, *, alpha: float) -> tuple[LinUCBBandit, dict[str, float]]:
    bandit = LinUCBBandit()
    rewards: list[float] = []
    rows_frame = rows
    records = rows_frame.to_dict(orient="records")  # type: ignore[attr-defined]
    for row in records:
        fv = _feature_vector_from_row(row)
        action = _action_from_row(row)
        r_online = float(row["r_online"])
        q_offline = float(row["q_offline"])
        # Blending fast online and slower offline labels reduces estimator variance.
        combined = alpha * r_online + (1.0 - alpha) * q_offline
        bandit.update(action, combined, fv)
        rewards.append(combined)
    return bandit, _reward_stats(rewards)


def _feature_vector_from_row(row: dict[str, object]) -> FeatureVector:
    state = _state_dict(row["state_vector"])
    return FeatureVector(
        query_len=_as_int(state["query_len"]),
        token_budget=_as_int(state["token_budget"]),
        query_type=FeatureQueryType(str(state["query_type"])),
        tenant_tier=FeatureTenantTier(str(state["tenant_tier"])),
        latency_slo_ms=_as_int(state["latency_slo_ms"]),
        queue_depth=_as_int(state["queue_depth"]),
        gpu_load=_as_float(state["gpu_load"]),
        cache_hit_prob=_as_float(state["cache_hit_prob"]),
    )


def _state_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        data = json.loads(value)
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise TypeError(f"unsupported state_vector type: {type(value)!r}")
    return {
        "query_len": data["query_len"],
        "token_budget": data["token_budget"],
        "query_type": data["query_type"],
        "tenant_tier": data["tenant_tier"],
        "latency_slo_ms": data["latency_slo_ms"],
        "queue_depth": data["queue_depth"],
        "gpu_load": data["gpu_load"],
        "cache_hit_prob": data["cache_hit_prob"],
    }


def _action_from_row(row: dict[str, object]) -> ActionSpec:
    return ActionSpec(
        model_tier=ModelTier(str(row["model_tier"])),
        precision=Precision(str(row["precision"])),
        retrieval_mode=RetrievalMode(str(row["retrieval_mode"])),
        output_budget=OutputBudget(str(row["output_budget"])),
    )


def _serialize_bandit_bytes(bandit: LinUCBBandit) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        bandit.save(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _reward_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"rows": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "rows": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to int")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to float")


__all__ = ["RetrainingTrigger"]
