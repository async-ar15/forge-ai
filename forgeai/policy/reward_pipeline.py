"""Synchronous hot-path online reward pipeline for request completion."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Final, cast

from forgeai.kafka.constants import TOPIC_REWARDS
from forgeai.kafka.events import RewardEvent
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.observability.metrics import REQUEST_COST_USD, REWARD_VALUE
from forgeai.policy.bandit_actions import action_spec_to_proto
from forgeai.policy.constants import FeatureQueryType, FeatureTenantTier
from forgeai.policy.decision_log_record import DecisionLog
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.policy.features import FeatureVector
from forgeai.policy.reward import (
    RequestOutcome,
    compute_online_reward,
    reward_components_from_outcome,
)
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import policy_service_pb2, policy_service_pb2_grpc

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_HOT_PATH_BUDGET_MS: Final[float] = 5.0


@dataclass(slots=True)
class RewardPipeline:
    """Computes and records R_online in the synchronous response path.

    Performance contract: DB reward update, Kafka publish enqueue, bandit update,
    and reward histogram observation should complete in under 5ms on warm path.
    Failures are logged and swallowed to avoid blocking responses.
    """

    decision_logger: DecisionLogger
    tenant_policy: TenantPolicy
    policy_stub: policy_service_pb2_grpc.PolicyServiceStub | None
    kafka_producer: ForgeKafkaProducer | None

    async def compute_and_record(
        self,
        decision_log: DecisionLog,
        request_outcome: RequestOutcome,
    ) -> float:
        """Run reward compute and side effects; never raises to caller."""

        t0 = time.perf_counter()
        reward = compute_online_reward(request_outcome, self.tenant_policy)
        components = reward_components_from_outcome(request_outcome, self.tenant_policy)
        await self._update_db(decision_log, reward, components)
        self._publish_event(decision_log, reward, components)
        await self._record_policy_feedback(decision_log, reward)
        self._record_histogram(decision_log, reward)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > _HOT_PATH_BUDGET_MS:
            _LOG.warning(
                "reward_pipeline_slow request_id=%s elapsed_ms=%.3f budget_ms=%.1f",
                decision_log.request_id,
                elapsed_ms,
                _HOT_PATH_BUDGET_MS,
            )
        return reward

    async def _update_db(
        self,
        decision_log: DecisionLog,
        reward: float,
        components: dict[str, float],
    ) -> None:
        try:
            await self.decision_logger.update_online_reward(
                decision_log.request_id,
                r_online=reward,
                r_components=components,
            )
        except Exception:
            _LOG.error("reward_db_update_failed request_id=%s", decision_log.request_id)

    def _publish_event(
        self,
        decision_log: DecisionLog,
        reward: float,
        components: dict[str, float],
    ) -> None:
        if self.kafka_producer is None:
            return
        try:
            event = RewardEvent.from_decision_log(
                decision_log=decision_log,
                r_online=reward,
                r_components=components,
            )
            self.kafka_producer.send_event(TOPIC_REWARDS, event)
        except Exception:
            _LOG.error(
                "reward_event_publish_failed request_id=%s", decision_log.request_id
            )

    async def _record_policy_feedback(
        self, decision_log: DecisionLog, reward: float
    ) -> None:
        if self.policy_stub is None:
            _LOG.error(
                "reward_policy_feedback_failed request_id=%s", decision_log.request_id
            )
            return
        try:
            req = policy_service_pb2.RecordRewardRequest(
                request_id=str(decision_log.request_id),
                action=action_spec_to_proto(decision_log.chosen_action),
                r_online=float(reward),
                feature_vector=_feature_vector_from_state(
                    decision_log.state_vector
                ).to_proto(),
            )
            call = self.policy_stub.RecordReward(
                req,
                metadata=(("x-request-id", str(decision_log.request_id)),),
            )
            resp = await cast(Awaitable[policy_service_pb2.RecordRewardResponse], call)
            if not bool(resp.acknowledged):
                _LOG.error(
                    "reward_policy_feedback_nack request_id=%s policy_version=%s",
                    decision_log.request_id,
                    resp.policy_version,
                )
        except Exception:
            _LOG.error(
                "reward_policy_feedback_failed request_id=%s", decision_log.request_id
            )

    def _record_histogram(self, decision_log: DecisionLog, reward: float) -> None:
        try:
            REWARD_VALUE.labels(
                model_tier=decision_log.model_tier,
                precision=decision_log.precision,
                retrieval_mode=decision_log.retrieval_mode,
            ).observe(float(reward))
            REQUEST_COST_USD.labels(
                model_tier=decision_log.model_tier,
                precision=decision_log.precision,
                retrieval_mode=decision_log.retrieval_mode,
                cache_hit="true" if decision_log.cache_hit else "false",
            ).observe(float(decision_log.final_cost_usd))
        except Exception:
            _LOG.error(
                "reward_metric_record_failed request_id=%s", decision_log.request_id
            )


def _feature_vector_from_state(state: dict[str, object]) -> FeatureVector:
    q_raw = str(state.get("query_type", FeatureQueryType.UNKNOWN.value)).strip().lower()
    t_raw = str(state.get("tenant_tier", FeatureTenantTier.FREE.value)).strip().lower()
    q = (
        FeatureQueryType(q_raw)
        if q_raw in {e.value for e in FeatureQueryType}
        else FeatureQueryType.UNKNOWN
    )
    t = (
        FeatureTenantTier(t_raw)
        if t_raw in {e.value for e in FeatureTenantTier}
        else FeatureTenantTier.FREE
    )
    return FeatureVector(
        query_len=_to_int(state.get("query_len", 0)),
        token_budget=_to_int(state.get("token_budget", 1)),
        query_type=q,
        tenant_tier=t,
        latency_slo_ms=_to_int(state.get("latency_slo_ms", 1)),
        queue_depth=_to_int(state.get("queue_depth", 0)),
        gpu_load=_to_float(state.get("gpu_load", 0.0)),
        cache_hit_prob=_to_float(state.get("cache_hit_prob", 0.0)),
    )


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to int")


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to float")


__all__ = ["RewardPipeline"]
