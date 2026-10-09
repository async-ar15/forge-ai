"""Reward pipeline tests for synchronous policy feedback RPC calls."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.enums import ModelTier, OutputBudget, Precision, RetrievalMode
from forgeai.policy.bandit_actions import ActionSpec
from forgeai.policy.decision_log_record import DecisionLog
from forgeai.policy.reward import RequestOutcome
from forgeai.policy.reward_pipeline import RewardPipeline
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import common_pb2, policy_service_pb2


def _decision_log() -> DecisionLog:
    action = ActionSpec(
        model_tier=ModelTier.SMALL,
        precision=Precision.INT8,
        retrieval_mode=RetrievalMode.CACHE_ONLY,
        output_budget=OutputBudget.SHORT,
    )
    return DecisionLog(
        request_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        policy_version="pol-a",
        deployment_version="dep-a",
        tenant_id=uuid.uuid4(),
        query_len=12,
        token_budget=128,
        query_type="rag",
        tenant_tier="pro",
        latency_slo_ms=500,
        queue_depth=2,
        gpu_load=0.3,
        cache_hit_prob=0.7,
        state_vector={
            "query_len": 12,
            "token_budget": 128,
            "query_type": "rag",
            "tenant_tier": "pro",
            "latency_slo_ms": 500,
            "queue_depth": 2,
            "gpu_load": 0.3,
            "cache_hit_prob": 0.7,
        },
        chosen_action=action,
        model_tier=action.model_tier.value,
        precision=action.precision.value,
        retrieval_mode=action.retrieval_mode.value,
        output_budget=action.output_budget.value,
        exploration_flag=False,
        exploration_type="none",
        greedy_action=action,
        action_scores={"small_int8_cache_only_short": 1.0},
        final_latency_ms=100,
        final_cost_usd=0.001,
        fallback_used=False,
        retry_count=0,
        cache_hit=True,
        slo_violation=False,
        r_online=0.0,
        r_components={},
    )


def _outcome() -> RequestOutcome:
    return RequestOutcome(
        final_latency_ms=100.0,
        final_cost_usd=0.001,
        slo_violated=False,
        fallback_used=False,
        retry_count=0,
        cache_hit=True,
        latency_slo_ms=500.0,
    )


@pytest.mark.asyncio
async def test_compute_and_record_calls_policy_feedback_rpc() -> None:
    logger = MagicMock()
    logger.update_online_reward = AsyncMock(return_value=None)
    policy_stub = MagicMock()
    policy_stub.RecordReward = AsyncMock(
        return_value=policy_service_pb2.RecordRewardResponse(
            acknowledged=True,
            policy_version="new-pol",
        )
    )
    pipeline = RewardPipeline(
        decision_logger=logger,
        tenant_policy=TenantPolicy(
            coeff_latency=0.01,
            coeff_cost=1.0,
            coeff_slo_violation=1.0,
            coeff_fallback=1.0,
            coeff_retry=1.0,
            coeff_cache_hit=0.1,
        ),
        policy_stub=policy_stub,
        kafka_producer=None,
    )

    reward = await pipeline.compute_and_record(_decision_log(), _outcome())

    assert isinstance(reward, float)
    logger.update_online_reward.assert_awaited_once()
    policy_stub.RecordReward.assert_awaited_once()
    call_req = policy_stub.RecordReward.await_args.args[0]
    assert isinstance(call_req, policy_service_pb2.RecordRewardRequest)
    assert call_req.action.model_tier == common_pb2.MODEL_TIER_SMALL
    assert call_req.feature_vector.query_type == common_pb2.QUERY_TYPE_RAG
    assert call_req.feature_vector.tenant_tier == common_pb2.TENANT_TIER_PRO
