"""Policy feedback RPC tests for synchronous online reward updates."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from forgeai.policy.bandit import LinUCBBandit
from forgeai.policy.bandit_actions import ALL_ACTIONS, action_spec_to_proto
from forgeai.policy.grpc_service import ForgePolicyServicer
from forgeai.proto import common_pb2, policy_service_pb2


def _state_vector() -> common_pb2.StateVector:
    return common_pb2.StateVector(
        query_len=12,
        token_budget=256,
        query_type=common_pb2.QUERY_TYPE_RAG,
        tenant_tier=common_pb2.TENANT_TIER_PRO,
        latency_slo_ms=500,
        queue_depth=1,
        gpu_load=0.2,
        cache_hit_prob=0.3,
    )


def test_record_reward_acknowledges_and_updates_policy_version() -> None:
    bandit = LinUCBBandit(
        ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(1)
    )
    servicer = ForgePolicyServicer(bandit)
    before = bandit.policy_version
    req = policy_service_pb2.RecordRewardRequest(
        request_id="5f53c5a8-31e9-4f9d-a4ac-f63170bbd001",
        action=action_spec_to_proto(ALL_ACTIONS[0]),
        r_online=1.25,
        feature_vector=_state_vector(),
    )

    resp = servicer.RecordReward(req, object())

    assert resp.acknowledged is True
    assert resp.policy_version == bandit.policy_version
    assert resp.policy_version != before


def test_load_policy_hot_swaps_bandit() -> None:
    bandit = LinUCBBandit(
        ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(2)
    )
    servicer = ForgePolicyServicer(bandit)
    trained = LinUCBBandit(
        ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(3)
    )
    for _ in range(5):
        trained.update(ALL_ACTIONS[1], 1.0, _state_vector_to_feature())
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        trained.save(path)
        req = policy_service_pb2.LoadPolicyRequest(
            serialized_policy_bytes=path.read_bytes(),
        )
    finally:
        path.unlink(missing_ok=True)
    resp = servicer.LoadPolicy(req, object())
    assert resp.acknowledged is True
    assert resp.new_policy_version == trained.policy_version


def _state_vector_to_feature():
    from forgeai.policy.constants import FeatureQueryType, FeatureTenantTier
    from forgeai.policy.features import FeatureVector

    return FeatureVector(
        query_len=12,
        token_budget=256,
        query_type=FeatureQueryType.RAG,
        tenant_tier=FeatureTenantTier.PRO,
        latency_slo_ms=500,
        queue_depth=1,
        gpu_load=0.2,
        cache_hit_prob=0.3,
    )
