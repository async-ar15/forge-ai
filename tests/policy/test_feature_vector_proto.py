"""Round-trip tests for ``FeatureVector`` and ``common_pb2.StateVector``."""

from __future__ import annotations

import pytest
from forgeai.policy.constants import FeatureQueryType, FeatureTenantTier
from forgeai.policy.features import FeatureVector
from forgeai.proto import common_pb2


def _sample_vector() -> FeatureVector:
    return FeatureVector(
        query_len=42,
        token_budget=2048,
        query_type=FeatureQueryType.SUMMARIZE,
        tenant_tier=FeatureTenantTier.ENTERPRISE,
        latency_slo_ms=750,
        queue_depth=3,
        gpu_load=0.8125,
        cache_hit_prob=0.375,
    )


def test_to_proto_populates_all_eight_fields() -> None:
    fv = _sample_vector()
    msg = fv.to_proto()
    assert msg.query_len == 42
    assert msg.token_budget == 2048
    assert msg.query_type == common_pb2.QUERY_TYPE_SUMMARIZE
    assert msg.tenant_tier == common_pb2.TENANT_TIER_ENTERPRISE
    assert msg.latency_slo_ms == 750
    assert msg.queue_depth == 3
    assert msg.gpu_load == pytest.approx(0.8125)
    assert msg.cache_hit_prob == pytest.approx(0.375)


def test_from_proto_populates_all_eight_fields() -> None:
    msg = common_pb2.StateVector(
        query_len=7,
        token_budget=512,
        query_type=common_pb2.QUERY_TYPE_CODE,
        tenant_tier=common_pb2.TENANT_TIER_PRO,
        latency_slo_ms=1200,
        queue_depth=11,
        gpu_load=0.25,
        cache_hit_prob=0.9,
    )
    fv = FeatureVector.from_proto(msg)
    assert fv.query_len == 7
    assert fv.token_budget == 512
    assert fv.query_type is FeatureQueryType.CODE
    assert fv.tenant_tier is FeatureTenantTier.PRO
    assert fv.latency_slo_ms == 1200
    assert fv.queue_depth == 11
    assert fv.gpu_load == pytest.approx(0.25)
    assert fv.cache_hit_prob == pytest.approx(0.9)


def test_round_trip_preserves_all_eight_fields() -> None:
    original = _sample_vector()
    restored = FeatureVector.from_proto(original.to_proto())
    assert restored == original


def test_from_proto_unspecified_query_maps_to_unknown() -> None:
    msg = common_pb2.StateVector(
        query_len=1,
        token_budget=1,
        query_type=common_pb2.QUERY_TYPE_UNSPECIFIED,
        tenant_tier=common_pb2.TENANT_TIER_FREE,
        latency_slo_ms=100,
        queue_depth=0,
        gpu_load=0.0,
        cache_hit_prob=0.0,
    )
    fv = FeatureVector.from_proto(msg)
    assert fv.query_type is FeatureQueryType.UNKNOWN


def test_from_proto_unspecified_tenant_maps_to_free() -> None:
    msg = common_pb2.StateVector(
        query_len=1,
        token_budget=1,
        query_type=common_pb2.QUERY_TYPE_CHAT,
        tenant_tier=common_pb2.TENANT_TIER_UNSPECIFIED,
        latency_slo_ms=100,
        queue_depth=0,
        gpu_load=0.0,
        cache_hit_prob=0.0,
    )
    fv = FeatureVector.from_proto(msg)
    assert fv.tenant_tier is FeatureTenantTier.FREE


def test_from_proto_unknown_enum_int_degrades() -> None:
    msg = common_pb2.StateVector()
    msg.query_len = 2
    msg.token_budget = 4
    msg.query_type = 99
    msg.tenant_tier = 99
    msg.latency_slo_ms = 50
    msg.queue_depth = 1
    msg.gpu_load = 0.1
    msg.cache_hit_prob = 0.2
    fv = FeatureVector.from_proto(msg)
    assert fv.query_type is FeatureQueryType.UNKNOWN
    assert fv.tenant_tier is FeatureTenantTier.FREE
