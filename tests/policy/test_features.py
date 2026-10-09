"""Tests for policy feature extraction (Section 3 state vector)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
import redis
import tiktoken
from forgeai.policy.constants import (
    CACHE_HIT_PROB_FALLBACK_TABLE,
    CACHE_HIT_REDIS_TIMEOUT_SECONDS,
    LATENCY_SLO_DEFAULT_MS_ENTERPRISE,
    LATENCY_SLO_DEFAULT_MS_FREE,
    LATENCY_SLO_DEFAULT_MS_PRO,
    FeatureQueryType,
    FeatureTenantTier,
)
from forgeai.policy.features import (
    FeatureExtractionInput,
    FeatureExtractor,
    FeatureVector,
    _cache_hit_prob_fallback,
    _count_query_tokens,
    _normalize_query_type,
    _normalize_tenant_tier,
    _resolve_latency_slo_ms,
    _validate_token_budget,
)
from forgeai.policy.probes import LocalGpuLoadProbe, LocalQueueDepthProbe


@pytest.fixture
def logger() -> logging.Logger:
    log = logging.getLogger("test_policy_features")
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def fallback_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    recorded: list[tuple[str, str]] = []

    def _spy(*, feature_name: str, reason: str) -> None:
        recorded.append((feature_name, reason))

    monkeypatch.setattr(
        "forgeai.policy.features.record_feature_extraction_fallback",
        _spy,
    )
    return recorded


def _make_extractor(
    *,
    logger: logging.Logger,
    redis_client: Any,
    queue: LocalQueueDepthProbe | None = None,
    gpu: LocalGpuLoadProbe | None = None,
    bloom_key: str = "test:bloom",
) -> FeatureExtractor:
    return FeatureExtractor(
        queue_depth_probe=queue or LocalQueueDepthProbe(0),
        gpu_load_probe=gpu or LocalGpuLoadProbe(0.0),
        redis_client=redis_client,
        redis_bloom_key=bloom_key,
        logger=logger,
    )


def _base_input(**overrides: Any) -> FeatureExtractionInput:
    base = dict(
        query_text="hello world",
        token_budget=128,
        query_type_raw="chat",
        latency_slo_ms_raw=500,
        tenant_tier_raw="pro",
    )
    base.update(overrides)
    return FeatureExtractionInput(**base)


@pytest.mark.asyncio
async def test_query_len_matches_tiktoken(logger: logging.Logger) -> None:
    text = "The quick brown fox."
    enc = tiktoken.get_encoding("cl100k_base")
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(
        _base_input(query_text=text, latency_slo_ms_raw=100, tenant_tier_raw="free"),
    )
    assert fv.query_len == len(enc.encode(text))
    assert _count_query_tokens(text) == len(enc.encode(text))


@pytest.mark.asyncio
async def test_token_budget_valid_passes(logger: logging.Logger) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(_base_input(token_budget=32768))
    assert fv.token_budget == 32768


@pytest.mark.parametrize("bad_budget", [0, -1, 32769])
@pytest.mark.asyncio
async def test_token_budget_invalid_raises(
    bad_budget: int,
    logger: logging.Logger,
) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    with pytest.raises(ValueError, match="token_budget"):
        await ext.extract(_base_input(token_budget=bad_budget))


def test_validate_token_budget_edge_cases() -> None:
    assert _validate_token_budget(1) == 1
    assert _validate_token_budget(32768) == 32768
    with pytest.raises(ValueError):
        _validate_token_budget(0)


@pytest.mark.asyncio
async def test_query_type_unknown_degrades(
    caplog: pytest.LogCaptureFixture,
    logger: logging.Logger,
) -> None:
    caplog.set_level(logging.WARNING)
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(_base_input(query_type_raw="not-a-real-type"))
    assert fv.query_type is FeatureQueryType.UNKNOWN
    assert any("unrecognized query_type" in r.message for r in caplog.records)


def test_normalize_query_type_none_warns(
    caplog: pytest.LogCaptureFixture,
    logger: logging.Logger,
) -> None:
    caplog.set_level(logging.WARNING)
    qt = _normalize_query_type(None, logger)
    assert qt is FeatureQueryType.UNKNOWN
    assert any("absent" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_tenant_tier_defaults_free_with_warning(
    caplog: pytest.LogCaptureFixture,
    logger: logging.Logger,
) -> None:
    caplog.set_level(logging.WARNING)
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(_base_input(tenant_tier_raw=None))
    assert fv.tenant_tier is FeatureTenantTier.FREE
    assert any("tenant_tier absent" in r.message for r in caplog.records)


def test_normalize_tenant_tier_invalid_warns(
    caplog: pytest.LogCaptureFixture,
    logger: logging.Logger,
) -> None:
    caplog.set_level(logging.WARNING)
    tier = _normalize_tenant_tier("platinum", logger)
    assert tier is FeatureTenantTier.FREE


@pytest.mark.parametrize(
    ("tier_raw", "expected_ms"),
    [
        ("free", LATENCY_SLO_DEFAULT_MS_FREE),
        ("pro", LATENCY_SLO_DEFAULT_MS_PRO),
        ("enterprise", LATENCY_SLO_DEFAULT_MS_ENTERPRISE),
    ],
)
@pytest.mark.asyncio
async def test_latency_slo_defaults_by_tier(
    tier_raw: str,
    expected_ms: int,
    logger: logging.Logger,
) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(
        _base_input(latency_slo_ms_raw=None, tenant_tier_raw=tier_raw),
    )
    assert fv.latency_slo_ms == expected_ms


@pytest.mark.parametrize(
    "tier",
    [FeatureTenantTier.FREE, FeatureTenantTier.PRO, FeatureTenantTier.ENTERPRISE],
)
def test_resolve_latency_slo_ms_defaults(tier: FeatureTenantTier) -> None:
    got = _resolve_latency_slo_ms(None, tier)
    if tier is FeatureTenantTier.FREE:
        assert got == LATENCY_SLO_DEFAULT_MS_FREE
    elif tier is FeatureTenantTier.PRO:
        assert got == LATENCY_SLO_DEFAULT_MS_PRO
    else:
        assert got == LATENCY_SLO_DEFAULT_MS_ENTERPRISE


@pytest.mark.asyncio
async def test_latency_slo_explicit_positive(logger: logging.Logger) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(_base_input(latency_slo_ms_raw=750))
    assert fv.latency_slo_ms == 750


@pytest.mark.asyncio
async def test_latency_slo_non_positive_raises(logger: logging.Logger) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    with pytest.raises(ValueError, match="latency_slo_ms"):
        await ext.extract(_base_input(latency_slo_ms_raw=0))


@pytest.mark.asyncio
async def test_queue_depth_probe_isolation(logger: logging.Logger) -> None:
    probe = LocalQueueDepthProbe(7)
    ext = _make_extractor(logger=logger, redis_client=None, queue=probe)
    fv = await ext.extract(_base_input())
    assert fv.queue_depth == 7


@pytest.mark.asyncio
async def test_gpu_load_probe_isolation(logger: logging.Logger) -> None:
    probe = LocalGpuLoadProbe(0.42)
    ext = _make_extractor(logger=logger, redis_client=None, gpu=probe)
    fv = await ext.extract(_base_input())
    assert fv.gpu_load == 0.42


@pytest.mark.asyncio
async def test_cache_hit_prob_redis_hit(
    logger: logging.Logger,
    fallback_spy: list[tuple[str, str]],
) -> None:
    class _FakeRedis:
        async def execute_command(self, *_a: Any, **_kw: Any) -> int:
            return 1

    ext = _make_extractor(logger=logger, redis_client=_FakeRedis())
    fv = await ext.extract(_base_input())
    assert fv.cache_hit_prob == 1.0
    assert fallback_spy == []


@pytest.mark.asyncio
async def test_cache_hit_prob_redis_miss(
    logger: logging.Logger,
    fallback_spy: list[tuple[str, str]],
) -> None:
    class _FakeRedis:
        async def execute_command(self, *_a: Any, **_kw: Any) -> int:
            return 0

    ext = _make_extractor(logger=logger, redis_client=_FakeRedis())
    fv = await ext.extract(_base_input())
    assert fv.cache_hit_prob == 0.0
    assert fallback_spy == []


@pytest.mark.asyncio
async def test_cache_hit_prob_connection_error_fallback(
    logger: logging.Logger,
    fallback_spy: list[tuple[str, str]],
) -> None:
    class _BadRedis:
        async def execute_command(self, *_a: Any, **_kw: Any) -> Any:
            raise redis.ConnectionError("simulated")

    ext = _make_extractor(logger=logger, redis_client=_BadRedis())
    fv = await ext.extract(
        _base_input(query_type_raw="rag", tenant_tier_raw="free"),
    )
    expected = CACHE_HIT_PROB_FALLBACK_TABLE[
        (FeatureQueryType.RAG, FeatureTenantTier.FREE)
    ]
    assert fv.cache_hit_prob == expected
    assert fallback_spy == [("cache_hit_prob", "redis_connection_error")]


@pytest.mark.asyncio
async def test_cache_hit_prob_timeout_fallback(
    logger: logging.Logger,
    fallback_spy: list[tuple[str, str]],
) -> None:
    class _SlowRedis:
        async def execute_command(self, *_a: Any, **_kw: Any) -> int:
            await asyncio.sleep(CACHE_HIT_REDIS_TIMEOUT_SECONDS * 3)
            return 0

    ext = _make_extractor(logger=logger, redis_client=_SlowRedis())
    fv = await ext.extract(
        _base_input(query_type_raw="code", tenant_tier_raw="enterprise"),
    )
    expected = CACHE_HIT_PROB_FALLBACK_TABLE[
        (FeatureQueryType.CODE, FeatureTenantTier.ENTERPRISE)
    ]
    assert fv.cache_hit_prob == expected
    assert fallback_spy == [("cache_hit_prob", "asyncio_timeout")]


@pytest.mark.asyncio
async def test_cache_hit_prob_no_redis_uses_fallback(
    logger: logging.Logger,
    fallback_spy: list[tuple[str, str]],
) -> None:
    ext = _make_extractor(logger=logger, redis_client=None)
    fv = await ext.extract(
        _base_input(query_type_raw="summarize", tenant_tier_raw="pro"),
    )
    expected = CACHE_HIT_PROB_FALLBACK_TABLE[
        (FeatureQueryType.SUMMARIZE, FeatureTenantTier.PRO)
    ]
    assert fv.cache_hit_prob == expected
    assert fallback_spy == [("cache_hit_prob", "redis_not_configured")]


def test_cache_hit_prob_fallback_function_logs_debug(
    caplog: pytest.LogCaptureFixture,
    logger: logging.Logger,
) -> None:
    caplog.set_level(logging.DEBUG)
    prob = _cache_hit_prob_fallback(
        FeatureQueryType.CHAT,
        FeatureTenantTier.ENTERPRISE,
        logger=logger,
        reason="unit_test",
    )
    assert (
        prob
        == CACHE_HIT_PROB_FALLBACK_TABLE[
            (FeatureQueryType.CHAT, FeatureTenantTier.ENTERPRISE)
        ]
    )
    assert any("cache_hit_prob_fallback" in r.message for r in caplog.records)


def test_feature_vector_to_log_dict_roundtrip() -> None:
    fv = FeatureVector(
        query_len=3,
        token_budget=64,
        query_type=FeatureQueryType.RAG,
        tenant_tier=FeatureTenantTier.PRO,
        latency_slo_ms=900,
        queue_depth=1,
        gpu_load=0.5,
        cache_hit_prob=0.25,
    )
    d = fv.to_log_dict()
    assert d["query_type"] == "rag"
    assert d["tenant_tier"] == "pro"
    assert d["cache_hit_prob"] == 0.25
