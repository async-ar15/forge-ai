"""Pre-execution feature extraction for the contextual bandit (Section 3 state vector).

Does not own: LinUCB scoring, routing_decisions persistence, or post-hoc rewards.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import redis
import tiktoken

from forgeai.policy.constants import (
    CACHE_HIT_PROB_FALLBACK_TABLE,
    CACHE_HIT_REDIS_TIMEOUT_SECONDS,
    FEATURE_EXTRACTION_TARGET_MS,
    FEATURE_EXTRACTION_TARGET_US,
    LATENCY_SLO_DEFAULT_MS_ENTERPRISE,
    LATENCY_SLO_DEFAULT_MS_FREE,
    LATENCY_SLO_DEFAULT_MS_PRO,
    NANOSECONDS_PER_MICROSECOND,
    TIKTOKEN_ENCODING_CL100K_BASE,
    TOKEN_BUDGET_MAX_VALID_INCLUSIVE,
    TOKEN_BUDGET_MIN_VALID,
    FeatureQueryType,
    FeatureTenantTier,
)
from forgeai.policy.metrics import record_feature_extraction_fallback
from forgeai.policy.probes import AsyncBloomRedisClient, GpuLoadProbe, QueueDepthProbe
from forgeai.proto import common_pb2

_T = TypeVar("_T")

_ENCODER: tiktoken.Encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING_CL100K_BASE)

_FEATURE_QUERY_TO_PROTO: dict[FeatureQueryType, common_pb2.QueryType.ValueType] = {
    FeatureQueryType.RAG: common_pb2.QUERY_TYPE_RAG,
    FeatureQueryType.CODE: common_pb2.QUERY_TYPE_CODE,
    FeatureQueryType.CHAT: common_pb2.QUERY_TYPE_CHAT,
    FeatureQueryType.SUMMARIZE: common_pb2.QUERY_TYPE_SUMMARIZE,
    FeatureQueryType.UNKNOWN: common_pb2.QUERY_TYPE_UNKNOWN,
}

_PROTO_INT_TO_FEATURE_QUERY: dict[int, FeatureQueryType] = {
    common_pb2.QUERY_TYPE_UNSPECIFIED: FeatureQueryType.UNKNOWN,
    common_pb2.QUERY_TYPE_RAG: FeatureQueryType.RAG,
    common_pb2.QUERY_TYPE_CODE: FeatureQueryType.CODE,
    common_pb2.QUERY_TYPE_CHAT: FeatureQueryType.CHAT,
    common_pb2.QUERY_TYPE_SUMMARIZE: FeatureQueryType.SUMMARIZE,
    common_pb2.QUERY_TYPE_UNKNOWN: FeatureQueryType.UNKNOWN,
}

_FEATURE_TENANT_TO_PROTO: dict[FeatureTenantTier, common_pb2.TenantTier.ValueType] = {
    FeatureTenantTier.FREE: common_pb2.TENANT_TIER_FREE,
    FeatureTenantTier.PRO: common_pb2.TENANT_TIER_PRO,
    FeatureTenantTier.ENTERPRISE: common_pb2.TENANT_TIER_ENTERPRISE,
    FeatureTenantTier.INTERNAL: common_pb2.TENANT_TIER_UNSPECIFIED,
}

_PROTO_INT_TO_FEATURE_TENANT: dict[int, FeatureTenantTier] = {
    common_pb2.TENANT_TIER_UNSPECIFIED: FeatureTenantTier.FREE,
    common_pb2.TENANT_TIER_FREE: FeatureTenantTier.FREE,
    common_pb2.TENANT_TIER_PRO: FeatureTenantTier.PRO,
    common_pb2.TENANT_TIER_ENTERPRISE: FeatureTenantTier.ENTERPRISE,
}


def _feature_query_from_proto_wire(value: int) -> FeatureQueryType:
    return _PROTO_INT_TO_FEATURE_QUERY.get(value, FeatureQueryType.UNKNOWN)


def _feature_tenant_from_proto_wire(value: int) -> FeatureTenantTier:
    return _PROTO_INT_TO_FEATURE_TENANT.get(value, FeatureTenantTier.FREE)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Eight Section-3 features available strictly before model execution."""

    query_len: int
    token_budget: int
    query_type: FeatureQueryType
    tenant_tier: FeatureTenantTier
    latency_slo_ms: int
    queue_depth: int
    gpu_load: float
    cache_hit_prob: float

    def to_log_dict(self) -> dict[str, float | int | str]:
        """JSON-friendly dict for structured logging and debugging."""

        return {
            "query_len": self.query_len,
            "token_budget": self.token_budget,
            "query_type": self.query_type.value,
            "tenant_tier": self.tenant_tier.value,
            "latency_slo_ms": self.latency_slo_ms,
            "queue_depth": self.queue_depth,
            "gpu_load": self.gpu_load,
            "cache_hit_prob": self.cache_hit_prob,
        }

    def to_proto(self) -> common_pb2.StateVector:
        """Serialize to ``forgeai.v1.StateVector`` for internal gRPC."""

        return common_pb2.StateVector(
            query_len=self.query_len,
            token_budget=self.token_budget,
            query_type=_FEATURE_QUERY_TO_PROTO[self.query_type],
            tenant_tier=_FEATURE_TENANT_TO_PROTO[self.tenant_tier],
            latency_slo_ms=self.latency_slo_ms,
            queue_depth=self.queue_depth,
            gpu_load=self.gpu_load,
            cache_hit_prob=self.cache_hit_prob,
        )

    @classmethod
    def from_proto(cls, message: common_pb2.StateVector) -> FeatureVector:
        """Parse ``forgeai.v1.StateVector``; stray enum ints map to safe defaults."""

        return cls(
            query_len=int(message.query_len),
            token_budget=int(message.token_budget),
            query_type=_feature_query_from_proto_wire(int(message.query_type)),
            tenant_tier=_feature_tenant_from_proto_wire(int(message.tenant_tier)),
            latency_slo_ms=int(message.latency_slo_ms),
            queue_depth=int(message.queue_depth),
            gpu_load=float(message.gpu_load),
            cache_hit_prob=float(message.cache_hit_prob),
        )


@dataclass(frozen=True, slots=True)
class FeatureExtractionInput:
    """Inbound HTTP/gateway fields required to build the state vector."""

    query_text: str
    token_budget: int
    query_type_raw: str | None
    latency_slo_ms_raw: int | None
    tenant_tier_raw: str | None


@dataclass(frozen=True, slots=True)
class _SyncFeaturePartial:
    """Intermediate bundle after synchronous extraction steps."""

    query_len: int
    token_budget: int
    query_type: FeatureQueryType
    tenant_tier: FeatureTenantTier
    latency_slo_ms: int


def _elapsed_microseconds(start_ns: int) -> int:
    delta_ns = time.perf_counter_ns() - start_ns
    return delta_ns // NANOSECONDS_PER_MICROSECOND


def _timed_sync(
    logger: logging.Logger,
    feature_name: str,
    fn: Callable[[], _T],
) -> tuple[_T, int]:
    start = time.perf_counter_ns()
    value = fn()
    elapsed_us = _elapsed_microseconds(start)
    logger.debug(
        "feature_extract_step feature=%s microseconds=%d",
        feature_name,
        elapsed_us,
    )
    return value, elapsed_us


async def _timed_async(
    logger: logging.Logger,
    feature_name: str,
    awaitable: Awaitable[_T],
) -> tuple[_T, int]:
    start = time.perf_counter_ns()
    value = await awaitable
    elapsed_us = _elapsed_microseconds(start)
    logger.debug(
        "feature_extract_step feature=%s microseconds=%d",
        feature_name,
        elapsed_us,
    )
    return value, elapsed_us


def _count_query_tokens(query_text: str) -> int:
    return len(_ENCODER.encode(query_text))


def _validate_token_budget(token_budget: int) -> int:
    if token_budget < TOKEN_BUDGET_MIN_VALID:
        msg = f"token_budget must be >= {TOKEN_BUDGET_MIN_VALID}, got {token_budget}"
        raise ValueError(msg)
    if token_budget > TOKEN_BUDGET_MAX_VALID_INCLUSIVE:
        msg = (
            "token_budget must be <= "
            f"{TOKEN_BUDGET_MAX_VALID_INCLUSIVE}, got {token_budget}"
        )
        raise ValueError(msg)
    return token_budget


def _normalize_query_type(
    raw: str | None,
    logger: logging.Logger,
) -> FeatureQueryType:
    if raw is None:
        logger.warning("query_type absent; defaulting to unknown")
        return FeatureQueryType.UNKNOWN
    key = raw.strip().lower()
    try:
        return FeatureQueryType(key)
    except ValueError:
        logger.warning("unrecognized query_type=%r; mapping to unknown", raw)
        return FeatureQueryType.UNKNOWN


def _normalize_tenant_tier(
    raw: str | None,
    logger: logging.Logger,
) -> FeatureTenantTier:
    if raw is None:
        logger.warning("tenant_tier absent from auth context; defaulting to free")
        return FeatureTenantTier.FREE
    key = raw.strip().lower()
    try:
        return FeatureTenantTier(key)
    except ValueError:
        logger.warning("unrecognized tenant_tier=%r; defaulting to free", raw)
        return FeatureTenantTier.FREE


def _latency_default_for_tier(tier: FeatureTenantTier) -> int:
    if tier is FeatureTenantTier.FREE:
        return LATENCY_SLO_DEFAULT_MS_FREE
    if tier is FeatureTenantTier.PRO:
        return LATENCY_SLO_DEFAULT_MS_PRO
    return LATENCY_SLO_DEFAULT_MS_ENTERPRISE


def _resolve_latency_slo_ms(
    raw: int | None,
    tenant_tier: FeatureTenantTier,
) -> int:
    if raw is None:
        return _latency_default_for_tier(tenant_tier)
    if raw <= 0:
        msg = f"latency_slo_ms must be > 0 when provided, got {raw}"
        raise ValueError(msg)
    return raw


def _cache_bloom_item(query_text: str) -> str:
    normalized = query_text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cache_hit_prob_fallback(
    query_type: FeatureQueryType,
    tenant_tier: FeatureTenantTier,
    *,
    logger: logging.Logger,
    reason: str,
) -> float:
    prob = CACHE_HIT_PROB_FALLBACK_TABLE[(query_type, tenant_tier)]
    logger.debug(
        "cache_hit_prob_fallback query_type=%s tenant_tier=%s reason=%s prob=%s",
        query_type.value,
        tenant_tier.value,
        reason,
        prob,
    )
    return prob


async def _redis_bf_exists(
    client: AsyncBloomRedisClient,
    bloom_key: str,
    item: str,
    timeout_seconds: float,
) -> bool:
    async def _call() -> bool:
        raw = await client.execute_command("BF.EXISTS", bloom_key, item)
        return bool(raw)

    return await asyncio.wait_for(_call(), timeout=timeout_seconds)


def _log_redis_failure(logger: logging.Logger, exc: BaseException) -> None:
    logger.info(
        "cache_hit_prob redis_path class=%s msg=%s",
        type(exc).__name__,
        exc,
    )


def _cache_hit_redis_fallback_path(
    exc: BaseException,
    logger: logging.Logger,
    query_type: FeatureQueryType,
    tenant_tier: FeatureTenantTier,
    *,
    metric_reason: str,
    log_reason: str,
) -> float:
    _log_redis_failure(logger, exc)
    record_feature_extraction_fallback(
        feature_name="cache_hit_prob",
        reason=metric_reason,
    )
    return _cache_hit_prob_fallback(
        query_type,
        tenant_tier,
        logger=logger,
        reason=log_reason,
    )


async def _cache_hit_prob_from_redis(
    *,
    client: AsyncBloomRedisClient,
    bloom_key: str,
    item: str,
    timeout_seconds: float,
    query_type: FeatureQueryType,
    tenant_tier: FeatureTenantTier,
    logger: logging.Logger,
) -> float:
    try:
        exists = await _redis_bf_exists(client, bloom_key, item, timeout_seconds)
        return 1.0 if exists else 0.0
    except TimeoutError as exc:
        return _cache_hit_redis_fallback_path(
            exc,
            logger,
            query_type,
            tenant_tier,
            metric_reason="asyncio_timeout",
            log_reason="asyncio_timeout",
        )
    except redis.ConnectionError as exc:
        return _cache_hit_redis_fallback_path(
            exc,
            logger,
            query_type,
            tenant_tier,
            metric_reason="redis_connection_error",
            log_reason="redis_connection_error",
        )
    except Exception as exc:
        return _cache_hit_redis_fallback_path(
            exc,
            logger,
            query_type,
            tenant_tier,
            metric_reason="redis_unexpected_error",
            log_reason=type(exc).__name__,
        )


class FeatureExtractor:
    """Builds the eight-dimensional Section-3 state vector on the inference hot path."""

    def __init__(
        self,
        queue_depth_probe: QueueDepthProbe,
        gpu_load_probe: GpuLoadProbe,
        redis_client: AsyncBloomRedisClient | None,
        redis_bloom_key: str,
        logger: logging.Logger,
    ) -> None:
        """Wire probes, optional Redis, and structured logger (no hidden globals).

        What it takes: DI probes, async Redis client (or ``None``), Bloom key, logger.
        What it raises: nothing during construction.
        Performance contract: construction is O(1); no network I/O.
        """

        self._queue = queue_depth_probe
        self._gpu = gpu_load_probe
        self._redis = redis_client
        self._bloom_key = redis_bloom_key
        self._log = logger

    async def extract(self, inp: FeatureExtractionInput) -> FeatureVector:
        """Compute the Section-3 ``FeatureVector`` for one request.

        What it does: runs eight extraction steps in order with per-step DEBUG timings.
        What it takes: ``FeatureExtractionInput`` (query text, budgets, auth hints).
        What it returns: an immutable ``FeatureVector``.
        What it raises: ``ValueError`` for invalid budget or SLO fields.
        Performance contract: target <5ms local CPU on warm path; Redis Bloom has a 1ms
        ``wait_for`` cap then mandatory static fallback so pre-execution can stay <10ms
        when the cache probe degrades.
        """

        return await _run_feature_extraction(self, inp)


def _measure_sync_features(
    inp: FeatureExtractionInput,
    log: logging.Logger,
) -> tuple[_SyncFeaturePartial, dict[str, int]]:
    timings: dict[str, int] = {}
    query_len, us = _timed_sync(
        log,
        "query_len",
        lambda: _count_query_tokens(inp.query_text),
    )
    timings["query_len"] = us
    token_budget, us = _timed_sync(
        log,
        "token_budget",
        lambda: _validate_token_budget(inp.token_budget),
    )
    timings["token_budget"] = us
    query_type, us = _timed_sync(
        log,
        "query_type",
        lambda: _normalize_query_type(inp.query_type_raw, log),
    )
    timings["query_type"] = us
    tenant_tier, us = _timed_sync(
        log,
        "tenant_tier",
        lambda: _normalize_tenant_tier(inp.tenant_tier_raw, log),
    )
    timings["tenant_tier"] = us
    latency_slo_ms, us = _timed_sync(
        log,
        "latency_slo_ms",
        lambda: _resolve_latency_slo_ms(inp.latency_slo_ms_raw, tenant_tier),
    )
    timings["latency_slo_ms"] = us
    bundle = _SyncFeaturePartial(
        query_len=query_len,
        token_budget=token_budget,
        query_type=query_type,
        tenant_tier=tenant_tier,
        latency_slo_ms=latency_slo_ms,
    )
    return bundle, timings


async def _measure_async_features(
    fx: FeatureExtractor,
    inp: FeatureExtractionInput,
    sync: _SyncFeaturePartial,
    timings: dict[str, int],
) -> FeatureVector:
    log = fx._log
    qd = fx._queue.read_queue_depth()
    queue_depth, us = await _timed_async(log, "queue_depth", qd)
    timings["queue_depth"] = us
    gpu_load, us = await _timed_async(log, "gpu_load", fx._gpu.read_gpu_load())
    timings["gpu_load"] = us
    cache_hit_prob, us = await _extract_cache_hit_prob_timed(
        fx,
        inp,
        sync.query_type,
        sync.tenant_tier,
    )
    timings["cache_hit_prob"] = us
    return FeatureVector(
        query_len=sync.query_len,
        token_budget=sync.token_budget,
        query_type=sync.query_type,
        tenant_tier=sync.tenant_tier,
        latency_slo_ms=sync.latency_slo_ms,
        queue_depth=queue_depth,
        gpu_load=gpu_load,
        cache_hit_prob=cache_hit_prob,
    )


async def _run_feature_extraction(
    fx: FeatureExtractor,
    inp: FeatureExtractionInput,
) -> FeatureVector:
    log = fx._log
    t0 = time.perf_counter_ns()
    sync, timings = _measure_sync_features(inp, log)
    vector = await _measure_async_features(fx, inp, sync, timings)
    total_us = _elapsed_microseconds(t0)
    log.info("feature_extract_total microseconds=%d", total_us)
    if total_us > FEATURE_EXTRACTION_TARGET_US:
        log.warning(
            "feature_extract_slow total_microseconds=%d target_ms=%d breakdown=%s",
            total_us,
            FEATURE_EXTRACTION_TARGET_MS,
            timings,
        )
    return vector


async def _extract_cache_hit_prob_timed(
    fx: FeatureExtractor,
    inp: FeatureExtractionInput,
    query_type: FeatureQueryType,
    tenant_tier: FeatureTenantTier,
) -> tuple[float, int]:
    log = fx._log
    start = time.perf_counter_ns()
    prob = await _resolve_cache_hit_prob(
        fx._redis,
        fx._bloom_key,
        inp.query_text,
        query_type,
        tenant_tier,
        log,
    )
    elapsed_us = _elapsed_microseconds(start)
    log.debug(
        "feature_extract_step feature=%s microseconds=%d",
        "cache_hit_prob",
        elapsed_us,
    )
    return prob, elapsed_us


async def _resolve_cache_hit_prob(
    client: AsyncBloomRedisClient | None,
    bloom_key: str,
    query_text: str,
    query_type: FeatureQueryType,
    tenant_tier: FeatureTenantTier,
    logger: logging.Logger,
) -> float:
    if client is None:
        logger.info(
            "cache_hit_prob redis_unconfigured class=%s msg=%s",
            "RuntimeError",
            "redis client not configured",
        )
        record_feature_extraction_fallback(
            feature_name="cache_hit_prob",
            reason="redis_not_configured",
        )
        return _cache_hit_prob_fallback(
            query_type,
            tenant_tier,
            logger=logger,
            reason="redis_not_configured",
        )

    item = _cache_bloom_item(query_text)
    return await _cache_hit_prob_from_redis(
        client=client,
        bloom_key=bloom_key,
        item=item,
        timeout_seconds=CACHE_HIT_REDIS_TIMEOUT_SECONDS,
        query_type=query_type,
        tenant_tier=tenant_tier,
        logger=logger,
    )


__all__ = [
    "FeatureExtractionInput",
    "FeatureExtractor",
    "FeatureVector",
]
