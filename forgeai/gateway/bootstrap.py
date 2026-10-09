"""Startup wiring for channels, Redis, extractors, and loggers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Final, cast

import grpc.aio
import httpx
from google.protobuf.timestamp_pb2 import Timestamp
from redis.asyncio import Redis
from redis.asyncio import from_url as redis_from_url
from sqlalchemy.ext.asyncio import async_sessionmaker

from forgeai.config import Settings
from forgeai.eval.response_store import ResponseStore
from forgeai.execution.engine_pool import EnginePool
from forgeai.execution.gpu_probe import ProductionGpuLoadProbe
from forgeai.execution.queue_depth_probe import ProductionQueueDepthProbe
from forgeai.gateway.constants import (
    GRPC_KEEPALIVE_TIME_MS,
    GRPC_KEEPALIVE_TIMEOUT_MS,
    GRPC_MAX_PINGS_WITHOUT_DATA,
    POLICY_STARTUP_TIMEOUT_SECONDS,
)
from forgeai.gateway.gateway_state import GatewayState
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.policy.features import FeatureExtractor
from forgeai.policy.probes import LocalQueueDepthProbe, QueueDepthProbe
from forgeai.policy.reward_pipeline import RewardPipeline
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import (
    common_pb2,
    execution_service_pb2_grpc,
    policy_service_pb2,
    policy_service_pb2_grpc,
    registry_service_pb2_grpc,
)
from forgeai.registry.database import create_registry_engine
from forgeai.retrieval.bloom_filter import BloomFeatureRedisAdapter

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_OLLAMA_PROBE_PATH: Final[str] = "/api/tags"
_OLLAMA_TIMEOUT_SECONDS: Final[float] = 10.0


def _grpc_options(settings: Settings) -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", settings.gateway_grpc_max_send_bytes),
        ("grpc.max_receive_message_length", settings.gateway_grpc_max_receive_bytes),
        ("grpc.keepalive_time_ms", GRPC_KEEPALIVE_TIME_MS),
        ("grpc.keepalive_timeout_ms", GRPC_KEEPALIVE_TIMEOUT_MS),
        ("grpc.http2.max_pings_without_data", GRPC_MAX_PINGS_WITHOUT_DATA),
    ]


def _policy_target(settings: Settings) -> str:
    return f"{settings.policy_engine_grpc_host}:{settings.policy_engine_grpc_port}"


def _execution_target(settings: Settings) -> str:
    h = settings.execution_engine_grpc_host
    p = settings.execution_engine_grpc_port
    return f"{h}:{p}"


def _registry_target(settings: Settings) -> str:
    return f"{settings.model_registry_grpc_host}:{settings.model_registry_grpc_port}"


def _tenant_policy(settings: Settings) -> TenantPolicy:
    return TenantPolicy(
        coeff_latency=settings.reward_coeff_latency_ms,
        coeff_cost=settings.reward_coeff_cost_usd,
        coeff_slo_violation=settings.reward_coeff_slo_violation,
        coeff_fallback=settings.reward_coeff_fallback,
        coeff_retry=settings.reward_coeff_retry,
        coeff_cache_hit=settings.reward_coeff_cache_hit_bonus,
    )


def _loaded_models_csv(settings: Settings) -> list[str]:
    raw = settings.gateway_health_loaded_models_csv.strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


class _BloomCmd:
    __slots__ = ("_r",)

    def __init__(self, r: Redis[str]) -> None:
        self._r = r

    async def execute_command(self, *args: object, **kwargs: object) -> object:
        return await cast(Any, self._r).execute_command(*args, **kwargs)


async def bootstrap_gateway_state(
    settings: Settings,
    *,
    engine_pool: EnginePool | None = None,
) -> GatewayState:
    """Construct shared gateway state (one call per process)."""

    redis_client: Redis[str] | None
    try:
        redis_client = redis_from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
    except Exception:
        _LOG.warning("gateway_redis_warm_failed", exc_info=True)
        redis_client = None

    registry_engine = create_registry_engine(settings)
    session_factory = async_sessionmaker(registry_engine, expire_on_commit=False)

    opts = _grpc_options(settings)
    p_ch = grpc.aio.insecure_channel(_policy_target(settings), options=opts)
    e_ch = grpc.aio.insecure_channel(_execution_target(settings), options=opts)
    r_ch = grpc.aio.insecure_channel(_registry_target(settings), options=opts)
    policy_stub = policy_service_pb2_grpc.PolicyServiceStub(p_ch)
    execution_stub = execution_service_pb2_grpc.ExecutionServiceStub(e_ch)
    registry_stub = registry_service_pb2_grpc.RegistryServiceStub(r_ch)

    q_probe: QueueDepthProbe
    if engine_pool is not None:
        q_probe = ProductionQueueDepthProbe(engine_pool)
        _LOG.info("gateway_queue_probe=production_engine_pool")
    else:
        q_probe = LocalQueueDepthProbe(0)
        _LOG.warning(
            "gateway_queue_probe=static_zero "
            "(ProductionQueueDepthProbe needs a co-located EnginePool; "
            "see architecture note for split deployments)",
        )
    gpu_probe = ProductionGpuLoadProbe()
    bloom_client = (
        BloomFeatureRedisAdapter(redis_client) if redis_client is not None else None
    )
    fx = FeatureExtractor(
        queue_depth_probe=q_probe,
        gpu_load_probe=gpu_probe,
        redis_client=bloom_client,
        redis_bloom_key=settings.retrieval_bloom_filter_key,
        logger=logging.getLogger("forgeai.gateway.features"),
    )
    dep_ver = settings.gateway_execution_deployment_version
    dlog = DecisionLogger(session_factory, dep_ver)
    producer = ForgeKafkaProducer()
    kafka_producer: ForgeKafkaProducer | None = producer
    try:
        await producer.start()
    except Exception:
        _LOG.warning("gateway_kafka_producer_start_failed", exc_info=True)
        kafka_producer = None
    tenant_policy = _tenant_policy(settings)
    reward_pipeline = RewardPipeline(
        decision_logger=dlog,
        tenant_policy=tenant_policy,
        policy_stub=policy_stub,
        kafka_producer=kafka_producer,
    )
    response_store = ResponseStore(redis_client)
    if settings.forgeai_demo_mode:
        await _probe_demo_ollama(settings.ollama_url)
    pol_ver = await _probe_policy_version(policy_stub, settings)
    return GatewayState(
        settings=settings,
        registry_engine=registry_engine,
        redis_client=redis_client,
        session_factory=session_factory,
        policy_stub=policy_stub,
        execution_stub=execution_stub,
        registry_stub=registry_stub,
        policy_channel=p_ch,
        execution_channel=e_ch,
        registry_channel=r_ch,
        feature_extractor=fx,
        decision_logger=dlog,
        reward_pipeline=reward_pipeline,
        kafka_producer=kafka_producer,
        response_store=response_store,
        tenant_policy=tenant_policy,
        policy_version_boot=pol_ver,
        loaded_models_boot=_loaded_models_csv(settings),
        shutdown_drain_seconds=settings.gateway_shutdown_drain_seconds,
    )


async def _probe_demo_ollama(base_url: str) -> None:
    target = f"{base_url.rstrip('/')}{_OLLAMA_PROBE_PATH}"
    demo_log = logging.getLogger("uvicorn.error")
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_SECONDS) as client:
            resp = await client.get(target)
            resp.raise_for_status()
            payload = cast(dict[str, object], resp.json())
            models = payload.get("models")
            model_count = len(models) if isinstance(models, list) else 0
        demo_log.info(
            "gateway_demo_ollama_probe_ok url=%s models=%d",
            target,
            model_count,
        )
    except Exception:
        demo_log.info(
            "gateway_demo_ollama_probe_failed url=%s",
            target,
            exc_info=True,
        )


async def _probe_policy_version(
    stub: policy_service_pb2_grpc.PolicyServiceStub,
    settings: Settings,
) -> str:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(UTC))
    req = policy_service_pb2.PolicyDecideRequest(
        correlation_request_id=str(uuid.uuid4()),
        tenant_id="00000000-0000-0000-0000-000000000001",
        routing_state_features=common_pb2.StateVector(
            query_len=1,
            token_budget=256,
            query_type=common_pb2.QUERY_TYPE_UNKNOWN,
            tenant_tier=common_pb2.TENANT_TIER_FREE,
            latency_slo_ms=60_000,
            queue_depth=0,
            gpu_load=0.0,
            cache_hit_prob=0.0,
        ),
        execution_deployment_version=settings.gateway_execution_deployment_version,
        gateway_ingress_timestamp_utc=ts,
    )
    try:
        call = stub.Decide(req, metadata=())
        resp = await asyncio.wait_for(
            cast(Awaitable[policy_service_pb2.PolicyDecideResponse], call),
            timeout=POLICY_STARTUP_TIMEOUT_SECONDS,
        )
        return str(resp.policy_version)
    except Exception:
        _LOG.warning("gateway_policy_version_probe_failed", exc_info=True)
        return "unavailable"


async def shutdown_gateway_state(state: GatewayState) -> None:
    """Close clients with a bounded wait for in-flight work."""

    await asyncio.sleep(0.05)
    try:
        await state.registry_engine.dispose()
    except Exception:
        _LOG.warning("gateway_registry_engine_dispose_swallowed", exc_info=True)
    try:
        await asyncio.wait_for(
            state.execution_channel.close(
                grace=float(state.shutdown_drain_seconds),
            ),
            timeout=float(state.shutdown_drain_seconds) + 1.0,
        )
    except Exception:
        _LOG.warning("gateway_execution_channel_close_swallowed", exc_info=True)
    try:
        await asyncio.wait_for(
            state.policy_channel.close(grace=5.0),
            timeout=6.0,
        )
    except Exception:
        _LOG.warning("gateway_policy_channel_close_swallowed", exc_info=True)
    if state.registry_channel is not None:
        try:
            await state.registry_channel.close(grace=5.0)
        except Exception:
            _LOG.warning("gateway_registry_channel_close_swallowed", exc_info=True)
    if state.redis_client is not None:
        try:
            await state.redis_client.close()
        except Exception:
            _LOG.warning("gateway_redis_close_swallowed", exc_info=True)
    if state.kafka_producer is not None:
        try:
            await asyncio.wait_for(state.kafka_producer.stop(), timeout=5.0)
        except Exception:
            _LOG.warning("gateway_kafka_stop_swallowed", exc_info=True)


__all__ = ["bootstrap_gateway_state", "shutdown_gateway_state"]
