"""Process-wide gateway dependencies constructed at startup."""

from __future__ import annotations

from dataclasses import dataclass

import grpc.aio
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from forgeai.config import Settings
from forgeai.eval.response_store import ResponseStore
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.policy.features import FeatureExtractor
from forgeai.policy.reward_pipeline import RewardPipeline
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import (
    execution_service_pb2_grpc,
    policy_service_pb2_grpc,
    registry_service_pb2_grpc,
)


@dataclass(slots=True)
class GatewayState:
    """Shared stubs and extractors; never constructed per request."""

    settings: Settings
    registry_engine: AsyncEngine
    redis_client: Redis[str] | None
    session_factory: async_sessionmaker[AsyncSession]
    policy_stub: policy_service_pb2_grpc.PolicyServiceStub
    execution_stub: execution_service_pb2_grpc.ExecutionServiceStub
    registry_stub: registry_service_pb2_grpc.RegistryServiceStub | None
    policy_channel: grpc.aio.Channel
    execution_channel: grpc.aio.Channel
    registry_channel: grpc.aio.Channel | None
    feature_extractor: FeatureExtractor
    decision_logger: DecisionLogger
    reward_pipeline: RewardPipeline
    kafka_producer: ForgeKafkaProducer | None
    response_store: ResponseStore
    tenant_policy: TenantPolicy
    policy_version_boot: str
    loaded_models_boot: list[str]
    shutdown_drain_seconds: int


__all__ = ["GatewayState"]
