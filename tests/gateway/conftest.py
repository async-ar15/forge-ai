"""Shared fixtures for gateway tests."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.config import Settings
from forgeai.eval.response_store import ResponseStore
from forgeai.gateway.app import create_test_app
from forgeai.gateway.gateway_state import GatewayState
from forgeai.policy.bandit_actions import (
    ALL_ACTIONS,
    action_spec_to_proto,
    canonical_action_key,
)
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.policy.features import FeatureExtractor
from forgeai.policy.probes import LocalGpuLoadProbe, LocalQueueDepthProbe
from forgeai.policy.reward_pipeline import RewardPipeline
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import common_pb2, execution_service_pb2, policy_service_pb2
from google.protobuf.timestamp_pb2 import Timestamp


def _full_action_scores() -> dict[str, float]:
    return {canonical_action_key(a): 0.0 for a in ALL_ACTIONS}


def _policy_response() -> policy_service_pb2.PolicyDecideResponse:
    chosen = ALL_ACTIONS[0]
    return policy_service_pb2.PolicyDecideResponse(
        chosen_action=action_spec_to_proto(chosen),
        exploration_flag=False,
        exploration_type=common_pb2.EXPLORATION_TYPE_NONE,
        greedy_action=action_spec_to_proto(chosen),
        action_scores=_full_action_scores(),
        policy_version="test-pol",
    )


class _FakeUnaryStream:
    __slots__ = ("_chunks", "_it")

    def __init__(
        self,
        chunks: list[execution_service_pb2.ExecutionExecuteServerStreamChunk],
    ):
        self._chunks = chunks
        self._it = iter(chunks)

    def __aiter__(
        self,
    ) -> AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]:
        return self

    async def __anext__(
        self,
    ) -> execution_service_pb2.ExecutionExecuteServerStreamChunk:
        try:
            return next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _terminal_chunk(
    *,
    text: str = "",
    cost_micro: int = 42,
    err: str | None = None,
) -> execution_service_pb2.ExecutionExecuteServerStreamChunk:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(UTC))
    money = common_pb2.MoneyMicrodollars(amount_microdollars=cost_micro)
    ch = execution_service_pb2.ExecutionExecuteServerStreamChunk(
        generated_text_delta_utf8=text,
        stream_finished=True,
        cumulative_output_tokens_estimate=1,
        stream_final_cost_microdollars=money,
        execution_end_timestamp_utc=ts,
    )
    if err:
        ch.stream_error_message_utf8 = err
    return ch


def _delta_chunk(text: str) -> execution_service_pb2.ExecutionExecuteServerStreamChunk:
    return execution_service_pb2.ExecutionExecuteServerStreamChunk(
        generated_text_delta_utf8=text,
        stream_finished=False,
        cumulative_output_tokens_estimate=1,
    )


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Minimal ``Settings`` for gateway tests (env-backed fields satisfied)."""

    env = {
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/db",
        "REDIS_URL": "redis://localhost:6379/0",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION_NAME": "test-collection",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "KAFKA_CLIENT_ID": "test",
        "S3_ENDPOINT_URL": "http://localhost:9000",
        "S3_ACCESS_KEY_ID": "k",
        "S3_SECRET_ACCESS_KEY": "s",
        "S3_REGION": "us-east-1",
        "MODEL_REGISTRY_BUCKET": "b",
        "REWARD_COEFF_LATENCY_MS": "0.01",
        "REWARD_COEFF_COST_USD": "1",
        "REWARD_COEFF_SLO_VIOLATION": "1",
        "REWARD_COEFF_FALLBACK": "1",
        "REWARD_COEFF_RETRY": "1",
        "REWARD_COEFF_CACHE_HIT_BONUS": "0.05",
        "POLICY_ENGINE_GRPC_HOST": "127.0.0.1",
        "POLICY_ENGINE_GRPC_PORT": "50051",
        "EXECUTION_ENGINE_GRPC_HOST": "127.0.0.1",
        "EXECUTION_ENGINE_GRPC_PORT": "50052",
        "RETRIEVAL_ENGINE_GRPC_HOST": "127.0.0.1",
        "RETRIEVAL_ENGINE_GRPC_PORT": "50053",
        "MODEL_REGISTRY_GRPC_HOST": "127.0.0.1",
        "MODEL_REGISTRY_GRPC_PORT": "50054",
        "GATEWAY_HOST": "0.0.0.0",
        "GATEWAY_PORT": "8080",
        "GATEWAY_LOG_LEVEL": "INFO",
        "PROMETHEUS_METRICS_PORT": "9099",
        "GATEWAY_INTERNAL_METRICS_TOKEN": "secret-metrics-token",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    get_settings = __import__(
        "forgeai.config",
        fromlist=["get_settings"],
    ).get_settings
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def mock_gateway_state(test_settings: Settings) -> GatewayState:
    """GatewayState with mocked gRPC stubs and extractors."""

    policy_stub = MagicMock()
    policy_stub.Decide = AsyncMock(return_value=_policy_response())
    exec_stub = MagicMock()

    def _exec(*_a: object, **_k: object) -> _FakeUnaryStream:
        return _FakeUnaryStream(
            [_delta_chunk("hi"), _terminal_chunk(text="", cost_micro=99)],
        )

    exec_stub.Execute = MagicMock(side_effect=_exec)
    reg_stub = MagicMock()
    redis = MagicMock()
    eng = MagicMock()
    eng.dispose = AsyncMock()
    sess = MagicMock()
    sess.execute = AsyncMock(
        return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=None),
        ),
    )
    sf = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=sess)
    cm.__aexit__ = AsyncMock(return_value=None)
    sf.return_value = cm
    fx = FeatureExtractor(
        queue_depth_probe=LocalQueueDepthProbe(0),
        gpu_load_probe=LocalGpuLoadProbe(0.0),
        redis_client=None,
        redis_bloom_key="t",
        logger=__import__("logging").getLogger("test.fx"),
    )
    dlog = MagicMock(spec=DecisionLogger)
    dlog.log_decision = AsyncMock(return_value=None)
    dlog.update_online_reward = AsyncMock(return_value=None)
    tp = TenantPolicy(
        coeff_latency=test_settings.reward_coeff_latency_ms,
        coeff_cost=test_settings.reward_coeff_cost_usd,
        coeff_slo_violation=test_settings.reward_coeff_slo_violation,
        coeff_fallback=test_settings.reward_coeff_fallback,
        coeff_retry=test_settings.reward_coeff_retry,
        coeff_cache_hit=test_settings.reward_coeff_cache_hit_bonus,
    )
    reward_pipeline = MagicMock(spec=RewardPipeline)
    reward_pipeline.compute_and_record = AsyncMock(return_value=0.0)
    response_store = MagicMock(spec=ResponseStore)
    response_store.store = AsyncMock(return_value=None)
    p_ch = MagicMock()
    e_ch = MagicMock()
    r_ch = MagicMock()
    return GatewayState(
        settings=test_settings,
        registry_engine=eng,
        redis_client=redis,
        session_factory=sf,
        policy_stub=policy_stub,
        execution_stub=exec_stub,
        registry_stub=reg_stub,
        policy_channel=p_ch,
        execution_channel=e_ch,
        registry_channel=r_ch,
        feature_extractor=fx,
        decision_logger=dlog,
        reward_pipeline=reward_pipeline,
        kafka_producer=None,
        response_store=response_store,
        tenant_policy=tp,
        policy_version_boot="test-pol",
        loaded_models_boot=["m1"],
        shutdown_drain_seconds=1,
    )


@pytest.fixture
def gateway_client(mock_gateway_state: GatewayState):
    """FastAPI ``TestClient`` with a pre-built gateway state."""

    from starlette.testclient import TestClient

    app = create_test_app(mock_gateway_state.settings, mock_gateway_state)
    with TestClient(app) as client:
        yield client, mock_gateway_state


@pytest.fixture
def tenant_row_bcrypt() -> tuple[str, str, uuid.UUID, str]:
    """Return (plaintext_key, sha256_hex, tenant_id, bcrypt_hash)."""

    import bcrypt

    key = "gw-test-api-key-abcdef"
    sha = (
        __import__(
            "hashlib",
            fromlist=["sha256"],
        )
        .sha256(key.encode())
        .hexdigest()
    )
    h = bcrypt.hashpw(key.encode(), bcrypt.gensalt(rounds=8)).decode()
    tid = uuid.uuid4()
    return key, sha, tid, h
