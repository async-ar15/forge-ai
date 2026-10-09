from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from forgeai.config import Settings
from forgeai.gateway.app import create_test_app
from forgeai.gateway.gateway_state import GatewayState
from forgeai.gateway.tenant_context import TenantContext
from forgeai.policy.bandit import LinUCBBandit
from forgeai.policy.bandit_actions import (
    ALL_ACTIONS,
    action_spec_to_proto,
    canonical_action_key,
)
from forgeai.policy.decision_log_record import DecisionLog
from forgeai.policy.features import FeatureExtractor
from forgeai.policy.grpc_service import ForgePolicyServicer
from forgeai.policy.probes import LocalGpuLoadProbe, LocalQueueDepthProbe
from forgeai.policy.reward_pipeline import RewardPipeline
from forgeai.policy.tenant_policy import TenantPolicy
from forgeai.proto import (
    common_pb2,
    execution_service_pb2,
    execution_service_pb2_grpc,
    policy_service_pb2,
    policy_service_pb2_grpc,
    retrieval_service_pb2,
    retrieval_service_pb2_grpc,
)
from forgeai.retrieval.grpc_service import ForgeRetrievalServicer
from sqlalchemy import create_engine, text


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    env = {
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
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
    get_settings = __import__("forgeai.config", fromlist=["get_settings"]).get_settings
    get_settings.cache_clear()
    return get_settings()


class _BloomToggle:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    async def execute_command(self, cmd: str, _key: str, item: str) -> int:
        if cmd != "BF.EXISTS":
            return 0
        if item in self._seen:
            return 1
        self._seen.add(item)
        return 0


class _PolicyStub:
    def __init__(self, *, force_explore: bool = False) -> None:
        self.force_explore = force_explore
        self.Decide = AsyncMock(side_effect=self._decide)

    async def _decide(
        self, request: policy_service_pb2.PolicyDecideRequest, metadata=()
    ):
        internal = int(request.routing_state_features.tenant_tier) == int(
            common_pb2.TENANT_TIER_UNSPECIFIED
        )
        q_unknown = int(request.routing_state_features.query_type) == int(
            common_pb2.QUERY_TYPE_UNKNOWN
        )
        chosen = (
            common_pb2.ActionSpec(
                model_tier=common_pb2.MODEL_TIER_MEDIUM,
                precision=common_pb2.PRECISION_FP16,
                retrieval_mode=common_pb2.RETRIEVAL_MODE_CACHE_ONLY,
                output_budget=common_pb2.OUTPUT_BUDGET_MEDIUM,
            )
            if q_unknown or internal
            else action_spec_to_proto(ALL_ACTIONS[0])
        )
        greedy = chosen
        scores = {} if internal else {canonical_action_key(a): 0.0 for a in ALL_ACTIONS}
        explore = self.force_explore and not internal
        et = (
            common_pb2.EXPLORATION_TYPE_EPSILON
            if explore
            else (
                common_pb2.EXPLORATION_TYPE_FORCED
                if (q_unknown or internal)
                else common_pb2.EXPLORATION_TYPE_NONE
            )
        )
        return policy_service_pb2.PolicyDecideResponse(
            chosen_action=chosen,
            exploration_flag=explore,
            exploration_type=et,
            greedy_action=greedy,
            action_scores=scores,
            policy_version="pol-test",
        )


class _ExecutionStream:
    def __init__(
        self, chunks: list[execution_service_pb2.ExecutionExecuteServerStreamChunk]
    ) -> None:
        self._chunks = iter(chunks)

    def __aiter__(
        self,
    ) -> AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]:
        return self

    async def __anext__(
        self,
    ) -> execution_service_pb2.ExecutionExecuteServerStreamChunk:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _ExecutionStub:
    def __init__(self) -> None:
        self._seen_prompts: set[str] = set()
        self.Execute = MagicMock(side_effect=self._execute)

    def _execute(
        self, request: execution_service_pb2.ExecutionExecuteRequest, metadata=()
    ):
        prompt = request.user_prompt_utf8
        warm = prompt in self._seen_prompts
        self._seen_prompts.add(prompt)
        delay = 0.001 if warm else 0.02

        async def _chunks() -> (
            AsyncIterator[execution_service_pb2.ExecutionExecuteServerStreamChunk]
        ):
            await asyncio.sleep(delay)
            yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
                generated_text_delta_utf8="hello",
                stream_finished=False,
                cumulative_output_tokens_estimate=1,
            )
            ts = __import__(
                "google.protobuf.timestamp_pb2", fromlist=["Timestamp"]
            ).Timestamp()
            ts.FromDatetime(datetime.now(UTC))
            yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
                generated_text_delta_utf8=" world",
                stream_finished=True,
                cumulative_output_tokens_estimate=2,
                stream_final_cost_microdollars=common_pb2.MoneyMicrodollars(
                    amount_microdollars=2000
                ),
                execution_end_timestamp_utc=ts,
            )

        return _chunks()


@dataclass(slots=True)
class _RecorderDecisionLogger:
    rows: list[DecisionLog]

    async def log_decision(self, decision: DecisionLog) -> None:
        self.rows.append(decision)

    async def update_online_reward(
        self, request_id: uuid.UUID, *, r_online: float, r_components: dict[str, float]
    ) -> None:
        for row in self.rows:
            if row.request_id == request_id:
                row.r_online = float(r_online)
                row.r_components = {k: float(v) for k, v in r_components.items()}
                return


def _gateway_state(settings: Settings, *, force_explore: bool = False) -> GatewayState:
    policy_stub = _PolicyStub(force_explore=force_explore)
    exec_stub = _ExecutionStub()
    bloom = _BloomToggle()
    fx = FeatureExtractor(
        queue_depth_probe=LocalQueueDepthProbe(0),
        gpu_load_probe=LocalGpuLoadProbe(0.0),
        redis_client=bloom,
        redis_bloom_key="forgeai:bloom:test",
        logger=__import__("logging").getLogger("integration.features"),
    )
    logger = _RecorderDecisionLogger(rows=[])
    tenant_policy = TenantPolicy(
        coeff_latency=settings.reward_coeff_latency_ms,
        coeff_cost=settings.reward_coeff_cost_usd,
        coeff_slo_violation=settings.reward_coeff_slo_violation,
        coeff_fallback=settings.reward_coeff_fallback,
        coeff_retry=settings.reward_coeff_retry,
        coeff_cache_hit=settings.reward_coeff_cache_hit_bonus,
    )
    reward = RewardPipeline(
        decision_logger=logger,  # type: ignore[arg-type]
        tenant_policy=tenant_policy,
        policy_stub=None,
        kafka_producer=None,
    )
    response_store = MagicMock()
    response_store.store = AsyncMock(return_value=None)
    return GatewayState(
        settings=settings,
        registry_engine=MagicMock(),
        redis_client=None,
        session_factory=MagicMock(),
        policy_stub=policy_stub,  # type: ignore[arg-type]
        execution_stub=exec_stub,  # type: ignore[arg-type]
        registry_stub=None,
        policy_channel=MagicMock(),
        execution_channel=MagicMock(),
        registry_channel=None,
        feature_extractor=fx,
        decision_logger=logger,  # type: ignore[arg-type]
        reward_pipeline=reward,
        kafka_producer=None,
        response_store=response_store,
        tenant_policy=tenant_policy,
        policy_version_boot="pol-test",
        loaded_models_boot=[],
        shutdown_drain_seconds=1,
    )


def _parse_sse(text: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def _auth_ctx(tier: str = "free") -> TenantContext:
    return TenantContext(tenant_id=uuid.uuid4(), tenant_tier=tier, auth_degraded=False)


@pytest.fixture
def integration_client(monkeypatch: pytest.MonkeyPatch):
    settings = _settings(monkeypatch)
    state = _gateway_state(settings)
    app = create_test_app(settings, state)
    from starlette.testclient import TestClient

    async def _auth(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return _auth_ctx("free"), True

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        TestClient(app) as client,
    ):
        yield client, state


def test_happy_path_full_retrieval(integration_client) -> None:
    client, state = integration_client
    resp = client.post(
        "/v1/infer",
        json={
            "query": "hello world",
            "query_type": "rag",
            "token_budget": 512,
            "latency_slo_ms": 1000,
            "stream": True,
        },
        headers={"X-API-Key": "k"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(bool(e.get("token")) for e in events)
    row = state.decision_logger.rows[-1]
    assert row.r_online < 0.0
    assert isinstance(row.exploration_flag, bool)
    assert len(row.action_scores) == 81
    assert isinstance(row.cache_hit, bool)


def test_cache_hit_path_lower_latency(integration_client) -> None:
    client, state = integration_client
    payload = {"query": "same text", "query_type": "rag", "stream": False}
    client.post("/v1/infer", json=payload, headers={"X-API-Key": "k"})
    first = state.decision_logger.rows[-1]
    client.post("/v1/infer", json=payload, headers={"X-API-Key": "k"})
    second = state.decision_logger.rows[-1]
    assert second.cache_hit is True
    assert second.final_latency_ms < first.final_latency_ms


def test_conservative_default_on_unknown(integration_client) -> None:
    client, state = integration_client
    client.post(
        "/v1/infer",
        json={"query": "x", "query_type": "unknown"},
        headers={"X-API-Key": "k"},
    )
    row = state.decision_logger.rows[-1]
    assert (row.model_tier, row.precision, row.retrieval_mode, row.output_budget) == (
        "medium",
        "fp16",
        "cache_only",
        "medium",
    )
    assert row.exploration_type == "forced"


def test_rate_limit_enforcement(integration_client) -> None:
    client, _state = integration_client
    calls = {"n": 0}

    async def _allow(*_a: object, **_k: object):
        calls["n"] += 1
        if calls["n"] == 11:
            return False, 12
        return True, 0

    with patch(
        "forgeai.gateway.middleware.auth_and_limit.sliding_window_allow", new=_allow
    ):
        for i in range(10):
            r = client.post(
                "/v1/infer", json={"query": f"{i}"}, headers={"X-API-Key": "k"}
            )
            assert r.status_code == 200
        blocked = client.post(
            "/v1/infer", json={"query": "11"}, headers={"X-API-Key": "k"}
        )
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After") == "12"


def test_auth_rejection_no_decision_logged(integration_client) -> None:
    client, state = integration_client
    before = len(state.decision_logger.rows)

    async def _deny(*_a: object, **_k: object):
        return None, False

    with patch(
        "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context", new=_deny
    ):
        resp = client.post(
            "/v1/infer", json={"query": "x"}, headers={"X-API-Key": "bad"}
        )
    assert resp.status_code == 401
    assert len(state.decision_logger.rows) == before


def test_slo_violation_recording(integration_client) -> None:
    client, state = integration_client
    client.post(
        "/v1/infer",
        json={"query": "slow", "latency_slo_ms": 1},
        headers={"X-API-Key": "k"},
    )
    bad = state.decision_logger.rows[-1]
    client.post(
        "/v1/infer",
        json={"query": "slow2", "latency_slo_ms": 2000},
        headers={"X-API-Key": "k"},
    )
    good = state.decision_logger.rows[-1]
    assert bad.slo_violation is True
    assert bad.r_online < good.r_online


def test_exploration_distribution_and_view_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    state = _gateway_state(settings, force_explore=True)
    app = create_test_app(settings, state)
    from starlette.testclient import TestClient

    async def _auth(*_a: object, **_k: object):
        return _auth_ctx("free"), True

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        TestClient(app) as client,
    ):
        for i in range(100):
            client.post(
                "/v1/infer", json={"query": f"e{i}"}, headers={"X-API-Key": "k"}
            )
    assert all(r.exploration_flag for r in state.decision_logger.rows)
    eng = create_engine("sqlite+pysqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE routing_decisions ("
                "request_id TEXT, exploration_flag INTEGER, tenant_tier TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE VIEW routing_decisions_exploitative AS SELECT * "
                "FROM routing_decisions WHERE exploration_flag = FALSE "
                "AND tenant_tier != 'internal'"
            )
        )
        for i, _row in enumerate(state.decision_logger.rows):
            conn.execute(
                text("INSERT INTO routing_decisions VALUES (:r,:e,:t)"),
                {"r": f"{i}", "e": 1, "t": "free"},
            )
        count = conn.execute(
            text("SELECT COUNT(*) FROM routing_decisions_exploitative")
        ).scalar_one()
    assert int(count) == 0


def test_internal_tier_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    state = _gateway_state(settings)
    app = create_test_app(settings, state)
    from starlette.testclient import TestClient

    async def _auth(*_a: object, **_k: object):
        return _auth_ctx("internal"), True

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        TestClient(app) as client,
    ):
        client.post(
            "/v1/infer",
            json={"query": "judge request", "query_type": "chat"},
            headers={"X-API-Key": "k"},
        )
    row = state.decision_logger.rows[-1]
    assert row.tenant_tier == "internal"
    assert row.action_scores == {}
    eng = create_engine("sqlite+pysqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE routing_decisions ("
                "request_id TEXT, exploration_flag INTEGER, tenant_tier TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE VIEW routing_decisions_exploitative AS SELECT * "
                "FROM routing_decisions WHERE exploration_flag = FALSE "
                "AND tenant_tier != 'internal'"
            )
        )
        conn.execute(text("INSERT INTO routing_decisions VALUES ('a', 0, 'internal')"))
        count = conn.execute(
            text("SELECT COUNT(*) FROM routing_decisions_exploitative")
        ).scalar_one()
    assert int(count) == 0


def test_decision_log_completeness(integration_client) -> None:
    client, state = integration_client
    client.post("/v1/infer", json={"query": "once"}, headers={"X-API-Key": "k"})
    row = state.decision_logger.rows[-1]
    assert row.request_id is not None
    assert row.policy_version
    assert row.tenant_id is not None
    assert len(row.state_vector.keys()) == 8
    assert len(row.r_components.keys()) == 6
    assert row.greedy_action in ALL_ACTIONS


def test_proto_contract_validation_inprocess_grpc() -> None:
    policy_server = grpc.server(
        __import__(
            "concurrent.futures", fromlist=["ThreadPoolExecutor"]
        ).ThreadPoolExecutor(max_workers=4)
    )
    policy_servicer = ForgePolicyServicer(
        LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0)
    )
    policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(
        policy_servicer, policy_server
    )
    p_port = policy_server.add_insecure_port("127.0.0.1:0")
    policy_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{p_port}") as ch:
            stub = policy_service_pb2_grpc.PolicyServiceStub(ch)
            ts = __import__(
                "google.protobuf.timestamp_pb2", fromlist=["Timestamp"]
            ).Timestamp()
            ts.FromDatetime(datetime.now(UTC))
            req = policy_service_pb2.PolicyDecideRequest(
                correlation_request_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                routing_state_features=common_pb2.StateVector(
                    query_len=10,
                    token_budget=256,
                    query_type=common_pb2.QUERY_TYPE_RAG,
                    tenant_tier=common_pb2.TENANT_TIER_PRO,
                    latency_slo_ms=1000,
                    queue_depth=1,
                    gpu_load=0.1,
                    cache_hit_prob=0.2,
                ),
                execution_deployment_version="dep",
                gateway_ingress_timestamp_utc=ts,
            )
            resp = stub.Decide(req)
            assert resp.policy_version
            assert len(resp.action_scores) == 81
            if not resp.exploration_flag:
                assert resp.greedy_action == resp.chosen_action
    finally:
        policy_server.stop(grace=None)

    class _TinyRetrievalEngine:
        async def retrieve(self, req: retrieval_service_pb2.RetrievalRetrieveRequest):
            return retrieval_service_pb2.RetrievalRetrieveResponse(
                context_passage_utf8=[f"mode={int(req.retrieval_mode)}"],
                cache_hit=False,
                retrieval_internal_latency_ms=1,
            )

    retrieval_server = grpc.server(
        __import__(
            "concurrent.futures", fromlist=["ThreadPoolExecutor"]
        ).ThreadPoolExecutor(max_workers=4)
    )
    retrieval_service_pb2_grpc.add_RetrievalServiceServicer_to_server(
        ForgeRetrievalServicer(_TinyRetrievalEngine()), retrieval_server
    )
    r_port = retrieval_server.add_insecure_port("127.0.0.1:0")
    retrieval_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{r_port}") as ch:
            stub = retrieval_service_pb2_grpc.RetrievalServiceStub(ch)
            for mode in (
                common_pb2.RETRIEVAL_MODE_OFF,
                common_pb2.RETRIEVAL_MODE_CACHE_ONLY,
                common_pb2.RETRIEVAL_MODE_FULL,
            ):
                out = stub.Retrieve(
                    retrieval_service_pb2.RetrievalRetrieveRequest(
                        correlation_request_id="c",
                        tenant_id="t",
                        retrieval_mode=mode,
                        retrieval_query_text_utf8="q",
                    )
                )
                assert isinstance(out.cache_hit, bool)
                assert out.retrieval_internal_latency_ms >= 0
    finally:
        retrieval_server.stop(grace=None)

    class _ExecServicer(execution_service_pb2_grpc.ExecutionServiceServicer):
        def Execute(self, request, context):  # noqa: ANN001
            ts = __import__(
                "google.protobuf.timestamp_pb2", fromlist=["Timestamp"]
            ).Timestamp()
            ts.FromDatetime(datetime.now(UTC))
            yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
                generated_text_delta_utf8="a",
                stream_finished=False,
                cumulative_output_tokens_estimate=1,
            )
            yield execution_service_pb2.ExecutionExecuteServerStreamChunk(
                generated_text_delta_utf8="",
                stream_finished=True,
                cumulative_output_tokens_estimate=1,
                stream_final_cost_microdollars=common_pb2.MoneyMicrodollars(
                    amount_microdollars=1
                ),
                execution_end_timestamp_utc=ts,
            )

    exec_server = grpc.server(
        __import__(
            "concurrent.futures", fromlist=["ThreadPoolExecutor"]
        ).ThreadPoolExecutor(max_workers=4)
    )
    execution_service_pb2_grpc.add_ExecutionServiceServicer_to_server(
        _ExecServicer(), exec_server
    )
    e_port = exec_server.add_insecure_port("127.0.0.1:0")
    exec_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{e_port}") as ch:
            stub = execution_service_pb2_grpc.ExecutionServiceStub(ch)
            req = execution_service_pb2.ExecutionExecuteRequest(
                correlation_request_id="c",
                tenant_id="t",
                routing_action_spec=common_pb2.ActionSpec(
                    model_tier=common_pb2.MODEL_TIER_SMALL,
                    precision=common_pb2.PRECISION_FP16,
                    retrieval_mode=common_pb2.RETRIEVAL_MODE_OFF,
                    output_budget=common_pb2.OUTPUT_BUDGET_SHORT,
                ),
                user_prompt_utf8="hi",
                max_output_token_limit=32,
            )
            chunks = list(stub.Execute(req))
            assert len(chunks) == 2
            assert chunks[-1].stream_finished is True
    finally:
        exec_server.stop(grace=None)
