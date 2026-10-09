"""Retrieval tests: env-backed ``Settings`` and cache-clear between cases."""

from __future__ import annotations

import pytest
from forgeai.config import Settings


@pytest.fixture
def retrieval_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Minimal settings including mandatory Qdrant collection name."""

    env = {
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/db",
        "REDIS_URL": "redis://localhost:6379/0",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION_NAME": "retrieval-test-coll",
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
    }
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    gs = __import__("forgeai.config", fromlist=["get_settings"]).get_settings
    gs.cache_clear()
    return gs()
