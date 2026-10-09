"""Loading and validation of process configuration from environment variables.

Does not own: per-tenant Postgres policy rows, secret managers, or v2 feature flags.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced exclusively from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(
        ...,
        alias="DATABASE_URL",
        description="Async SQLAlchemy URL for Postgres.",
    )
    redis_url: str = Field(..., alias="REDIS_URL")
    qdrant_url: HttpUrl = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(..., alias="QDRANT_COLLECTION_NAME")
    qdrant_timeout_seconds: float = Field(default=2.0, alias="QDRANT_TIMEOUT_SECONDS")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    retrieval_bloom_filter_key: str = Field(
        default="forgeai:bloom:retrieval",
        alias="RETRIEVAL_BLOOM_FILTER_KEY",
    )

    kafka_bootstrap_servers: str = Field(..., alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_client_id: str = Field(..., alias="KAFKA_CLIENT_ID")
    kafka_consumer_group_id: str = Field(
        default="forgeai-default",
        alias="KAFKA_CONSUMER_GROUP_ID",
    )
    kafka_security_protocol: str = Field(
        default="PLAINTEXT",
        alias="KAFKA_SECURITY_PROTOCOL",
    )

    s3_endpoint_url: HttpUrl = Field(..., alias="S3_ENDPOINT_URL")
    s3_access_key_id: str = Field(..., alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(..., alias="S3_SECRET_ACCESS_KEY")
    s3_region: str = Field(..., alias="S3_REGION")
    model_registry_bucket: str = Field(..., alias="MODEL_REGISTRY_BUCKET")

    reward_coeff_latency_ms: float = Field(..., alias="REWARD_COEFF_LATENCY_MS")
    reward_coeff_cost_usd: float = Field(..., alias="REWARD_COEFF_COST_USD")
    reward_coeff_slo_violation: float = Field(..., alias="REWARD_COEFF_SLO_VIOLATION")
    reward_coeff_fallback: float = Field(..., alias="REWARD_COEFF_FALLBACK")
    reward_coeff_retry: float = Field(..., alias="REWARD_COEFF_RETRY")
    reward_coeff_cache_hit_bonus: float = Field(
        ...,
        alias="REWARD_COEFF_CACHE_HIT_BONUS",
    )

    policy_engine_grpc_host: str = Field(..., alias="POLICY_ENGINE_GRPC_HOST")
    policy_engine_grpc_port: int = Field(..., alias="POLICY_ENGINE_GRPC_PORT")
    policy_service_port: int = Field(default=50051, alias="POLICY_SERVICE_PORT")
    execution_engine_grpc_host: str = Field(..., alias="EXECUTION_ENGINE_GRPC_HOST")
    execution_engine_grpc_port: int = Field(..., alias="EXECUTION_ENGINE_GRPC_PORT")
    retrieval_engine_grpc_host: str = Field(..., alias="RETRIEVAL_ENGINE_GRPC_HOST")
    retrieval_engine_grpc_port: int = Field(..., alias="RETRIEVAL_ENGINE_GRPC_PORT")
    model_registry_grpc_host: str = Field(..., alias="MODEL_REGISTRY_GRPC_HOST")
    model_registry_grpc_port: int = Field(..., alias="MODEL_REGISTRY_GRPC_PORT")

    gateway_host: str = Field(..., alias="GATEWAY_HOST")
    gateway_port: int = Field(..., alias="GATEWAY_PORT")
    gateway_log_level: str = Field(..., alias="GATEWAY_LOG_LEVEL")

    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    prometheus_metrics_port: int = Field(..., alias="PROMETHEUS_METRICS_PORT")

    gateway_grpc_max_send_bytes: int = Field(
        default=32 * 1024 * 1024,
        alias="GATEWAY_GRPC_MAX_SEND_BYTES",
    )
    gateway_grpc_max_receive_bytes: int = Field(
        default=32 * 1024 * 1024,
        alias="GATEWAY_GRPC_MAX_RECEIVE_BYTES",
    )
    gateway_internal_metrics_token: str = Field(
        default="",
        alias="GATEWAY_INTERNAL_METRICS_TOKEN",
    )
    gateway_execution_deployment_version: str = Field(
        default="unknown",
        alias="GATEWAY_EXECUTION_DEPLOYMENT_VERSION",
    )
    gateway_feature_bloom_key: str = Field(
        default="forgeai:cache_hit_bloom",
        alias="GATEWAY_FEATURE_BLOOM_KEY",
    )
    gateway_shutdown_drain_seconds: int = Field(
        default=30,
        alias="GATEWAY_SHUTDOWN_DRAIN_SECONDS",
    )
    forgeai_demo_mode: bool = Field(default=False, alias="FORGEAI_DEMO_MODE")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    ollama_model: str = Field(default="phi", alias="OLLAMA_MODEL")
    gateway_health_loaded_models_csv: str = Field(
        default="",
        alias="GATEWAY_HEALTH_LOADED_MODELS_CSV",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings parsed from the environment."""

    return Settings()


__all__ = ["Settings", "get_settings"]
