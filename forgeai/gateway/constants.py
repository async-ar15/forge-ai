"""Named limits and route paths for the inference gateway (units in symbol names)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# --- HTTP paths (external contract) ---
INFER_PATH: Final[str] = "/v1/infer"
OPENAI_CHAT_COMPLETIONS_PATH: Final[str] = "/v1/chat/completions"
OPENAI_MODELS_PATH: Final[str] = "/v1/models"
HEALTH_PATH: Final[str] = "/v1/health"
METRICS_PATH: Final[str] = "/v1/metrics"

# --- Request body limits ---
QUERY_MAX_CHARS: Final[int] = 32768
TOKEN_BUDGET_DEFAULT: Final[int] = 2048
TOKEN_BUDGET_MIN_INCLUSIVE: Final[int] = 1
TOKEN_BUDGET_MAX_INCLUSIVE: Final[int] = 32768

# --- Auth ---
AUTH_CACHE_TTL_SECONDS: Final[int] = 60
AUTH_REDIS_KEY_PREFIX: Final[str] = "forgeai:gateway:auth:v1:"
# bcrypt hash of b"forgeai_auth_timing_dummy_v1" (cost 12).
# Reject path always checks this hash.
AUTH_DUMMY_BCRYPT_HASH_UTF8: Final[str] = (
    "$2b$12$DQ1eyOF53dPrGtuEWQ4YZexXeQF.GgBHOMl6Ep5uc4xEbDnVA7ZtO"
)

# --- Rate limit (requests per rolling 60s window) ---
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
RATE_LIMIT_FREE_RPM: Final[int] = 10
RATE_LIMIT_PRO_RPM: Final[int] = 100
RATE_LIMIT_ENTERPRISE_RPM: Final[int] = 1000
RATE_LIMIT_REDIS_KEY_PREFIX: Final[str] = "forgeai:gateway:rl:v1:"

# --- gRPC keepalive ---
GRPC_KEEPALIVE_TIME_MS: Final[int] = 10_000
GRPC_KEEPALIVE_TIMEOUT_MS: Final[int] = 5_000
GRPC_MAX_PINGS_WITHOUT_DATA: Final[int] = 0

# --- Policy / health probes ---
POLICY_STARTUP_TIMEOUT_SECONDS: Final[float] = 2.0
REGISTRY_HEALTH_TIMEOUT_SECONDS: Final[float] = 1.0
METRICS_PROXY_TIMEOUT_SECONDS: Final[float] = 5.0

# --- Degraded / fail-open ---
FAIL_OPEN_TENANT_ID_ZERO: Final[str] = "00000000-0000-0000-0000-000000000000"


class GatewayRoute(StrEnum):
    """Legacy fragments retained for imports; prefer INFER_PATH constants."""

    HEALTHZ = "/healthz"
    METRICS = "/metrics"


__all__ = [
    "AUTH_CACHE_TTL_SECONDS",
    "AUTH_DUMMY_BCRYPT_HASH_UTF8",
    "AUTH_REDIS_KEY_PREFIX",
    "FAIL_OPEN_TENANT_ID_ZERO",
    "GatewayRoute",
    "GRPC_KEEPALIVE_TIME_MS",
    "GRPC_KEEPALIVE_TIMEOUT_MS",
    "GRPC_MAX_PINGS_WITHOUT_DATA",
    "HEALTH_PATH",
    "INFER_PATH",
    "METRICS_PATH",
    "OPENAI_CHAT_COMPLETIONS_PATH",
    "OPENAI_MODELS_PATH",
    "POLICY_STARTUP_TIMEOUT_SECONDS",
    "QUERY_MAX_CHARS",
    "RATE_LIMIT_ENTERPRISE_RPM",
    "RATE_LIMIT_FREE_RPM",
    "RATE_LIMIT_PRO_RPM",
    "RATE_LIMIT_REDIS_KEY_PREFIX",
    "RATE_LIMIT_WINDOW_SECONDS",
    "REGISTRY_HEALTH_TIMEOUT_SECONDS",
    "METRICS_PROXY_TIMEOUT_SECONDS",
    "TOKEN_BUDGET_DEFAULT",
    "TOKEN_BUDGET_MAX_INCLUSIVE",
    "TOKEN_BUDGET_MIN_INCLUSIVE",
]
