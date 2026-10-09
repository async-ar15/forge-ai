"""Central Prometheus metric definitions used by all ForgeAI services.

No service-specific module should define new metric objects. Import from here.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Gateway.
METRIC_GATEWAY_REQUESTS_TOTAL = "forgeai_gateway_requests_total"
METRIC_GATEWAY_REQUEST_DURATION_SECONDS = "forgeai_gateway_request_duration_seconds"
METRIC_GATEWAY_AUTH_CACHE_HITS_TOTAL = "forgeai_gateway_auth_cache_hits_total"
METRIC_GATEWAY_RATE_LIMIT_REJECTIONS_TOTAL = (
    "forgeai_gateway_rate_limit_rejections_total"
)

# Policy.
METRIC_BANDIT_UPDATES_TOTAL = "forgeai_bandit_updates_total"
METRIC_POLICY_RPC_DURATION_SECONDS = "forgeai_policy_rpc_duration_seconds"
METRIC_FEATURE_EXTRACTION_FALLBACK_TOTAL = "forgeai_feature_extraction_fallback_total"
METRIC_DECISIONS_LOGGED_TOTAL = "forgeai_decisions_logged_total"
METRIC_DECISIONS_FALLBACK_TOTAL = "forgeai_decisions_fallback_total"
METRIC_DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL = (
    "forgeai_decisions_consistency_violations_total"
)
METRIC_DECISIONS_LABEL_MISSING_TOTAL = "forgeai_decisions_label_missing_total"
METRIC_REWARD_VALUE = "forgeai_reward_value"
METRIC_REQUEST_COST_USD = "forgeai_request_cost_usd"

# Execution.
METRIC_EXECUTION_REQUEST_DURATION_SECONDS = "forgeai_execution_request_duration_seconds"
METRIC_EXECUTION_TOKENS_PER_SECOND = "forgeai_execution_tokens_per_second"
METRIC_EXECUTION_MODEL_LOAD_ERRORS_TOTAL = "forgeai_execution_model_load_errors_total"

# Retrieval.
METRIC_RETRIEVAL_REQUEST_DURATION_SECONDS = "forgeai_retrieval_request_duration_seconds"
METRIC_RETRIEVAL_CACHE_HIT_TOTAL = "forgeai_retrieval_cache_hit_total"
METRIC_RETRIEVAL_QDRANT_ERRORS_TOTAL = "forgeai_retrieval_qdrant_errors_total"
METRIC_RETRIEVAL_PREFETCH_ERRORS_TOTAL = "forgeai_retrieval_prefetch_errors_total"

# Kafka.
METRIC_KAFKA_SEND_ERRORS_TOTAL = "forgeai_kafka_send_errors_total"
METRIC_KAFKA_SERIALIZATION_ERRORS_TOTAL = "forgeai_kafka_serialization_errors_total"
METRIC_KAFKA_CONSUMER_LAG_GAUGE = "forgeai_kafka_consumer_lag_gauge"

# Eval.
METRIC_JUDGE_SCORE_MEAN = "forgeai_judge_score_mean"
METRIC_JUDGE_SCORE_STD = "forgeai_judge_score_std"
METRIC_JUDGE_PARSE_FAILURE_RATE = "forgeai_judge_parse_failure_rate"
METRIC_EVAL_LABELS_WRITTEN_TOTAL = "forgeai_eval_labels_written_total"
METRIC_EVAL_JUDGE_FAILURES_TOTAL = "forgeai_eval_judge_failures_total"
METRIC_POLICY_RETRAINING_TOTAL = "forgeai_policy_retraining_total"
METRIC_REGRESSION_DETECTED_TOTAL = "forgeai_regression_detected_total"

GATEWAY_REQUESTS_TOTAL = Counter(
    METRIC_GATEWAY_REQUESTS_TOTAL,
    "Count of gateway HTTP requests.",
    labelnames=("query_type", "tenant_tier", "status_code"),
)
GATEWAY_REQUEST_DURATION_SECONDS = Histogram(
    METRIC_GATEWAY_REQUEST_DURATION_SECONDS,
    "Wall time for gateway-handled HTTP requests.",
    labelnames=("query_type", "tenant_tier"),
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        float("inf"),
    ),
)
GATEWAY_AUTH_CACHE_HITS_TOTAL = Counter(
    METRIC_GATEWAY_AUTH_CACHE_HITS_TOTAL,
    "Authentication cache hits in the gateway auth path.",
)
GATEWAY_RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    METRIC_GATEWAY_RATE_LIMIT_REJECTIONS_TOTAL,
    "Rate-limit rejections at gateway middleware.",
    labelnames=("tenant_tier",),
)

BANDIT_UPDATES_TOTAL = Counter(
    METRIC_BANDIT_UPDATES_TOTAL,
    "LinUCB disjoint-arm posterior updates applied on the reward path.",
    labelnames=("action_key",),
)
POLICY_RPC_DURATION_SECONDS = Histogram(
    METRIC_POLICY_RPC_DURATION_SECONDS,
    "Wall time spent inside PolicyService RPC handlers (seconds).",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
FEATURE_EXTRACTION_FALLBACK_TOTAL = Counter(
    METRIC_FEATURE_EXTRACTION_FALLBACK_TOTAL,
    "Feature extraction used a fallback instead of the primary probe/backend.",
    labelnames=("feature_name", "reason"),
)
DECISIONS_LOGGED_TOTAL = Counter(
    METRIC_DECISIONS_LOGGED_TOTAL,
    "routing_decisions rows committed successfully.",
)
DECISIONS_FALLBACK_TOTAL = Counter(
    METRIC_DECISIONS_FALLBACK_TOTAL,
    "Decision rows written to JSONL fallback instead of Postgres.",
    labelnames=("reason",),
)
DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL = Counter(
    METRIC_DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL,
    "Pre-write invariant violations logged but still persisted.",
)
DECISIONS_LABEL_MISSING_TOTAL = Counter(
    METRIC_DECISIONS_LABEL_MISSING_TOTAL,
    "Phase-2 label updates that targeted a missing request_id.",
)
REWARD_VALUE = Histogram(
    METRIC_REWARD_VALUE,
    "Online reward value written on synchronous request completion path.",
    labelnames=("model_tier", "precision", "retrieval_mode"),
    buckets=(-10_000.0, -5_000.0, -1_000.0, -500.0, -100.0, -10.0, 0.0, 10.0, 100.0),
)
REQUEST_COST_USD = Histogram(
    METRIC_REQUEST_COST_USD,
    "Final request cost in USD observed at request completion.",
    labelnames=("model_tier", "precision", "retrieval_mode", "cache_hit"),
    buckets=(0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, float("inf")),
)

EXECUTION_REQUEST_DURATION_SECONDS = Histogram(
    METRIC_EXECUTION_REQUEST_DURATION_SECONDS,
    "Wall time for one Execute RPC (stream lifetime).",
    labelnames=("model_tier", "precision"),
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        float("inf"),
    ),
)
EXECUTION_TOKENS_PER_SECOND = Gauge(
    METRIC_EXECUTION_TOKENS_PER_SECOND,
    "Observed output token throughput for the last completed stream.",
    labelnames=("model_tier", "precision"),
)
EXECUTION_MODEL_LOAD_ERRORS_TOTAL = Counter(
    METRIC_EXECUTION_MODEL_LOAD_ERRORS_TOTAL,
    "Model materialization failures before streaming.",
)

RETRIEVAL_REQUEST_DURATION_SECONDS = Histogram(
    METRIC_RETRIEVAL_REQUEST_DURATION_SECONDS,
    "Wall time for one Retrieve RPC.",
    labelnames=("retrieval_mode", "cache_hit"),
    buckets=(
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        float("inf"),
    ),
)
RETRIEVAL_CACHE_HIT_TOTAL = Counter(
    METRIC_RETRIEVAL_CACHE_HIT_TOTAL,
    "Semantic cache outcomes labeled by bandit retrieval mode.",
    labelnames=("retrieval_mode",),
)
RETRIEVAL_QDRANT_ERRORS_TOTAL = Counter(
    METRIC_RETRIEVAL_QDRANT_ERRORS_TOTAL,
    "Qdrant failures during full retrieval.",
)
RETRIEVAL_PREFETCH_ERRORS_TOTAL = Counter(
    METRIC_RETRIEVAL_PREFETCH_ERRORS_TOTAL,
    "PCR prefetch background failures.",
)

KAFKA_SEND_ERRORS_TOTAL = Counter(
    METRIC_KAFKA_SEND_ERRORS_TOTAL,
    "Kafka producer send errors.",
    labelnames=("topic",),
)
KAFKA_SERIALIZATION_ERRORS_TOTAL = Counter(
    METRIC_KAFKA_SERIALIZATION_ERRORS_TOTAL,
    "Kafka event serialization errors.",
)
KAFKA_CONSUMER_LAG_GAUGE = Gauge(
    METRIC_KAFKA_CONSUMER_LAG_GAUGE,
    "Kafka consumer lag estimate.",
    labelnames=("topic",),
)

JUDGE_SCORE_MEAN = Gauge(
    METRIC_JUDGE_SCORE_MEAN,
    "Rolling mean of rubric judge scores by dimension.",
    labelnames=("dimension",),
)
JUDGE_SCORE_STD = Gauge(
    METRIC_JUDGE_SCORE_STD,
    "Rolling stddev of rubric judge scores by dimension.",
    labelnames=("dimension",),
)
JUDGE_PARSE_FAILURE_RATE = Gauge(
    METRIC_JUDGE_PARSE_FAILURE_RATE,
    "Rolling judge parse failure rate over the reliability window.",
)
EVAL_LABELS_WRITTEN_TOTAL = Counter(
    METRIC_EVAL_LABELS_WRITTEN_TOTAL,
    "Offline labels successfully written to routing_decisions.",
)
EVAL_JUDGE_FAILURES_TOTAL = Counter(
    METRIC_EVAL_JUDGE_FAILURES_TOTAL,
    "Judge parse failures that skipped offline label writes.",
)
POLICY_RETRAINING_TOTAL = Counter(
    METRIC_POLICY_RETRAINING_TOTAL,
    "Successful policy retraining and hot-swap operations.",
)
REGRESSION_DETECTED_TOTAL = Counter(
    METRIC_REGRESSION_DETECTED_TOTAL,
    "Regression alerts emitted by metric type.",
    labelnames=("metric_name",),
)


__all__ = [name for name in globals() if name.isupper()]
