"""Kafka constants for topics and environment keys."""

from __future__ import annotations

TOPIC_REWARDS = "forgeai.rewards"
TOPIC_DECISIONS = "forgeai.decisions"
TOPIC_MODEL_LOADS = "forgeai.model_loads"
TOPIC_EVAL_LABELS = "forgeai.eval_labels"
TOPIC_REGRESSIONS = "forgeai.regressions"
TOPIC_DLQ = "forgeai.dlq"

ENV_KAFKA_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_KAFKA_CONSUMER_GROUP_ID = "KAFKA_CONSUMER_GROUP_ID"

__all__ = [
    "ENV_KAFKA_BOOTSTRAP_SERVERS",
    "ENV_KAFKA_CONSUMER_GROUP_ID",
    "TOPIC_DECISIONS",
    "TOPIC_DLQ",
    "TOPIC_EVAL_LABELS",
    "TOPIC_MODEL_LOADS",
    "TOPIC_REGRESSIONS",
    "TOPIC_REWARDS",
]
