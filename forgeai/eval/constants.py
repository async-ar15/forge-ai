"""Offline evaluation constants and controlled defaults for v1."""

from __future__ import annotations

import os
from enum import StrEnum


class LabelSource(StrEnum):
    """Provenance values compatible with routing_decisions.label_source."""

    JUDGE = "judge"
    HUMAN = "human"
    AUTO = "auto"


# Rubric weights. Changing these redefines what q_offline measures and therefore
# changes retraining targets and historical comparability across policy versions.
RUBRIC_WEIGHT_RELEVANCE = 0.30
RUBRIC_WEIGHT_GROUNDEDNESS = 0.25
RUBRIC_WEIGHT_COMPLETENESS = 0.25
RUBRIC_WEIGHT_CONCISENESS = 0.20
_WEIGHT_SUM = (
    RUBRIC_WEIGHT_RELEVANCE
    + RUBRIC_WEIGHT_GROUNDEDNESS
    + RUBRIC_WEIGHT_COMPLETENESS
    + RUBRIC_WEIGHT_CONCISENESS
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-12, "rubric weights must sum to exactly 1.0"

# Prompt version. Bump when rubric prompt wording changes to preserve auditability.
JUDGE_PROMPT_VERSION = "v1"

# Judge model identifier for audit and reproducibility. Changing this may shift
# scoring calibration and should trigger reliability re-baselining.
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "mistral-7b-instruct")

# Response store settings. TTL controls how long decisions remain labelable; lower
# values reduce Redis usage but increase missing-label rate.
RESPONSE_STORE_TTL_SECONDS = int(os.environ.get("RESPONSE_STORE_TTL_SECONDS", "86400"))
RESPONSE_STORE_KEY_PREFIX = "forgeai:responses:"

# Reliability monitor cadence. Shorter intervals detect collapse faster but may
# overreact to noise; longer intervals delay detection of judge failure modes.
JUDGE_RELIABILITY_WINDOW_SIZE = 100
JUDGE_LOW_STD_THRESHOLD = 0.05
JUDGE_PARSE_FAILURE_RATE_THRESHOLD = 0.10

# Retraining threshold. Lower values retrain more often (faster adaptation, more
# churn); higher values stabilize but delay learning from offline labels.
RETRAINING_LABEL_THRESHOLD = int(os.environ.get("RETRAINING_LABEL_THRESHOLD", "1000"))

# Blend weight between online reward (fast signal) and offline quality label
# (slow but accurate). Higher alpha = trust online reward more.
RETRAINING_REWARD_BLEND_ALPHA = float(
    os.environ.get("RETRAINING_REWARD_BLEND_ALPHA", "0.7")
)

# Regression checks. Smaller intervals catch regressions sooner but increase
# alert volume. Thresholds tune sensitivity and on-call load.
REGRESSION_CHECK_INTERVAL_SECONDS = int(
    os.environ.get("REGRESSION_CHECK_INTERVAL_SECONDS", "3600")
)
REGRESSION_R_ONLINE_THRESHOLD = float(
    os.environ.get("REGRESSION_R_ONLINE_THRESHOLD", "0.10")
)
REGRESSION_LATENCY_THRESHOLD_MS = float(
    os.environ.get("REGRESSION_LATENCY_THRESHOLD_MS", "100")
)
REGRESSION_SLO_THRESHOLD = float(os.environ.get("REGRESSION_SLO_THRESHOLD", "0.05"))

# Internal judge routing identity for evaluation traffic separation.
INTERNAL_JUDGE_TENANT_ID = "00000000-0000-0000-0000-0000000000f1"
INTERNAL_JUDGE_SESSION_ID = "forgeai-rubriceval-judge"


__all__ = [
    "INTERNAL_JUDGE_SESSION_ID",
    "INTERNAL_JUDGE_TENANT_ID",
    "JUDGE_LOW_STD_THRESHOLD",
    "JUDGE_MODEL_NAME",
    "JUDGE_PARSE_FAILURE_RATE_THRESHOLD",
    "JUDGE_PROMPT_VERSION",
    "JUDGE_RELIABILITY_WINDOW_SIZE",
    "LabelSource",
    "REGRESSION_CHECK_INTERVAL_SECONDS",
    "REGRESSION_LATENCY_THRESHOLD_MS",
    "REGRESSION_R_ONLINE_THRESHOLD",
    "REGRESSION_SLO_THRESHOLD",
    "RESPONSE_STORE_KEY_PREFIX",
    "RESPONSE_STORE_TTL_SECONDS",
    "RETRAINING_LABEL_THRESHOLD",
    "RETRAINING_REWARD_BLEND_ALPHA",
    "RUBRIC_WEIGHT_COMPLETENESS",
    "RUBRIC_WEIGHT_CONCISENESS",
    "RUBRIC_WEIGHT_GROUNDEDNESS",
    "RUBRIC_WEIGHT_RELEVANCE",
]
