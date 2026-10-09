"""StateVector → LinUCB context vector (min-max + one-hot, Section 3).

Does not own: protobuf parsing, arm algebra, or exploration policy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from forgeai.policy.constants import (
    FEATURE_CACHE_HIT_PROB_MAX,
    FEATURE_CACHE_HIT_PROB_MIN,
    FEATURE_DIM,
    FEATURE_GPU_LOAD_MAX,
    FEATURE_GPU_LOAD_MIN,
    FEATURE_LATENCY_SLO_MS_MAX,
    FEATURE_LATENCY_SLO_MS_MIN,
    FEATURE_QUERY_LEN_MAX,
    FEATURE_QUERY_LEN_MIN,
    FEATURE_QUEUE_DEPTH_MAX,
    FEATURE_QUEUE_DEPTH_MIN,
    FEATURE_TOKEN_BUDGET_MAX,
    FEATURE_TOKEN_BUDGET_MIN,
    FeatureQueryType,
    FeatureTenantTier,
)
from forgeai.policy.features import FeatureVector


def _norm_interval(value: float, lo: float, hi: float) -> float:
    """Linear min-max to ``[0, 1]``; constant interval yields 0.0 (defensive)."""

    clipped = min(max(value, lo), hi)
    if hi <= lo:
        return 0.0
    return (clipped - lo) / (hi - lo)


def _continuous_features(features: FeatureVector) -> list[float]:
    """Six min-max normalized scalars in fixed Section 3 order."""

    return [
        _norm_interval(
            float(features.query_len),
            float(FEATURE_QUERY_LEN_MIN),
            float(FEATURE_QUERY_LEN_MAX),
        ),
        _norm_interval(
            float(features.token_budget),
            float(FEATURE_TOKEN_BUDGET_MIN),
            float(FEATURE_TOKEN_BUDGET_MAX),
        ),
        _norm_interval(
            float(features.latency_slo_ms),
            float(FEATURE_LATENCY_SLO_MS_MIN),
            float(FEATURE_LATENCY_SLO_MS_MAX),
        ),
        _norm_interval(
            float(features.queue_depth),
            float(FEATURE_QUEUE_DEPTH_MIN),
            float(FEATURE_QUEUE_DEPTH_MAX),
        ),
        _norm_interval(
            float(features.gpu_load),
            FEATURE_GPU_LOAD_MIN,
            FEATURE_GPU_LOAD_MAX,
        ),
        _norm_interval(
            float(features.cache_hit_prob),
            FEATURE_CACHE_HIT_PROB_MIN,
            FEATURE_CACHE_HIT_PROB_MAX,
        ),
    ]


def encode_feature_vector(features: FeatureVector) -> npt.NDArray[np.float64]:
    """``x`` shaped ``(FEATURE_DIM,)`` float64: 6 continuous + 5 query + 4 tier one-hot.

    Time O(FEATURE_DIM); one heap allocation for the dense vector.
    """

    cont = _continuous_features(features)
    query_oh = [1.0 if features.query_type is q else 0.0 for q in FeatureQueryType]
    tier_oh = [1.0 if features.tenant_tier is t else 0.0 for t in FeatureTenantTier]
    vec = np.array(cont + query_oh + tier_oh, dtype=np.float64)
    assert vec.shape == (FEATURE_DIM,)
    return vec


__all__ = ["encode_feature_vector"]
