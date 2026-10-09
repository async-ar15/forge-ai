"""Unit tests for LinUCB bandit, persistence, exploration, and online reward."""

from __future__ import annotations

import hashlib
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from forgeai.enums import (
    ExplorationType,
    ModelTier,
    OutputBudget,
    Precision,
    RetrievalMode,
)
from forgeai.policy.bandit import LinUCBBandit, conservative_default_action
from forgeai.policy.bandit_actions import (
    ACTION_SPACE_SIZE,
    ALL_ACTIONS,
    ActionSpec,
    canonical_action_key,
)
from forgeai.policy.bandit_arm import LinUCBArm
from forgeai.policy.bandit_encoding import encode_feature_vector
from forgeai.policy.constants import (
    DEFAULT_TENANT_POLICY,
    FEATURE_DIM,
    FeatureQueryType,
    FeatureTenantTier,
)
from forgeai.policy.features import FeatureVector
from forgeai.policy.reward import RequestOutcome, compute_online_reward


def _fv(
    *,
    qt: FeatureQueryType = FeatureQueryType.RAG,
    qd: int = 1,
    gl: float = 0.5,
) -> FeatureVector:
    """Build a non-cold-start ``FeatureVector`` for bandit tests."""

    return FeatureVector(
        query_len=100,
        token_budget=2048,
        query_type=qt,
        tenant_tier=FeatureTenantTier.PRO,
        latency_slo_ms=800,
        queue_depth=qd,
        gpu_load=gl,
        cache_hit_prob=0.2,
    )


def test_constants_feature_dim_locked_to_fifteen() -> None:
    """Import-time invariant: encoding width must stay 15."""

    assert FEATURE_DIM == 15


def test_feature_dim_matches_encoding() -> None:
    """``encode_feature_vector`` must emit exactly ``FEATURE_DIM``."""

    x = encode_feature_vector(_fv())
    assert x.shape == (FEATURE_DIM,)
    assert x.dtype == np.float64


def test_encode_known_feature_vector_has_shape_fifteen() -> None:
    """Explicit ``FeatureVector`` encodes to shape ``(15,)`` (not import-only)."""

    fv = FeatureVector(
        query_len=512,
        token_budget=4096,
        query_type=FeatureQueryType.CODE,
        tenant_tier=FeatureTenantTier.ENTERPRISE,
        latency_slo_ms=400,
        queue_depth=42,
        gpu_load=0.73,
        cache_hit_prob=0.45,
    )
    x = encode_feature_vector(fv)
    assert x.shape == (15,)
    assert x.dtype == np.float64


def test_ucb_score_hand_computed_identity_ridge() -> None:
    """Brand-new arm: θ=0, A=2I ⇒ UCB term = sqrt(1/2) for unit ``e₀`` and α_ucb=1."""

    arm = LinUCBArm(2.0)
    x = np.zeros(FEATURE_DIM, dtype=np.float64)
    x[0] = 1.0
    expected = math.sqrt(0.5)
    got = arm.ucb_score(x, 1.0)
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_updates_increase_relative_ucb_for_target_arm() -> None:
    """Heavy updates on arm 0 raise its UCB vs arm 1 for the same context."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=0.5, epsilon=0.0, rng=random.Random(0))
    fv = _fv()
    x = encode_feature_vector(fv)
    a0 = ALL_ACTIONS[0]
    for _ in range(80):
        b.update(a0, 5.0, fv)
    s0 = b._arms[0].ucb_score(x, b._ucb_alpha)
    s1 = b._arms[1].ucb_score(x, b._ucb_alpha)
    assert s0 > s1


def test_exploration_epsilon_forces_exploration_flag() -> None:
    """ε=1.0 always samples uniformly; ``exploration_flag`` and type ``epsilon``."""

    rng = random.Random(123)
    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=1.0, rng=rng)
    d = b.select_action(_fv())
    assert d.exploration_flag is True
    assert d.exploration_type is ExplorationType.EPSILON


def test_greedy_path_no_exploration() -> None:
    """ε=0 never explores; greedy equals chosen."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(0))
    d = b.select_action(_fv())
    assert d.exploration_flag is False
    assert d.exploration_type is ExplorationType.NONE
    assert d.chosen_action == d.greedy_action


def test_conservative_safety_gate_unknown_query_unconditional() -> None:
    """UNKNOWN forces conservative default; non-zero probes do not disable the gate."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=1.0, rng=random.Random(0))
    fv = FeatureVector(
        query_len=10,
        token_budget=100,
        query_type=FeatureQueryType.UNKNOWN,
        tenant_tier=FeatureTenantTier.FREE,
        latency_slo_ms=2000,
        queue_depth=50,
        gpu_load=0.85,
        cache_hit_prob=0.1,
    )
    d = b.select_action(fv)
    assert d.exploration_type is ExplorationType.FORCED
    assert d.exploration_flag is True
    assert d.chosen_action == conservative_default_action()
    assert d.greedy_action == conservative_default_action()
    assert len(d.action_scores) == ACTION_SPACE_SIZE


def test_internal_tier_forces_conservative_without_scoring() -> None:
    """INTERNAL tier bypasses scoring and always returns conservative default."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=1.0, rng=random.Random(0))
    fv = FeatureVector(
        query_len=5,
        token_budget=100,
        query_type=FeatureQueryType.RAG,
        tenant_tier=FeatureTenantTier.INTERNAL,
        latency_slo_ms=500,
        queue_depth=1,
        gpu_load=0.2,
        cache_hit_prob=0.0,
    )
    d = b.select_action(fv)
    assert d.chosen_action == conservative_default_action()
    assert d.greedy_action == conservative_default_action()
    assert d.exploration_flag is False
    assert d.exploration_type is ExplorationType.FORCED
    assert d.action_scores == {}


def test_action_scores_cover_all_canonical_keys() -> None:
    """Every arm must appear in ``action_scores`` for counterfactual logging."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(0))
    d = b.select_action(_fv())
    keys = set(d.action_scores)
    assert len(keys) == ACTION_SPACE_SIZE
    for a in ALL_ACTIONS:
        assert canonical_action_key(a) in keys


def test_save_load_round_trip_scores_and_version(tmp_path) -> None:
    """Atomic save/load preserves posterior and SHA-256 ``policy_version``."""

    path = tmp_path / "bandit.npz"
    rng = random.Random(7)
    b1 = LinUCBBandit(ridge_scale=1.5, ucb_alpha=0.8, epsilon=0.05, rng=rng)
    fv = _fv()
    b1.update(ALL_ACTIONS[3], 2.0, fv)
    v1 = b1.policy_version
    b1.save(path)
    raw = path.read_bytes()
    assert b1.policy_version == hashlib.sha256(raw).hexdigest()
    b2 = LinUCBBandit.load(path, rng=random.Random(9))
    assert b2.policy_version == v1
    x = encode_feature_vector(fv)
    for i, arm in enumerate(b1._arms):
        assert np.allclose(arm.copy_state()[0], b2._arms[i].copy_state()[0])
        assert np.allclose(arm.copy_state()[1], b2._arms[i].copy_state()[1])
        assert math.isclose(
            arm.ucb_score(x, b1._ucb_alpha),
            b2._arms[i].ucb_score(x, b2._ucb_alpha),
            rel_tol=1e-9,
        )


def test_policy_version_changes_after_update() -> None:
    """Posterior mutation must change the serialized fingerprint."""

    b = LinUCBBandit(ridge_scale=1.0, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(0))
    before = b.policy_version
    b.update(ALL_ACTIONS[10], 1.0, _fv())
    after = b.policy_version
    assert before != after


def test_thread_safety_parallel_updates_positive_definite() -> None:
    """Parallel updates keep each arm matrix symmetric positive definite."""

    b = LinUCBBandit(ridge_scale=0.5, ucb_alpha=1.0, epsilon=0.0, rng=random.Random(0))
    fv = _fv()
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        rng_local = random.Random()
        for _ in range(100):
            idx = rng_local.randrange(ACTION_SPACE_SIZE)
            b.update(ALL_ACTIONS[idx], rng_local.uniform(-1.0, 2.0), fv)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(worker) for _ in range(10)]
        for f in as_completed(futs):
            f.result()
    for arm in b._arms:
        assert arm.min_eigenvalue() > 0.0


def test_compute_online_reward_hand_computed() -> None:
    """Pure reward matches manual linear combination."""

    outcome = RequestOutcome(
        final_latency_ms=100.0,
        final_cost_usd=0.01,
        slo_violated=True,
        fallback_used=False,
        retry_count=2,
        cache_hit=False,
        latency_slo_ms=500.0,
    )
    p = DEFAULT_TENANT_POLICY
    expected = (
        -p.coeff_latency * 100.0
        - p.coeff_cost * 0.01
        - p.coeff_slo_violation * 1.0
        - p.coeff_fallback * 0.0
        - p.coeff_retry * 2.0
        + p.coeff_cache_hit * 0.0
    )
    assert math.isclose(compute_online_reward(outcome, p), expected, rel_tol=1e-12)


def test_reward_cache_hit_bonus_increases_value() -> None:
    """Identical outcome except ``cache_hit`` must score higher when hit is true."""

    base = RequestOutcome(
        final_latency_ms=50.0,
        final_cost_usd=0.0,
        slo_violated=False,
        fallback_used=False,
        retry_count=0,
        cache_hit=False,
        latency_slo_ms=1000.0,
    )
    hit = RequestOutcome(
        final_latency_ms=base.final_latency_ms,
        final_cost_usd=base.final_cost_usd,
        slo_violated=base.slo_violated,
        fallback_used=base.fallback_used,
        retry_count=base.retry_count,
        cache_hit=True,
        latency_slo_ms=base.latency_slo_ms,
    )
    p = DEFAULT_TENANT_POLICY
    assert compute_online_reward(hit, p) > compute_online_reward(base, p)


def test_reward_slo_violation_penalizes() -> None:
    """``slo_violated`` lowers reward versus identical compliant outcome."""

    good = RequestOutcome(
        final_latency_ms=40.0,
        final_cost_usd=0.0,
        slo_violated=False,
        fallback_used=False,
        retry_count=0,
        cache_hit=False,
        latency_slo_ms=1000.0,
    )
    bad = RequestOutcome(
        final_latency_ms=good.final_latency_ms,
        final_cost_usd=good.final_cost_usd,
        slo_violated=True,
        fallback_used=good.fallback_used,
        retry_count=good.retry_count,
        cache_hit=good.cache_hit,
        latency_slo_ms=good.latency_slo_ms,
    )
    p = DEFAULT_TENANT_POLICY
    assert compute_online_reward(bad, p) < compute_online_reward(good, p)


def test_joint_action_space_cardinality() -> None:
    """Section 3 joint space is 3⁴ arms."""

    assert ACTION_SPACE_SIZE == 81
    assert len(ALL_ACTIONS) == 81
    assert conservative_default_action() == ActionSpec(
        ModelTier.MEDIUM,
        Precision.FP16,
        RetrievalMode.CACHE_ONLY,
        OutputBudget.MEDIUM,
    )
