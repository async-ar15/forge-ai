"""Disjoint LinUCB over 81 joint actions (Section 3 policy engine).

Does not own: feature extraction, Postgres routing_decisions writes, or execution.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from forgeai.enums import (
    ExplorationType,
    ModelTier,
    OutputBudget,
    Precision,
    RetrievalMode,
)
from forgeai.policy.bandit_actions import (
    ACTION_SPACE_SIZE,
    ALL_ACTIONS,
    ActionSpec,
    action_index,
    canonical_action_key,
)
from forgeai.policy.bandit_arm import LinUCBArm
from forgeai.policy.bandit_encoding import encode_feature_vector
from forgeai.policy.bandit_metrics import record_bandit_update
from forgeai.policy.constants import (
    DEFAULT_LINUCB_ALPHA_STR,
    DEFAULT_LINUCB_EXPLORATION_EPSILON_STR,
    FeatureQueryType,
    FeatureTenantTier,
    LinUCBConfigKey,
)
from forgeai.policy.features import FeatureVector

_CONSERVATIVE_ACTION: Final[ActionSpec] = ActionSpec(
    model_tier=ModelTier.MEDIUM,
    precision=Precision.FP16,
    retrieval_mode=RetrievalMode.CACHE_ONLY,
    output_budget=OutputBudget.MEDIUM,
)


def conservative_default_action() -> ActionSpec:
    """Section 3 conservative joint action when the unknown-query safety gate fires."""

    return _CONSERVATIVE_ACTION


def _unknown_query_safety_gate(features: FeatureVector) -> bool:
    """Whether to bypass the bandit and return the conservative default (v1).

    Conservative default fires on ``query_type == UNKNOWN`` unconditionally —
    probe values are not a reliable cold-start signal (e.g. ``gpu_load`` may read
    0.0 between probe cycles on a loaded system).
    """

    return features.query_type is FeatureQueryType.UNKNOWN


def _serialize_arms(
    arms: tuple[LinUCBArm, ...],
    ridge_scale: float,
    ucb_alpha: float,
    epsilon: float,
) -> bytes:
    """Serialize every arm's ``A`` and ``b`` into one compressed ``npz`` byte blob."""

    a_list: list[npt.NDArray[np.float64]] = []
    b_list: list[npt.NDArray[np.float64]] = []
    for arm in arms:
        a_i, b_i = arm.copy_state()
        a_list.append(a_i)
        b_list.append(b_i)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        ridge_scale=np.float64(ridge_scale),
        ucb_alpha=np.float64(ucb_alpha),
        epsilon=np.float64(epsilon),
        A_stack=np.stack(a_list, axis=0),
        b_stack=np.stack(b_list, axis=0),
    )
    return buf.getvalue()


def _sha256_hex(data: bytes) -> str:
    """Hex digest of SHA-256 over ``data`` (policy artifact fingerprint)."""

    return hashlib.sha256(data).hexdigest()


def _resolve_hparams(
    ridge_scale: float | None,
    ucb_alpha: float | None,
    epsilon: float | None,
) -> tuple[float, float, float]:
    """Read LinUCB hyperparameters from arguments or environment."""

    rs = (
        float(ridge_scale)
        if ridge_scale is not None
        else float(os.environ.get(LinUCBConfigKey.ALPHA, DEFAULT_LINUCB_ALPHA_STR))
    )
    ua = (
        float(ucb_alpha)
        if ucb_alpha is not None
        else float(os.environ.get(LinUCBConfigKey.ALPHA, DEFAULT_LINUCB_ALPHA_STR))
    )
    eps = (
        float(epsilon)
        if epsilon is not None
        else float(
            os.environ.get(
                LinUCBConfigKey.EXPLORATION_EPSILON,
                DEFAULT_LINUCB_EXPLORATION_EPSILON_STR,
            )
        )
    )
    return rs, ua, eps


@dataclass(frozen=True, slots=True)
class BanditDecision:
    """One routing decision for gRPC and ``routing_decisions`` logging."""

    chosen_action: ActionSpec
    greedy_action: ActionSpec
    action_scores: dict[str, float]
    exploration_flag: bool
    exploration_type: ExplorationType


class LinUCBBandit:
    """81-arm disjoint LinUCB: scores, epsilon-greedy, persistence, per-arm locks."""

    __slots__ = (
        "_arms",
        "_epsilon",
        "_rng",
        "_ridge_scale",
        "_ucb_alpha",
        "_version_bytes",
        "_version_lock",
    )

    def __init__(
        self,
        *,
        ridge_scale: float | None = None,
        ucb_alpha: float | None = None,
        epsilon: float | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Parse hyperparameters from kwargs or ``LINUCB_*`` environment variables."""

        rs, ua, eps = _resolve_hparams(ridge_scale, ucb_alpha, epsilon)
        self._ridge_scale = rs
        self._ucb_alpha = ua
        self._epsilon = eps
        self._rng = rng if rng is not None else random.Random()
        self._arms = tuple(LinUCBArm(rs) for _ in range(ACTION_SPACE_SIZE))
        self._version_lock = threading.Lock()
        self._version_bytes: bytes | None = _serialize_arms(self._arms, rs, ua, eps)

    @property
    def policy_version(self) -> str:
        """SHA-256 hex of the canonical ``npz`` blob matching ``save`` output."""

        with self._version_lock:
            blob = self._version_bytes
            if blob is None:
                blob = _serialize_arms(
                    self._arms,
                    self._ridge_scale,
                    self._ucb_alpha,
                    self._epsilon,
                )
                self._version_bytes = blob
            return _sha256_hex(blob)

    def _invalidate_version(self) -> None:
        """Mark version stale after a mutating update."""

        with self._version_lock:
            self._version_bytes = None

    def _score_all(self, x: npt.NDArray[np.float64]) -> dict[str, float]:
        """UCB score for every arm; O(arms·d³); each arm uses its own lock."""

        return {
            canonical_action_key(action): arm.ucb_score(x, self._ucb_alpha)
            for action, arm in zip(ALL_ACTIONS, self._arms, strict=True)
        }

    def select_action(self, features: FeatureVector) -> BanditDecision:
        """Epsilon-greedy UCB; forced gate skips selection; scores every arm."""

        if features.tenant_tier is FeatureTenantTier.INTERNAL:
            act = conservative_default_action()
            return BanditDecision(
                chosen_action=act,
                greedy_action=act,
                action_scores={},
                exploration_flag=False,
                exploration_type=ExplorationType.FORCED,
            )
        x = encode_feature_vector(features)
        scores = self._score_all(x)
        if _unknown_query_safety_gate(features):
            act = conservative_default_action()
            return BanditDecision(
                chosen_action=act,
                greedy_action=act,
                action_scores=scores,
                exploration_flag=True,
                exploration_type=ExplorationType.FORCED,
            )
        greedy_action = max(ALL_ACTIONS, key=lambda a: scores[canonical_action_key(a)])
        if self._rng.random() < self._epsilon:
            chosen = ALL_ACTIONS[self._rng.randrange(ACTION_SPACE_SIZE)]
            return BanditDecision(
                chosen_action=chosen,
                greedy_action=greedy_action,
                action_scores=scores,
                exploration_flag=True,
                exploration_type=ExplorationType.EPSILON,
            )
        return BanditDecision(
            chosen_action=greedy_action,
            greedy_action=greedy_action,
            action_scores=scores,
            exploration_flag=False,
            exploration_type=ExplorationType.NONE,
        )

    def update(
        self, action: ActionSpec, reward: float, features: FeatureVector
    ) -> None:
        """Update one arm; increment metrics; defer ``policy_version`` recomputation."""

        x = encode_feature_vector(features)
        self._arms[action_index(action)].update(x, float(reward))
        record_bandit_update(action_key=canonical_action_key(action))
        self._invalidate_version()

    def save(self, path: str | Path) -> None:
        """Atomic ``npz`` write; cached version bytes match on-disk artifact."""

        p = Path(path)
        data = _serialize_arms(
            self._arms,
            self._ridge_scale,
            self._ucb_alpha,
            self._epsilon,
        )
        with self._version_lock:
            self._version_bytes = data
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    @classmethod
    def load(
        cls, path: str | Path, *, rng: random.Random | None = None
    ) -> LinUCBBandit:
        """Restore from ``npz``; ``policy_version`` equals SHA-256 of file bytes."""

        p = Path(path)
        raw = p.read_bytes()
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
        rs = float(loaded["ridge_scale"])
        ua = float(loaded["ucb_alpha"])
        eps = float(loaded["epsilon"])
        A_stack = loaded["A_stack"]
        b_stack = loaded["b_stack"]
        bandit = cls(ridge_scale=rs, ucb_alpha=ua, epsilon=eps, rng=rng)
        for i, arm in enumerate(bandit._arms):
            arm.restore_state(A_stack[i], b_stack[i])
        with bandit._version_lock:
            bandit._version_bytes = raw
        return bandit


LinUCBPolicy = LinUCBBandit
RoutingAction = ActionSpec

__all__ = [
    "ActionSpec",
    "BanditDecision",
    "LinUCBBandit",
    "LinUCBPolicy",
    "RoutingAction",
    "conservative_default_action",
]
