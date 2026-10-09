"""Single LinUCB disjoint arm: ridge ``A``, reward accumulator ``b``, per-arm lock.

Does not own: 81-way orchestration, exploration, or persistence.
"""

from __future__ import annotations

import math
import threading
from typing import Final

import numpy as np
import numpy.typing as npt

from forgeai.policy.constants import FEATURE_DIM


class LinUCBArm:
    """One disjoint arm: PSD ``A`` (d×d), ``b`` (d,), per-arm ``threading.Lock``.

    Scores use ``numpy.linalg.solve`` (never ``inv``). All arrays are float64.
    """

    __slots__ = ("_A", "_b", "_lock", "_ridge_scale")

    def __init__(self, ridge_scale: float) -> None:
        """Initialize ``A = ridge_scale * I``, ``b = 0``. O(d²) memory, O(d²) time."""

        self._lock: Final[threading.Lock] = threading.Lock()
        self._ridge_scale: Final[float] = ridge_scale
        eye: npt.NDArray[np.float64] = np.eye(FEATURE_DIM, dtype=np.float64)
        # (FEATURE_DIM, FEATURE_DIM) ridge prior on contextual covariance.
        self._A: npt.NDArray[np.float64] = eye * float(ridge_scale)
        # (FEATURE_DIM,) cumulated reward-weighted features.
        self._b: npt.NDArray[np.float64] = np.zeros(FEATURE_DIM, dtype=np.float64)

    def ucb_score(self, x: npt.NDArray[np.float64], ucb_alpha: float) -> float:
        """LinUCB upper confidence value; O(d³); acquires this arm's lock."""

        with self._lock:
            return self._ucb_score_unlocked(x, ucb_alpha)

    def _ucb_score_unlocked(
        self, x: npt.NDArray[np.float64], ucb_alpha: float
    ) -> float:
        """UCB from ``A``, ``b``; internal — caller must hold ``_lock``."""

        # x: (FEATURE_DIM,), A: (FEATURE_DIM, FEATURE_DIM), b: (FEATURE_DIM,)
        theta = np.linalg.solve(self._A, self._b)
        pred: float = float(theta @ x)
        inv_v = np.linalg.solve(self._A, x)
        quad: float = float(x @ inv_v)
        width: float = math.sqrt(max(quad, 0.0))
        return pred + float(ucb_alpha) * width

    def update(self, x: npt.NDArray[np.float64], reward: float) -> None:
        """Ridge-style update ``A += xxᵀ``, ``b += r x``; O(d²); holds arm lock."""

        with self._lock:
            self._A += np.outer(x, x)
            self._b += float(reward) * x

    def copy_state(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return copies of ``(A, b)`` for persistence. O(d²); holds arm lock."""

        with self._lock:
            return self._A.copy(), self._b.copy()

    def restore_state(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> None:
        """Overwrite ``A`` and ``b`` from checkpoint arrays; O(d²); holds lock."""

        with self._lock:
            self._A[:, :] = A
            self._b[:] = b

    def min_eigenvalue(self) -> float:
        """Minimum eigenvalue of symmetrized ``A``; for PD regression tests."""

        with self._lock:
            sym = (self._A + self._A.T) * 0.5
            vals = np.linalg.eigvalsh(sym)
            return float(vals[0])
