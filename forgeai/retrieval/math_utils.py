"""Pure numeric helpers for vector similarity (no I/O)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cosine_similarity(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
) -> float:
    """Return cosine similarity in ``[-1.0, 1.0]`` for non-zero vectors."""

    an = np.linalg.norm(a.astype(np.float64, copy=False))
    bn = np.linalg.norm(b.astype(np.float64, copy=False))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


__all__ = ["cosine_similarity"]
