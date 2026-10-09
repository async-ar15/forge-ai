"""Unit tests for ``cosine_similarity`` (known vectors)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from forgeai.retrieval.math_utils import cosine_similarity


def test_cosine_parallel() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite() -> None:
    a = np.array([1.0, 0.0], dtype=np.float64)
    b = np.array([-2.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_normalized_45_deg() -> None:
    a = np.array([1.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    expected = 1.0 / math.sqrt(2.0)
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_zero_vector() -> None:
    a = np.zeros(3, dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == 0.0
