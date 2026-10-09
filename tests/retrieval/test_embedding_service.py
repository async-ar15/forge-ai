"""``EmbeddingService`` shape, LRU identity, and dimension guards (mocked model)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from forgeai.retrieval.constants import EMBEDDING_DIM
from forgeai.retrieval.embedding_service import EmbeddingService


@contextmanager
def _fake_sentence_transformers(model_mock: MagicMock) -> object:
    """Fake ``sentence_transformers`` module (no package install in CI)."""

    mod = ModuleType("sentence_transformers")
    mod.SentenceTransformer = MagicMock(return_value=model_mock)
    with patch.dict(sys.modules, {"sentence_transformers": mod}):
        yield


@pytest.fixture
def mock_st() -> MagicMock:
    m = MagicMock()
    m.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
    return m


def test_embed_returns_shape_384_float32(mock_st: MagicMock) -> None:
    vec = np.arange(EMBEDDING_DIM, dtype=np.float32)
    mock_st.encode.return_value = vec
    with _fake_sentence_transformers(mock_st):
        svc = EmbeddingService()
        out = svc.embed("alpha")
    assert out.shape == (EMBEDDING_DIM,)
    assert out.dtype == np.float32


def test_embed_batch_shape_n_384(mock_st: MagicMock) -> None:
    mat = np.ones((4, EMBEDDING_DIM), dtype=np.float32)
    mock_st.encode.return_value = mat
    with _fake_sentence_transformers(mock_st):
        svc = EmbeddingService()
        out = svc.embed_batch(["a", "b", "c", "d"])
    assert out.shape == (4, EMBEDDING_DIM)
    assert out.dtype == np.float32


def test_embed_lru_same_text_same_object(mock_st: MagicMock) -> None:
    arr = np.linspace(0, 1, EMBEDDING_DIM, dtype=np.float32)
    mock_st.encode.return_value = arr
    with _fake_sentence_transformers(mock_st):
        svc = EmbeddingService()
        v1 = svc.embed("  shared  ")
        v2 = svc.embed("shared")
    assert v1 is v2


def test_embedding_dim_assert_on_wrong_model_dimension(mock_st: MagicMock) -> None:
    mock_st.get_sentence_embedding_dimension.return_value = 999
    with (
        _fake_sentence_transformers(mock_st),
        pytest.raises(AssertionError, match="embedding dim"),
    ):
        EmbeddingService()


def test_embed_batch_assert_wrong_last_dim(mock_st: MagicMock) -> None:
    mock_st.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
    mock_st.encode.return_value = np.zeros((2, 10), dtype=np.float32)
    with _fake_sentence_transformers(mock_st):
        svc = EmbeddingService()
    with pytest.raises(AssertionError, match="batch embedding dim"):
        svc.embed_batch(["x", "y"])
