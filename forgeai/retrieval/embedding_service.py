"""sentence_transformers is imported lazily inside EmbeddingService.__init__
to keep the retrieval package importable in environments where the package is
not installed. The ruff I001 suppression exists because the lazy import cannot
be at the top of the file by definition."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from forgeai.retrieval.constants import (
    EMBEDDING_DIM,
    EMBEDDING_LRU_MAXSIZE,
    EMBEDDING_MODEL_NAME,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class EmbeddingService:
    """``all-MiniLM-L6-v2`` with an in-process LRU keyed by normalized text."""

    __slots__ = ("_lru", "_model")

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        raw_dim = cast(Any, self._model).get_sentence_embedding_dimension()
        dim = EMBEDDING_DIM if raw_dim is None else int(raw_dim)
        if dim != EMBEDDING_DIM:
            msg = f"embedding dim {dim} != {EMBEDDING_DIM}"
            raise AssertionError(msg)
        self._lru: OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()

    def _normalize(self, text: str) -> str:
        return text.strip()

    def _encode_one(self, text: str) -> npt.NDArray[np.float32]:
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = np.asarray(vec, dtype=np.float32).reshape(-1)
        if out.shape != (EMBEDDING_DIM,):
            msg = f"bad embedding shape {out.shape}"
            raise AssertionError(msg)
        return out

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Return shape ``(EMBEDDING_DIM,)`` float32; LRU-cached per normalized text."""

        key = self._normalize(text)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        vec = self._encode_one(key)
        self._lru[key] = vec
        while len(self._lru) > EMBEDDING_LRU_MAXSIZE:
            self._lru.popitem(last=False)
        return vec

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return shape ``(N, EMBEDDING_DIM)`` float32."""

        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        normed = [self._normalize(t) for t in texts]
        mat = self._model.encode(
            normed,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = np.asarray(mat, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        if out.shape[-1] != EMBEDDING_DIM:
            msg = f"batch embedding dim {out.shape} invalid"
            raise AssertionError(msg)
        return out


__all__ = ["EmbeddingService"]
