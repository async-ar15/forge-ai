"""Qdrant vector search with explicit error containment."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Final, cast

import numpy.typing as npt
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from forgeai.retrieval.chunks import RetrievedChunk
from forgeai.retrieval.constants import QDRANT_DEFAULT_TOP_K

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def search_chunks(
    client: QdrantClient,
    *,
    collection: str,
    vector: npt.NDArray[Any],
    top_k: int,
) -> tuple[list[RetrievedChunk], bool]:
    """Return chunks and ``qdrant_error`` flag (True on failure)."""

    lim = top_k if top_k > 0 else QDRANT_DEFAULT_TOP_K
    vec_list = [float(x) for x in vector.reshape(-1)]
    try:
        res = cast(Any, client).search(
            collection_name=collection,
            query_vector=vec_list,
            limit=lim,
            with_payload=True,
            search_params=qm.SearchParams(hnsw_ef=128),
        )
    except (UnexpectedResponse, ResponseHandlingException) as exc:
        _LOG.error(
            "qdrant_search_failed collection=%s err=%s",
            collection,
            exc,
            exc_info=True,
        )
        return [], True
    except Exception as exc:
        _LOG.error(
            "qdrant_search_failed collection=%s err=%s",
            collection,
            exc,
            exc_info=True,
        )
        return [], True
    chunks: list[RetrievedChunk] = []
    for hit in res:
        pid = hit.payload or {}
        cid = str(pid.get("chunk_id", str(uuid.uuid4())))
        text = str(pid.get("text", ""))
        meta = {k: v for k, v in pid.items() if k not in ("chunk_id", "text")}
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=text,
                score=float(hit.score),
                metadata=meta,
            ),
        )
    return chunks, False


__all__ = ["search_chunks"]
