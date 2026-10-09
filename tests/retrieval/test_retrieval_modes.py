"""Retrieval engine mode routing (Redis/Qdrant mocked)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from forgeai.config import Settings
from forgeai.proto import common_pb2, retrieval_service_pb2
from forgeai.retrieval.chunks import RetrievedChunk
from forgeai.retrieval.constants import EMBEDDING_DIM
from forgeai.retrieval.engine import RetrievalEngine


def _req(
    mode: int,
    *,
    text: str = "user query",
) -> retrieval_service_pb2.RetrievalRetrieveRequest:
    return retrieval_service_pb2.RetrievalRetrieveRequest(
        correlation_request_id="c1",
        tenant_id="tenant-1",
        retrieval_mode=mode,
        retrieval_query_text_utf8=text,
    )


def _engine(settings: Settings) -> tuple[RetrievalEngine, MagicMock, AsyncMock]:
    emb = MagicMock()
    vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
    vec /= float(np.linalg.norm(vec))
    emb.embed.return_value = vec
    cache = AsyncMock()
    bloom = AsyncMock()
    qdrant = MagicMock()
    redis = AsyncMock()
    eng = RetrievalEngine(
        settings=settings,
        qdrant=qdrant,
        redis=redis,
        embedding=emb,
        semantic_cache=cache,
        bloom=bloom,
    )
    return eng, emb, cache


@pytest.mark.asyncio
async def test_off_no_backend_calls(retrieval_settings: Settings) -> None:
    eng, emb, cache = _engine(retrieval_settings)
    resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_OFF))
    assert resp.context_passage_utf8 == []
    assert resp.cache_hit is False
    emb.embed.assert_not_called()
    cache.lookup_best.assert_not_called()


@pytest.mark.asyncio
async def test_cache_only_hit_no_qdrant(retrieval_settings: Settings) -> None:
    eng, _emb, cache = _engine(retrieval_settings)
    chunks = [RetrievedChunk("a", "txt", 0.88, {})]
    cache.lookup_best = AsyncMock(return_value=(chunks, 0.99, "e1"))
    with patch("forgeai.retrieval.engine.search_chunks") as sq:
        resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_CACHE_ONLY))
    sq.assert_not_called()
    assert resp.cache_hit is True
    assert len(resp.context_passage_utf8) == 1


@pytest.mark.asyncio
async def test_cache_only_miss_semantic_flag(
    retrieval_settings: Settings,
) -> None:
    eng, _emb, cache = _engine(retrieval_settings)
    cache.lookup_best = AsyncMock(return_value=None)
    with patch("forgeai.retrieval.engine.search_chunks") as sq:
        resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_CACHE_ONLY))
    sq.assert_not_called()
    assert resp.cache_hit is False
    assert resp.context_passage_utf8 == []
    assert resp.semantic_cache_miss is True


@pytest.mark.asyncio
async def test_full_hit_no_qdrant_no_prefetch(retrieval_settings: Settings) -> None:
    eng, _emb, cache = _engine(retrieval_settings)
    chunks = [RetrievedChunk("z", "ok", 0.91, {})]
    cache.lookup_best = AsyncMock(return_value=(chunks, 0.93, "e2"))
    with (
        patch("forgeai.retrieval.engine.search_chunks") as sq,
        patch("forgeai.retrieval.engine.schedule_prefetch") as sp,
    ):
        resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_FULL))
    sq.assert_not_called()
    sp.assert_not_called()
    assert resp.cache_hit is True


@pytest.mark.asyncio
async def test_full_miss_qdrant_cache_prefetch(retrieval_settings: Settings) -> None:
    eng, _emb, cache = _engine(retrieval_settings)
    cache.lookup_best = AsyncMock(return_value=None)
    got = [RetrievedChunk("q", "chunk body", 0.7, {"m": 1})]
    with (
        patch("forgeai.retrieval.engine.search_chunks", return_value=(got, False)),
        patch("forgeai.retrieval.engine.schedule_prefetch") as sp,
    ):
        resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_FULL))
    cache.store_entry.assert_awaited_once()
    assert sp.called
    assert resp.cache_hit is False
    assert len(resp.context_passage_utf8) == 1


@pytest.mark.asyncio
async def test_full_qdrant_error_no_exception(retrieval_settings: Settings) -> None:
    eng, _emb, cache = _engine(retrieval_settings)
    cache.lookup_best = AsyncMock(return_value=None)
    with patch("forgeai.retrieval.engine.search_chunks", return_value=([], True)):
        resp = await eng.retrieve(_req(common_pb2.RETRIEVAL_MODE_FULL))
    assert resp.qdrant_error is True
    assert resp.context_passage_utf8 == []


def test_prefetch_budget_fourth_dropped_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import forgeai.retrieval.prefetch_scheduler as ps

    caplog.set_level(logging.DEBUG, logger="forgeai.retrieval.prefetch_scheduler")
    sem = MagicMock()
    sem.acquire.side_effect = [True, True, True, False]
    sem.release = MagicMock()
    with (
        patch.object(ps, "_sem", sem),
        patch.object(ps.threading, "Thread") as t_cls,
    ):

        async def _noop() -> None:
            await asyncio.sleep(0)

        for _ in range(4):
            ps.schedule_prefetch(lambda: _noop())
    assert t_cls.call_count == 3
    assert "prefetch_dropped" in caplog.text
