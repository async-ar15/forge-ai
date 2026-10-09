"""Orchestrates OFF / cache_only / full retrieval modes (PCR + Qdrant)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt
from qdrant_client import QdrantClient
from redis.asyncio import Redis

from forgeai.config import Settings
from forgeai.enums import RetrievalMode
from forgeai.observability.metrics import (
    RETRIEVAL_CACHE_HIT_TOTAL,
    RETRIEVAL_QDRANT_ERRORS_TOTAL,
    RETRIEVAL_REQUEST_DURATION_SECONDS,
)
from forgeai.proto import common_pb2, retrieval_service_pb2
from forgeai.retrieval.bloom_filter import BloomFilterService
from forgeai.retrieval.chunks import RetrievedChunk, chunk_to_passage_utf8
from forgeai.retrieval.clients import RetrievalClients
from forgeai.retrieval.constants import (
    CACHE_CANDIDATE_ZSET_SCAN_LIMIT,
    PREFETCH_TOP_SIMILAR_ENTRIES,
    QDRANT_DEFAULT_TOP_K,
)
from forgeai.retrieval.embedding_service import EmbeddingService
from forgeai.retrieval.math_utils import cosine_similarity
from forgeai.retrieval.prefetch_scheduler import schedule_prefetch
from forgeai.retrieval.qdrant_search import search_chunks
from forgeai.retrieval.semantic_cache import SemanticCache

SemanticHit: TypeAlias = tuple[list[RetrievedChunk], float, str]

_PROTO_MODE: Final[dict[int, RetrievalMode]] = {
    int(common_pb2.RETRIEVAL_MODE_OFF): RetrievalMode.OFF,
    int(common_pb2.RETRIEVAL_MODE_CACHE_ONLY): RetrievalMode.CACHE_ONLY,
    int(common_pb2.RETRIEVAL_MODE_FULL): RetrievalMode.FULL,
}


def _mode_from_proto(v: int) -> RetrievalMode:
    return _PROTO_MODE.get(v, RetrievalMode.OFF)


def _finalize_metrics(mode_l: str, cache_hit: bool, t0: float) -> int:
    dt = time.perf_counter() - t0
    ms = max(0, int(dt * 1000))
    ch = "true" if cache_hit else "false"
    RETRIEVAL_REQUEST_DURATION_SECONDS.labels(
        retrieval_mode=mode_l,
        cache_hit=ch,
    ).observe(dt)
    RETRIEVAL_CACHE_HIT_TOTAL.labels(
        retrieval_mode=mode_l,
    ).inc()
    return ms


def _response(
    passages: list[str],
    cache_hit: bool,
    latency_ms: int,
    *,
    sim: float | None = None,
    qdrant_error: bool = False,
    semantic_miss: bool = False,
) -> retrieval_service_pb2.RetrievalRetrieveResponse:
    r = retrieval_service_pb2.RetrievalRetrieveResponse(
        context_passage_utf8=passages,
        cache_hit=cache_hit,
        retrieval_internal_latency_ms=latency_ms,
    )
    if sim is not None:
        r.cache_similarity_score = float(sim)
    if qdrant_error:
        r.qdrant_error = True
    if semantic_miss:
        r.semantic_cache_miss = True
    return r


@dataclass(slots=True)
class RetrievalEngine:
    """Process-wide retrieval dependencies (constructed at server startup)."""

    settings: Settings
    qdrant: QdrantClient
    redis: Redis[Any]
    embedding: EmbeddingService
    semantic_cache: SemanticCache
    bloom: BloomFilterService

    def _top_k(self, req: retrieval_service_pb2.RetrievalRetrieveRequest) -> int:
        n = int(req.max_context_passage_count)
        return n if n > 0 else QDRANT_DEFAULT_TOP_K

    async def _prefetch_coroutine(
        self,
        tenant_id: str,
        chunk_texts: list[str],
    ) -> None:
        if not chunk_texts:
            return
        mat = self.embedding.embed_batch(chunk_texts)
        mean = mat.mean(axis=0).astype("float32")
        cands = await self.semantic_cache.list_candidate_ids(
            tenant_id,
            CACHE_CANDIDATE_ZSET_SCAN_LIMIT,
        )
        scored: list[tuple[float, str]] = []
        for eid in cands:
            ev = await self.semantic_cache.load_embedding_for_entry(tenant_id, eid)
            if ev is None:
                continue
            scored.append((cosine_similarity(mean, ev), eid))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _s, eid in scored[:PREFETCH_TOP_SIMILAR_ENTRIES]:
            await self.semantic_cache.refresh_ttl(tenant_id, eid)

    async def retrieve(
        self,
        req: retrieval_service_pb2.RetrievalRetrieveRequest,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        """Unary retrieval (PCR + optional Qdrant).

        Performance contracts (wall time, warm Redis/Qdrant):
        * ``off``: <1 ms — no backend I/O.
        * ``cache_only`` hit / ``full`` hit: <10 ms — semantic cache only.
        * ``cache_only`` miss: <5 ms — candidate scan only.
        * ``full`` miss: <50 ms — Qdrant round-trip + cache write + prefetch schedule.
        """

        t0 = time.perf_counter()
        mode = _mode_from_proto(int(req.retrieval_mode))
        mode_l = mode.value
        if mode is RetrievalMode.OFF:
            ms = _finalize_metrics(mode_l, False, t0)
            return _response([], False, ms, semantic_miss=False)

        emb = self.embedding.embed(req.retrieval_query_text_utf8)
        if mode is RetrievalMode.CACHE_ONLY:
            return await self._cache_only(req, emb, t0, mode_l)
        return await self._full(req, emb, t0, mode_l)

    async def _cache_only(
        self,
        req: retrieval_service_pb2.RetrievalRetrieveRequest,
        emb: npt.NDArray[np.float32],
        t0: float,
        mode_l: str,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        tenant = req.tenant_id
        qtext = req.retrieval_query_text_utf8
        hit = await self.semantic_cache.lookup_best(tenant, emb)
        if hit is None:
            await self.bloom.record_query(qtext)
            ms = _finalize_metrics(mode_l, False, t0)
            return _response([], False, ms, semantic_miss=True)
        chunks, sim, eid = hit
        await self.semantic_cache.increment_hit_count(tenant, eid)
        await self.bloom.record_query(qtext)
        ms = _finalize_metrics(mode_l, True, t0)
        ps = [chunk_to_passage_utf8(c) for c in chunks]
        return _response(ps, True, ms, sim=sim)

    async def _full(
        self,
        req: retrieval_service_pb2.RetrievalRetrieveRequest,
        emb: npt.NDArray[np.float32],
        t0: float,
        mode_l: str,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        tenant = req.tenant_id
        qtext = req.retrieval_query_text_utf8
        hit = await self.semantic_cache.lookup_best(tenant, emb)
        if hit is not None:
            return await self._full_semantic_hit(req, hit, qtext, t0, mode_l)
        return await self._full_semantic_miss(req, emb, qtext, tenant, t0, mode_l)

    async def _full_semantic_hit(
        self,
        req: retrieval_service_pb2.RetrievalRetrieveRequest,
        hit: SemanticHit,
        qtext: str,
        t0: float,
        mode_l: str,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        chunks, sim, eid = hit
        await self.semantic_cache.increment_hit_count(req.tenant_id, eid)
        await self.bloom.record_query(qtext)
        ms = _finalize_metrics(mode_l, True, t0)
        ps = [chunk_to_passage_utf8(c) for c in chunks]
        return _response(ps, True, ms, sim=sim)

    async def _full_semantic_miss(
        self,
        req: retrieval_service_pb2.RetrievalRetrieveRequest,
        emb: npt.NDArray[np.float32],
        qtext: str,
        tenant: str,
        t0: float,
        mode_l: str,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        chunks, qerr = search_chunks(
            self.qdrant,
            collection=self.settings.qdrant_collection_name,
            vector=emb,
            top_k=self._top_k(req),
        )
        if qerr:
            RETRIEVAL_QDRANT_ERRORS_TOTAL.inc()
            await self.bloom.record_query(qtext)
            ms = _finalize_metrics(mode_l, False, t0)
            return _response([], False, ms, qdrant_error=True, semantic_miss=True)
        await self.semantic_cache.store_entry(tenant, qtext, emb, chunks)
        await self.bloom.record_query(qtext)
        texts = [c.text for c in chunks if c.text]
        schedule_prefetch(lambda: self._prefetch_coroutine(tenant, texts))
        ms = _finalize_metrics(mode_l, False, t0)
        ps = [chunk_to_passage_utf8(c) for c in chunks]
        return _response(ps, False, ms, semantic_miss=False)


def build_engine(settings: Settings, clients: RetrievalClients) -> RetrievalEngine:
    """Construct a ``RetrievalEngine`` from shared clients."""

    emb = EmbeddingService()
    cache = SemanticCache(clients.redis, ttl_seconds=settings.cache_ttl_seconds)
    bloom = BloomFilterService(
        clients.redis,
        bloom_key=settings.retrieval_bloom_filter_key,
    )
    return RetrievalEngine(
        settings=settings,
        qdrant=clients.qdrant,
        redis=clients.redis,
        embedding=emb,
        semantic_cache=cache,
        bloom=bloom,
    )


__all__ = ["RetrievalEngine", "build_engine"]
