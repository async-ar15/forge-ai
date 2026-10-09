"""PCR semantic cache: Redis-backed embedding similarity (pre-Qdrant gate)."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from redis.asyncio import Redis

from forgeai.retrieval.chunks import RetrievedChunk
from forgeai.retrieval.constants import (
    CACHE_CANDIDATE_ZSET_SCAN_LIMIT,
    CACHE_SIMILARITY_THRESHOLD,
    EMBEDDING_DIM,
    SEMANTIC_CACHE_HITS_ZSET_SUFFIX,
    SEMANTIC_CACHE_KEY_PREFIX,
)
from forgeai.retrieval.math_utils import cosine_similarity

_F_EMB: Final[str] = "emb"
_F_CHUNKS: Final[str] = "chunks"
_F_QUERY: Final[str] = "query"
_F_TS: Final[str] = "ts"
_F_HITS: Final[str] = "hits"


def _hits_zkey(tenant_id: str) -> str:
    return f"{SEMANTIC_CACHE_KEY_PREFIX}:{tenant_id}:{SEMANTIC_CACHE_HITS_ZSET_SUFFIX}"


def _entry_key(tenant_id: str, entry_id: str) -> str:
    return f"{SEMANTIC_CACHE_KEY_PREFIX}:{tenant_id}:e:{entry_id}"


def _chunks_from_json(raw: bytes) -> list[RetrievedChunk]:
    data = json.loads(raw.decode("utf-8"))
    return [
        RetrievedChunk(
            chunk_id=str(c["chunk_id"]),
            text=str(c["text"]),
            score=float(c["score"]),
            metadata=dict(c.get("metadata", {})),
        )
        for c in data
    ]


def _chunks_to_json(chunks: list[RetrievedChunk]) -> bytes:
    payload = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "score": c.score,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    dumped = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return dumped.encode("utf-8")


class SemanticCache:
    """Tenant-scoped Redis hashes + hit-count zset for PCR."""

    __slots__ = ("_redis", "_ttl_seconds")

    def __init__(self, redis: Redis[Any], *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def _load_candidate(
        self,
        tenant_id: str,
        entry_id: str,
        query_emb: npt.NDArray[np.float32],
    ) -> tuple[float, list[RetrievedChunk]] | None:
        hk = _entry_key(tenant_id, entry_id)
        raw_emb = await self._redis.hget(hk, _F_EMB)
        if raw_emb is None:
            return None
        emb = np.frombuffer(raw_emb, dtype=np.float32)
        if emb.shape != (EMBEDDING_DIM,):
            return None
        sim = cosine_similarity(query_emb, emb)
        raw_chunks = await self._redis.hget(hk, _F_CHUNKS)
        if raw_chunks is None:
            return None
        chunks = _chunks_from_json(raw_chunks)
        return sim, chunks

    async def lookup_best(
        self,
        tenant_id: str,
        query_emb: npt.NDArray[np.float32],
    ) -> tuple[list[RetrievedChunk], float, str] | None:
        """Return cached chunks, similarity, entry id if >= threshold."""

        zkey = _hits_zkey(tenant_id)
        ids = await self._redis.zrevrange(
            zkey,
            0,
            CACHE_CANDIDATE_ZSET_SCAN_LIMIT - 1,
        )
        best: tuple[float, list[RetrievedChunk], str] | None = None
        for raw_id in ids:
            eid = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            loaded = await self._load_candidate(tenant_id, eid, query_emb)
            if loaded is None:
                continue
            sim, chunks = loaded
            if best is None or sim > best[0]:
                best = (sim, chunks, eid)
        if best is None or best[0] < CACHE_SIMILARITY_THRESHOLD:
            return None
        sim, chunks, eid = best
        return (chunks, float(sim), eid)

    async def increment_hit_count(self, tenant_id: str, entry_id: str) -> None:
        """Bump hit_count and zset score for PCR prioritization."""

        hk = _entry_key(tenant_id, entry_id)
        zkey = _hits_zkey(tenant_id)
        await self._redis.hincrby(hk, _F_HITS, 1)
        nh = await self._redis.hget(hk, _F_HITS)
        n = int(nh) if nh else 1
        await self._redis.zadd(zkey, {entry_id: float(n)})

    async def store_entry(
        self,
        tenant_id: str,
        query_text: str,
        query_emb: npt.NDArray[np.float32],
        chunks: list[RetrievedChunk],
    ) -> str:
        """Persist embedding + chunks; register in hit zset; set TTL."""

        eid = str(uuid.uuid4())
        hk = _entry_key(tenant_id, eid)
        zkey = _hits_zkey(tenant_id)
        emb_b = np.asarray(query_emb, dtype=np.float32).tobytes()
        pipe = self._redis.pipeline()
        pipe.hset(
            hk,
            mapping={
                _F_EMB: emb_b,
                _F_CHUNKS: _chunks_to_json(chunks),
                _F_QUERY: query_text.encode("utf-8"),
                _F_TS: str(time.time()).encode("ascii"),
                _F_HITS: b"1",
            },
        )
        pipe.expire(hk, self._ttl_seconds)
        pipe.zadd(zkey, {eid: 1.0})
        await pipe.execute()
        return eid

    async def refresh_ttl(self, tenant_id: str, entry_id: str) -> None:
        """PCR prefetch: extend TTL for a hot entry."""

        hk = _entry_key(tenant_id, entry_id)
        await self._redis.expire(hk, self._ttl_seconds)

    async def list_candidate_ids(self, tenant_id: str, limit: int) -> list[str]:
        zkey = _hits_zkey(tenant_id)
        ids = await self._redis.zrevrange(zkey, 0, max(0, limit - 1))
        out: list[str] = []
        for raw_id in ids:
            sid = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            out.append(sid)
        return out

    async def load_embedding_for_entry(
        self,
        tenant_id: str,
        entry_id: str,
    ) -> npt.NDArray[np.float32] | None:
        hk = _entry_key(tenant_id, entry_id)
        raw = await self._redis.hget(hk, _F_EMB)
        if raw is None:
            return None
        emb = np.frombuffer(raw, dtype=np.float32)
        if emb.shape != (EMBEDDING_DIM,):
            return None
        return emb


__all__ = ["SemanticCache"]
