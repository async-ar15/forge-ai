"""Semantic cache Redis interactions (fully mocked)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from forgeai.retrieval.chunks import RetrievedChunk
from forgeai.retrieval.constants import EMBEDDING_DIM
from forgeai.retrieval.semantic_cache import SemanticCache


def _norm_emb() -> np.ndarray:
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _chunks_json(chunks: list[RetrievedChunk]) -> bytes:
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


@pytest.mark.asyncio
async def test_lookup_hit_above_threshold_returns_chunks() -> None:
    emb = _norm_emb()
    chunks = [
        RetrievedChunk("c1", "hello", 0.9, {"src": "t"}),
    ]
    raw_c = _chunks_json(chunks)
    r = AsyncMock()
    r.zrevrange = AsyncMock(return_value=[b"e1"])
    r.hget = AsyncMock(side_effect=[emb.tobytes(), raw_c])
    cache = SemanticCache(r, ttl_seconds=3600)
    hit = await cache.lookup_best("tenant-a", emb)
    assert hit is not None
    out_chunks, sim, eid = hit
    assert eid == "e1"
    assert sim >= 0.92
    assert len(out_chunks) == 1
    assert out_chunks[0].text == "hello"


@pytest.mark.asyncio
async def test_lookup_miss_below_threshold() -> None:
    q = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    q[0] = 1.0
    stored = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    stored[1] = 1.0
    chunks = [RetrievedChunk("c1", "x", 0.1, {})]
    r = AsyncMock()
    r.zrevrange = AsyncMock(return_value=[b"e1"])
    r.hget = AsyncMock(side_effect=[stored.tobytes(), _chunks_json(chunks)])
    cache = SemanticCache(r, ttl_seconds=3600)
    assert await cache.lookup_best("t", q) is None


@pytest.mark.asyncio
async def test_store_entry_writes_fields_and_ttl() -> None:
    r = AsyncMock()
    pipe = MagicMock()
    pipe.hset = MagicMock()
    pipe.expire = MagicMock()
    pipe.zadd = MagicMock()
    pipe.execute = AsyncMock(return_value=None)
    r.pipeline = MagicMock(return_value=pipe)
    cache = SemanticCache(r, ttl_seconds=1234)
    emb = _norm_emb()
    chunks = [RetrievedChunk("id", "body", 0.5, {"k": "v"})]
    eid = await cache.store_entry("tenant-x", "original query", emb, chunks)
    assert len(eid) > 0
    pipe.hset.assert_called_once()
    call_kw = pipe.hset.call_args.kwargs["mapping"]
    assert call_kw["emb"] == emb.astype(np.float32).tobytes()
    assert call_kw["query"] == b"original query"
    pipe.expire.assert_called_once()
    _hk, ttl = pipe.expire.call_args[0]
    assert ttl == 1234
    pipe.zadd.assert_called_once()


@pytest.mark.asyncio
async def test_increment_hit_count_bumps_redis() -> None:
    r = AsyncMock()
    r.hincrby = AsyncMock(return_value=2)
    r.hget = AsyncMock(return_value=b"3")
    r.zadd = AsyncMock()
    cache = SemanticCache(r, ttl_seconds=3600)
    await cache.increment_hit_count("tenant-z", "e99")
    r.hincrby.assert_awaited_once()
    r.zadd.assert_awaited_once()
