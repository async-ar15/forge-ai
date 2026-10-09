"""Construction hooks for Qdrant and binary-safe async Redis (PCR embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from redis.asyncio import Redis

from forgeai.config import Settings


@dataclass(frozen=True, slots=True)
class RetrievalClients:
    """Qdrant + Redis ``decode_responses=False`` for float32 cache payloads."""

    qdrant: QdrantClient
    redis: Redis[Any]


def build_retrieval_clients(settings: Settings) -> RetrievalClients:
    """Instantiate retrieval clients from validated settings."""

    qdrant = QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key,
        timeout=max(1, int(settings.qdrant_timeout_seconds)),
    )
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    return RetrievalClients(qdrant=qdrant, redis=redis)


__all__ = ["RetrievalClients", "build_retrieval_clients"]
