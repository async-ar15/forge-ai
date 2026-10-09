"""Redis Bloom filter for query fingerprinting (feature path + retrieval recording)."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Final, cast

from redis.asyncio import Redis

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def bloom_item_from_query(query_text: str) -> str:
    """Stable item key (hex digest) aligned with gateway feature extraction."""

    normalized = query_text.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BloomFeatureRedisAdapter:
    """Implements ``AsyncBloomRedisClient`` for ``FeatureExtractor`` injection."""

    __slots__ = ("_r",)

    def __init__(self, redis: Redis[Any]) -> None:
        self._r = redis

    async def execute_command(self, *args: object, **kwargs: object) -> object:
        return await cast(Any, self._r).execute_command(*args, **kwargs)


class BloomFilterService:
    """BF.ADD after retrieval; BF.EXISTS for bandit cache-hit probe."""

    __slots__ = ("_key", "_r")

    def __init__(self, redis: Redis[Any], *, bloom_key: str) -> None:
        self._r = redis
        self._key = bloom_key

    async def record_query(self, query_text: str) -> None:
        """Record normalized query fingerprint (never raises to callers)."""

        item = bloom_item_from_query(query_text)
        try:
            await cast(Any, self._r).execute_command("BF.ADD", self._key, item)
        except Exception:
            _LOG.warning("bloom_add_failed key=%s", self._key, exc_info=True)

    async def check_query(self, query_text: str) -> bool:
        """Return existence bit; swallow Redis errors as ``False``."""

        item = bloom_item_from_query(query_text)
        try:
            raw = await cast(Any, self._r).execute_command(
                "BF.EXISTS",
                self._key,
                item,
            )
            return bool(raw)
        except Exception:
            _LOG.warning("bloom_exists_failed key=%s", self._key, exc_info=True)
            return False


__all__ = [
    "BloomFeatureRedisAdapter",
    "BloomFilterService",
    "bloom_item_from_query",
]
