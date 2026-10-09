from __future__ import annotations

import uuid

import pytest
from forgeai.eval.constants import RESPONSE_STORE_KEY_PREFIX
from forgeai.eval.response_store import (
    ResponseStore,
    StoredResponse,
    response_store_key,
)


class _RedisStub:
    def __init__(self) -> None:
        self.saved: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.fail_get = False

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.saved[key] = value
        self.ttl[key] = ttl

    async def get(self, key: str):
        if self.fail_get:
            raise RuntimeError("redis down")
        return self.saved.get(key)


@pytest.mark.asyncio
async def test_store_writes_json_with_ttl() -> None:
    redis = _RedisStub()
    store = ResponseStore(redis_client=redis, ttl_seconds=55)
    rid = uuid.uuid4()
    await store.store(rid, "q", "r", "ctx")
    key = response_store_key(rid)
    assert key in redis.saved
    assert redis.ttl[key] == 55


@pytest.mark.asyncio
async def test_fetch_returns_stored_response_on_hit() -> None:
    redis = _RedisStub()
    store = ResponseStore(redis_client=redis)
    rid = uuid.uuid4()
    await store.store(rid, "q", "r", "ctx")
    item = await store.fetch(rid)
    assert isinstance(item, StoredResponse)
    assert item is not None
    assert item.request_id == rid
    assert item.response_text == "r"


@pytest.mark.asyncio
async def test_fetch_returns_none_on_miss() -> None:
    store = ResponseStore(redis_client=_RedisStub())
    item = await store.fetch(uuid.uuid4())
    assert item is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_redis_failure_without_raising() -> None:
    redis = _RedisStub()
    redis.fail_get = True
    store = ResponseStore(redis_client=redis)
    assert await store.fetch(uuid.uuid4()) is None


def test_key_pattern_matches_constant() -> None:
    rid = uuid.UUID("f57f5562-c28c-4f3f-a4cf-c59599d7db30")
    key = response_store_key(rid)
    assert key.startswith(RESPONSE_STORE_KEY_PREFIX)
    assert key == f"{RESPONSE_STORE_KEY_PREFIX}{rid}"
