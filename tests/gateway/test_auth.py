"""Gateway authentication and rate limiting."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forgeai.gateway.auth_core import api_key_sha256_hex, resolve_tenant_context
from forgeai.gateway.rate_limit import sliding_window_allow
from forgeai.gateway.tenant_context import TenantContext, fail_open_context
from forgeai.registry.models.tenant import Tenant


@pytest.mark.asyncio
async def test_resolve_valid_key_round_trip(
    tenant_row_bcrypt: tuple[str, str, uuid.UUID, str],
) -> None:
    """Postgres row + bcrypt accepts the plaintext API key."""

    key, _sha, tid, bcrypt_hash = tenant_row_bcrypt
    row = MagicMock(spec=Tenant)
    row.api_key_bcrypt_hash = bcrypt_hash
    row.tenant_tier = "free"
    row.tenant_id = tid

    class _Sess:
        async def execute(self, _stmt: object) -> MagicMock:
            r = MagicMock()
            r.scalar_one_or_none = MagicMock(return_value=row)
            return r

        async def __aenter__(self) -> _Sess:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

    class _Sf:
        def __call__(self) -> _Sess:
            return _Sess()

    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)

    ctx, ok = await resolve_tenant_context(
        raw_api_key=key,
        redis=redis,
        session_factory=_Sf(),
    )
    assert ok and ctx is not None
    assert ctx.tenant_id == tid
    redis.setex.assert_awaited()


@pytest.mark.asyncio
async def test_redis_cache_hit_skips_postgres(
    tenant_row_bcrypt: tuple[str, str, uuid.UUID, str],
) -> None:
    """Cached auth JSON avoids a second database round trip."""

    key, _sha, tid, _h = tenant_row_bcrypt
    cached = (
        '{"tenant_id":"' + str(tid) + '","tenant_tier":"pro","auth_degraded":false}'
    )
    redis = MagicMock()
    redis.get = AsyncMock(return_value=cached)
    called = {"n": 0}

    class _Sf:
        def __call__(self) -> object:
            called["n"] += 1
            raise AssertionError("session_factory should not run on cache hit")

    ctx, ok = await resolve_tenant_context(
        raw_api_key=key,
        redis=redis,
        session_factory=_Sf(),
    )
    assert ok and ctx is not None
    assert ctx.tenant_tier == "pro"
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_redis_down_falls_back_to_postgres(
    tenant_row_bcrypt: tuple[str, str, uuid.UUID, str],
) -> None:
    """Redis errors are swallowed; Postgres path still validates."""

    key, _sha, tid, bcrypt_hash = tenant_row_bcrypt
    row = MagicMock(spec=Tenant)
    row.api_key_bcrypt_hash = bcrypt_hash
    row.tenant_tier = "free"
    row.tenant_id = tid

    class _Sess:
        async def execute(self, _stmt: object) -> MagicMock:
            r = MagicMock()
            r.scalar_one_or_none = MagicMock(return_value=row)
            return r

        async def __aenter__(self) -> _Sess:
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

    class _Sf:
        def __call__(self) -> _Sess:
            return _Sess()

    redis = MagicMock()
    redis.get = AsyncMock(side_effect=RuntimeError("redis down"))

    ctx, ok = await resolve_tenant_context(
        raw_api_key=key,
        redis=redis,
        session_factory=_Sf(),
    )
    assert ok and ctx is not None


@pytest.mark.asyncio
async def test_postgres_down_fails_open() -> None:
    """Auth infrastructure failure yields synthetic free-tier context."""

    redis = MagicMock()
    redis.get = AsyncMock(side_effect=OSError("redis"))

    class _Sf:
        def __call__(self) -> object:
            raise ConnectionError("db down")

    ctx, ok = await resolve_tenant_context(
        raw_api_key="k",
        redis=redis,
        session_factory=_Sf(),
    )
    assert ok and ctx == fail_open_context()


@pytest.mark.asyncio
async def test_sliding_window_blocks_over_limit() -> None:
    """Sorted-set window returns 429 semantics when over limit."""

    redis = MagicMock()
    pipe = MagicMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[None, 10])
    redis.pipeline = MagicMock(return_value=pipe)
    redis.zrange = AsyncMock(return_value=[("m", 1000.0)])

    allowed, retry = await sliding_window_allow(
        redis,
        tenant_id=uuid.uuid4(),
        limit=10,
    )
    assert allowed is False
    assert retry >= 1


def test_missing_and_invalid_key_same_status(gateway_client) -> None:
    """401 for missing and invalid keys (timing not asserted — contract only)."""

    client, _state = gateway_client
    r0 = client.post(
        "/v1/infer",
        json={"query": "x"},
        headers={"X-API-Key": ""},
    )
    r1 = client.post(
        "/v1/infer",
        json={"query": "x"},
        headers={"X-API-Key": "not-a-real-key"},
    )
    assert r0.status_code == 401
    assert r1.status_code == 401


def test_valid_key_allows_infer(gateway_client) -> None:
    """Synthetic tenant context passes auth middleware."""

    client, _state = gateway_client
    tid = uuid.uuid4()
    ctx = TenantContext(tenant_id=tid, tenant_tier="free", auth_degraded=False)

    async def _ok(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return ctx, True

    with patch(
        "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
        new=_ok,
    ):
        resp = client.post(
            "/v1/infer",
            json={"query": "hello", "query_type": "unknown"},
            headers={"X-API-Key": "any"},
        )
    assert resp.status_code == 200


def test_rate_limit_429_retry_after(gateway_client) -> None:
    """Over-quota requests receive ``Retry-After``."""

    client, _state = gateway_client
    tid = uuid.uuid4()
    ctx = TenantContext(tenant_id=tid, tenant_tier="free", auth_degraded=False)

    async def _auth(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return ctx, True

    async def _block(*_a: object, **_k: object) -> tuple[bool, int]:
        return False, 12

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        patch(
            "forgeai.gateway.middleware.auth_and_limit.sliding_window_allow",
            new=_block,
        ),
    ):
        resp = client.post(
            "/v1/infer",
            json={"query": "hello"},
            headers={"X-API-Key": "k"},
        )
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "12"


def test_rate_limit_pro_tier_limit(gateway_client) -> None:
    """``sliding_window_allow`` receives pro RPM (100)."""

    client, _state = gateway_client
    tid = uuid.uuid4()
    ctx = TenantContext(tenant_id=tid, tenant_tier="pro", auth_degraded=False)
    captured: dict[str, int] = {}

    async def _auth(*_a: object, **_k: object) -> tuple[TenantContext | None, bool]:
        return ctx, True

    async def _rl(
        redis: object,
        *,
        tenant_id: uuid.UUID,
        limit: int,
    ) -> tuple[bool, int]:
        captured["limit"] = limit
        return True, 0

    with (
        patch(
            "forgeai.gateway.middleware.auth_and_limit.resolve_tenant_context",
            new=_auth,
        ),
        patch(
            "forgeai.gateway.middleware.auth_and_limit.sliding_window_allow",
            new=_rl,
        ),
    ):
        client.post(
            "/v1/infer",
            json={"query": "hello"},
            headers={"X-API-Key": "k"},
        )
    assert captured["limit"] == 100


def test_api_key_sha256_hex_stable() -> None:
    """SHA-256 cache key helper is deterministic."""

    assert (
        api_key_sha256_hex("abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
