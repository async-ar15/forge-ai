"""API key verification with bcrypt and Redis cache (SHA-256 cache keys only)."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Final

import bcrypt
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgeai.gateway.constants import (
    AUTH_CACHE_TTL_SECONDS,
    AUTH_DUMMY_BCRYPT_HASH_UTF8,
    AUTH_REDIS_KEY_PREFIX,
)
from forgeai.gateway.db_tenant import load_tenant_by_api_key_sha, tenant_id_uuid
from forgeai.gateway.tenant_context import TenantContext, fail_open_context
from forgeai.observability.metrics import GATEWAY_AUTH_CACHE_HITS_TOTAL

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
_REJECT_PLAINTEXT_UTF8: Final[str] = "\x00" * 72


def api_key_sha256_hex(raw_key: str) -> str:
    """Hex digest used for Redis keys and Postgres lookup (raw key never stored)."""

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _bcrypt_verify(plaintext_utf8: str, stored_hash_utf8: str) -> bool:
    try:
        return bool(
            bcrypt.checkpw(
                plaintext_utf8.encode("utf-8"),
                stored_hash_utf8.encode("utf-8"),
            ),
        )
    except ValueError:
        return False


def _cache_key(sha_hex: str) -> str:
    return f"{AUTH_REDIS_KEY_PREFIX}{sha_hex}"


def _tenant_json(ctx: TenantContext) -> str:
    payload = {
        "tenant_id": str(ctx.tenant_id),
        "tenant_tier": ctx.tenant_tier,
        "auth_degraded": ctx.auth_degraded,
    }
    return json.dumps(payload, separators=(",", ":"))


def _context_from_cache(raw: str) -> TenantContext | None:
    try:
        d = json.loads(raw)
        return TenantContext(
            tenant_id=uuid.UUID(d["tenant_id"]),
            tenant_tier=str(d["tenant_tier"]),
            auth_degraded=bool(d.get("auth_degraded", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


async def _redis_get_context(
    redis: Redis[str],
    sha_hex: str,
) -> TenantContext | None:
    raw = await redis.get(_cache_key(sha_hex))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _context_from_cache(raw)


async def _redis_set_context(
    redis: Redis[str],
    sha_hex: str,
    ctx: TenantContext,
) -> None:
    await redis.setex(
        _cache_key(sha_hex),
        AUTH_CACHE_TTL_SECONDS,
        _tenant_json(ctx),
    )


async def resolve_tenant_context(
    *,
    raw_api_key: str | None,
    redis: Redis[str] | None,
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> tuple[TenantContext | None, bool]:
    """Return (tenant, True) on success, (None, False) on auth failure.

    Auth-backend failures return ``fail_open_context()`` with True (allow).
    """

    present = raw_api_key is not None and raw_api_key != ""
    dummy = AUTH_DUMMY_BCRYPT_HASH_UTF8

    if not present:
        _bcrypt_verify(_REJECT_PLAINTEXT_UTF8, dummy)
        return None, False

    assert raw_api_key is not None
    plaintext = raw_api_key
    sha = api_key_sha256_hex(plaintext)

    try:
        if redis is not None:
            cached = await _redis_get_context(redis, sha)
            if cached is not None:
                GATEWAY_AUTH_CACHE_HITS_TOTAL.inc()
                return cached, True
    except Exception:
        _LOG.warning("auth_redis_cache_unavailable", exc_info=True)

    hash_for_bcrypt = dummy
    tier = "free"
    tid: uuid.UUID | None = None
    try:
        if session_factory is not None:
            async with session_factory() as session:
                row = await load_tenant_by_api_key_sha(session, sha)
                if row is not None and row.api_key_bcrypt_hash:
                    hash_for_bcrypt = row.api_key_bcrypt_hash
                    tier = str(row.tenant_tier)
                    tid = tenant_id_uuid(row)
    except Exception:
        _LOG.warning("auth_postgres_lookup_failed", exc_info=True)
        return fail_open_context(), True

    ok = _bcrypt_verify(plaintext, hash_for_bcrypt)
    if not ok:
        _bcrypt_verify(_REJECT_PLAINTEXT_UTF8, dummy)
        return None, False
    if tid is None:
        _bcrypt_verify(_REJECT_PLAINTEXT_UTF8, dummy)
        return None, False

    ctx = TenantContext(tenant_id=tid, tenant_tier=tier, auth_degraded=False)
    try:
        if redis is not None:
            await _redis_set_context(redis, sha, ctx)
    except Exception:
        _LOG.warning("auth_redis_cache_set_failed", exc_info=True)
    return ctx, True


__all__ = ["api_key_sha256_hex", "resolve_tenant_context"]
