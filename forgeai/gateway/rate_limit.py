"""Per-tenant sliding-window rate limits backed by Redis sorted sets."""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Final

from redis.asyncio import Redis

from forgeai.gateway.constants import (
    RATE_LIMIT_ENTERPRISE_RPM,
    RATE_LIMIT_FREE_RPM,
    RATE_LIMIT_PRO_RPM,
    RATE_LIMIT_REDIS_KEY_PREFIX,
    RATE_LIMIT_WINDOW_SECONDS,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def requests_per_minute_for_tier(tenant_tier: str) -> int:
    """Map tier string to RPM constant."""

    t = tenant_tier.strip().lower()
    if t == "enterprise":
        return RATE_LIMIT_ENTERPRISE_RPM
    if t == "pro":
        return RATE_LIMIT_PRO_RPM
    return RATE_LIMIT_FREE_RPM


def _rl_key(tenant_id: uuid.UUID) -> str:
    return f"{RATE_LIMIT_REDIS_KEY_PREFIX}{tenant_id}"


async def sliding_window_allow(
    redis: Redis[str] | None,
    *,
    tenant_id: uuid.UUID,
    limit: int,
) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds_if_blocked). Redis failure → allow."""

    if redis is None:
        return True, 0
    key = _rl_key(tenant_id)
    now = time.time()
    window = float(RATE_LIMIT_WINDOW_SECONDS)
    cutoff = now - window
    member = f"{now:.6f}:{uuid.uuid4().hex}"
    try:
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, float("-inf"), cutoff)
        pipe.zcard(key)
        results = await pipe.execute()
        count_before = int(results[1])
        if count_before >= limit:
            oldest = await redis.zrange(key, 0, 0, withscores=True)
            if not oldest:
                return True, 0
            oldest_ts = float(oldest[0][1])
            retry_after = max(1, int(math.ceil(window - (now - oldest_ts))))
            return False, retry_after
        await redis.zadd(key, {member: now})
        await redis.expire(key, int(window) + 1)
    except Exception:
        _LOG.warning("rate_limit_redis_failed allow_request", exc_info=True)
        return True, 0
    return True, 0


__all__ = ["requests_per_minute_for_tier", "sliding_window_allow"]
