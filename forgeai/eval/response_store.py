"""Redis-backed response store for asynchronous offline judging."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from forgeai.eval.constants import RESPONSE_STORE_KEY_PREFIX, RESPONSE_STORE_TTL_SECONDS

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoredResponse:
    request_id: uuid.UUID
    query: str
    response_text: str
    retrieved_context: str
    stored_at: datetime


class ResponseStore:
    """Best-effort response persistence used only by async eval workers."""

    __slots__ = ("_redis", "_ttl_seconds")

    def __init__(
        self,
        redis_client: Redis[str] | None,
        ttl_seconds: int = RESPONSE_STORE_TTL_SECONDS,
    ) -> None:
        self._redis = redis_client
        self._ttl_seconds = int(ttl_seconds)

    async def store(
        self,
        request_id: uuid.UUID,
        query: str,
        response_text: str,
        retrieved_context: str,
    ) -> None:
        if self._redis is None:
            return
        key = response_store_key(request_id)
        body = _serialize(
            StoredResponse(
                request_id=request_id,
                query=query,
                response_text=response_text,
                retrieved_context=retrieved_context,
                stored_at=datetime.now(UTC),
            )
        )
        try:
            await self._redis.setex(key, self._ttl_seconds, body)
        except Exception:
            _LOG.error(
                "response_store_write_failed request_id=%s", request_id, exc_info=True
            )

    async def fetch(self, request_id: uuid.UUID) -> StoredResponse | None:
        if self._redis is None:
            return None
        key = response_store_key(request_id)
        try:
            raw = await self._redis.get(key)
        except Exception:
            _LOG.error(
                "response_store_read_failed request_id=%s", request_id, exc_info=True
            )
            return None
        if raw is None:
            return None
        return _deserialize(raw, request_id)


def response_store_key(request_id: uuid.UUID) -> str:
    return f"{RESPONSE_STORE_KEY_PREFIX}{request_id}"


def _serialize(item: StoredResponse) -> str:
    payload = asdict(item)
    payload["request_id"] = str(item.request_id)
    payload["stored_at"] = item.stored_at.isoformat()
    return json.dumps(payload, separators=(",", ":"))


def _deserialize(raw: object, request_id: uuid.UUID) -> StoredResponse | None:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        payload = json.loads(text)
        return StoredResponse(
            request_id=uuid.UUID(str(payload["request_id"])),
            query=str(payload["query"]),
            response_text=str(payload["response_text"]),
            retrieved_context=str(payload["retrieved_context"]),
            stored_at=datetime.fromisoformat(str(payload["stored_at"])),
        )
    except Exception:
        _LOG.error(
            "response_store_parse_failed request_id=%s", request_id, exc_info=True
        )
        return None


__all__ = ["ResponseStore", "StoredResponse", "response_store_key"]
