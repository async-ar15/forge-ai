"""Kafka consumer wrapper with retries, DLQ, and manual commits."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import OffsetAndMetadata, TopicPartition

from forgeai.kafka.constants import (
    ENV_KAFKA_BOOTSTRAP_SERVERS,
    ENV_KAFKA_CONSUMER_GROUP_ID,
    TOPIC_DLQ,
)
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.observability.metrics import KAFKA_CONSUMER_LAG_GAUGE

_LOG = logging.getLogger(__name__)


class _ConsumerRecordLike(Protocol):
    topic: str
    partition: int
    offset: int
    value: bytes


class _KafkaConsumerLike(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def subscribe(self, topics: list[str]) -> None: ...
    async def getmany(
        self, timeout_ms: int = 0
    ) -> dict[Any, list[_ConsumerRecordLike]]: ...
    async def commit(
        self,
        offsets: dict[TopicPartition, OffsetAndMetadata] | None = None,
    ) -> None: ...
    def highwater(self, partition: TopicPartition) -> int | None: ...


@dataclass(frozen=True, slots=True)
class _DlqPayload:
    topic: str
    partition: int
    offset: int
    payload: dict[str, Any]
    error: str


class ForgeKafkaConsumer:
    """Kafka consumer with retry and DLQ behavior for handler failures."""

    __slots__ = ("_consumer", "_factory", "_producer", "_running")

    def __init__(
        self,
        *,
        producer: ForgeKafkaProducer,
        consumer_factory: Callable[[str, str], _KafkaConsumerLike] | None = None,
    ) -> None:
        self._producer = producer
        self._factory = consumer_factory or _default_consumer_factory
        self._consumer: _KafkaConsumerLike | None = None
        self._running = False

    async def start(self) -> None:
        servers = os.environ.get(ENV_KAFKA_BOOTSTRAP_SERVERS, "")
        group_id = os.environ.get(ENV_KAFKA_CONSUMER_GROUP_ID, "forgeai-default")
        self._consumer = self._factory(servers, group_id)
        await self._consumer.start()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._consumer is None:
            return
        try:
            await self._consumer.commit()
        except Exception:
            _LOG.warning("kafka_consumer_final_commit_failed", exc_info=True)
        await self._consumer.stop()
        self._consumer = None

    async def consume(
        self,
        topic: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if self._consumer is None:
            await self.start()
        assert self._consumer is not None
        self._consumer.subscribe([topic])
        while self._running:
            batches = await self._consumer.getmany(timeout_ms=1000)
            for tp, records in batches.items():
                for record in records:
                    success = await self._process_record(record, handler)
                    if success:
                        offsets = {tp: OffsetAndMetadata(record.offset + 1, "")}
                        await self._consumer.commit(offsets=offsets)
                    self._record_lag(tp, record.offset)

    async def _process_record(
        self,
        record: _ConsumerRecordLike,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> bool:
        payload = _decode_payload(record.value)
        for attempt in range(1, 4):
            try:
                await handler(payload)
                return True
            except Exception as exc:
                if attempt == 3:
                    await self._publish_dlq(record, payload, exc)
                    return False
                await asyncio.sleep(0.05 * attempt)
        return False

    async def _publish_dlq(
        self,
        record: _ConsumerRecordLike,
        payload: dict[str, Any],
        exc: Exception,
    ) -> None:
        dlq = _DlqPayload(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            payload=payload,
            error=f"{type(exc).__name__}: {exc}",
        )
        self._producer.send_event(TOPIC_DLQ, asdict(dlq))

    def _record_lag(self, tp: TopicPartition, offset: int) -> None:
        if self._consumer is None:
            return
        hi = self._consumer.highwater(tp)
        if hi is None:
            return
        lag = max(0, int(hi) - int(offset + 1))
        KAFKA_CONSUMER_LAG_GAUGE.labels(topic=tp.topic).set(float(lag))


def _decode_payload(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8")
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        return {"raw": text}


def _default_consumer_factory(bootstrap: str, group_id: str) -> _KafkaConsumerLike:
    return cast(
        _KafkaConsumerLike,
        AIOKafkaConsumer(
            bootstrap_servers=bootstrap,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
    )


__all__ = ["ForgeKafkaConsumer"]
