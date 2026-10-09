"""Kafka producer wrapper with non-blocking send semantics for hot paths."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Protocol

from aiokafka import AIOKafkaProducer

from forgeai.kafka.constants import ENV_KAFKA_BOOTSTRAP_SERVERS
from forgeai.observability.metrics import (
    KAFKA_SEND_ERRORS_TOTAL,
    KAFKA_SERIALIZATION_ERRORS_TOTAL,
)

_LOG = logging.getLogger(__name__)


class _KafkaProducerLike(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, topic: str, value: bytes) -> asyncio.Future[object]: ...


class ForgeKafkaProducer:
    """Thin wrapper around AIOKafkaProducer with error swallowing."""

    __slots__ = ("_factory", "_producer")

    def __init__(
        self,
        producer_factory: Callable[[str], _KafkaProducerLike] | None = None,
    ) -> None:
        self._factory = producer_factory or (
            lambda servers: AIOKafkaProducer(bootstrap_servers=servers)
        )
        self._producer: _KafkaProducerLike | None = None

    async def start(self) -> None:
        servers = os.environ.get(ENV_KAFKA_BOOTSTRAP_SERVERS, "")
        self._producer = self._factory(servers)
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        self._producer = None

    def send_event(self, topic: str, event: object) -> None:
        """Serialize and enqueue event send without awaiting delivery."""

        payload = self._serialize(event)
        if payload is None:
            return
        producer = self._producer
        if producer is None:
            _LOG.error("kafka_producer_uninitialized topic=%s", topic)
            KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic).inc()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.error("kafka_no_running_loop topic=%s", topic)
            KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic).inc()
            return
        loop.create_task(self._send_async(producer, topic, payload))

    def _serialize(self, event: object) -> bytes | None:
        try:
            if hasattr(event, "to_json"):
                data = str(event.to_json())
            else:
                data = json.dumps(event, separators=(",", ":"))
            return data.encode("utf-8")
        except Exception:
            _LOG.error(
                "kafka_serialize_failed event_type=%s",
                type(event).__name__,
                exc_info=True,
            )
            KAFKA_SERIALIZATION_ERRORS_TOTAL.inc()
            return None

    async def _send_async(
        self,
        producer: _KafkaProducerLike,
        topic: str,
        payload: bytes,
    ) -> None:
        try:
            delivery = await producer.send(topic, value=payload)
        except Exception:
            _LOG.error("kafka_send_enqueue_failed topic=%s", topic, exc_info=True)
            KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic).inc()
            return
        delivery.add_done_callback(lambda fut: self._on_delivery(topic, fut))

    def _on_delivery(self, topic: str, fut: asyncio.Future[object]) -> None:
        try:
            fut.result()
        except Exception:
            _LOG.error("kafka_send_delivery_failed topic=%s", topic, exc_info=True)
            KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic).inc()


__all__ = ["ForgeKafkaProducer"]
