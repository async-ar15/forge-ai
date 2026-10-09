from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from forgeai.kafka.constants import TOPIC_DLQ
from forgeai.kafka.consumer import ForgeKafkaConsumer
from forgeai.kafka.producer import ForgeKafkaProducer


@dataclass
class _Record:
    topic: str
    partition: int
    offset: int
    value: bytes


class _FakeConsumer:
    def __init__(self, records: list[_Record]) -> None:
        self.started = False
        self.stopped = False
        self._records = records
        self.commits: list[dict[TopicPartition, OffsetAndMetadata] | None] = []
        self.subscribed: list[str] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    async def getmany(self, timeout_ms: int = 0) -> dict[Any, list[_Record]]:
        if not self._records:
            await asyncio.sleep(0.01)
            return {}
        r = self._records.pop(0)
        tp = TopicPartition(topic=r.topic, partition=r.partition)
        return {tp: [r]}

    async def commit(
        self,
        offsets: dict[TopicPartition, OffsetAndMetadata] | None = None,
    ) -> None:
        self.commits.append(offsets)

    def highwater(self, partition: TopicPartition) -> int | None:
        return 50


class _ProducerStub(ForgeKafkaProducer):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def send_event(self, topic: str, event: Any) -> None:
        self.events.append((topic, event))


@pytest.mark.asyncio
async def test_successful_consumption_calls_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("KAFKA_CONSUMER_GROUP_ID", "group-x")
    rec = _Record("forgeai.decisions", 0, 10, b'{"x":1}')
    fake = _FakeConsumer([rec])
    producer = _ProducerStub()
    consumer = ForgeKafkaConsumer(
        producer=producer,
        consumer_factory=lambda _servers, _group: fake,
    )
    seen: list[dict[str, Any]] = []

    async def handler(event: dict[str, Any]) -> None:
        seen.append(event)
        consumer._running = False

    await consumer.consume("forgeai.decisions", handler)

    assert seen == [{"x": 1}]
    assert len(fake.commits) == 1
    assert fake.commits[0] is not None


@pytest.mark.asyncio
async def test_handler_failure_triggers_retry_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("KAFKA_CONSUMER_GROUP_ID", "group-x")
    rec = _Record("forgeai.decisions", 0, 11, b'{"x":2}')
    fake = _FakeConsumer([rec])
    producer = _ProducerStub()
    consumer = ForgeKafkaConsumer(
        producer=producer,
        consumer_factory=lambda _servers, _group: fake,
    )
    attempts = {"n": 0}

    async def handler(_event: dict[str, Any]) -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("try again")
        consumer._running = False

    await consumer.consume("forgeai.decisions", handler)
    assert attempts["n"] == 3
    assert len(fake.commits) == 1


@pytest.mark.asyncio
async def test_retry_exhaustion_publishes_to_dlq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("KAFKA_CONSUMER_GROUP_ID", "group-x")
    rec = _Record("forgeai.decisions", 0, 12, b'{"x":3}')
    fake = _FakeConsumer([rec])
    producer = _ProducerStub()
    consumer = ForgeKafkaConsumer(
        producer=producer,
        consumer_factory=lambda _servers, _group: fake,
    )

    async def handler(_event: dict[str, Any]) -> None:
        raise RuntimeError("always fail")

    task = asyncio.create_task(consumer.consume("forgeai.decisions", handler))
    await asyncio.sleep(0.25)
    await consumer.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert any(topic == TOPIC_DLQ for topic, _payload in producer.events)
    assert len(fake.commits) == 1  # final commit on graceful stop only


@pytest.mark.asyncio
async def test_offset_committed_only_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("KAFKA_CONSUMER_GROUP_ID", "group-x")
    rec = _Record("forgeai.decisions", 0, 20, json.dumps({"x": 4}).encode("utf-8"))
    fake = _FakeConsumer([rec])
    producer = _ProducerStub()
    consumer = ForgeKafkaConsumer(
        producer=producer,
        consumer_factory=lambda _servers, _group: fake,
    )

    async def handler(_event: dict[str, Any]) -> None:
        consumer._running = False

    await consumer.consume("forgeai.decisions", handler)
    assert len(fake.commits) == 1
    offsets = fake.commits[0]
    assert offsets is not None
    tp = next(iter(offsets))
    assert tp.topic == "forgeai.decisions"
