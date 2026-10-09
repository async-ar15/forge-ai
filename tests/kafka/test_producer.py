from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.observability.metrics import (
    KAFKA_SEND_ERRORS_TOTAL,
    KAFKA_SERIALIZATION_ERRORS_TOTAL,
)


class _FakeProducer:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.started = False
        self.stopped = False
        self.fail_send = fail_send
        self.sent: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, topic: str, value: bytes) -> asyncio.Future[Any]:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append((topic, value))
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        fut.set_result({"ok": True})
        return fut


@dataclass
class _Event:
    value: int

    def to_json(self) -> str:
        return f'{{"value":{self.value}}}'


class _BadEvent:
    def to_json(self) -> str:
        raise TypeError("boom")


@pytest.mark.asyncio
async def test_send_event_serializes_and_sends_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    fake = _FakeProducer()
    producer = ForgeKafkaProducer(producer_factory=lambda _servers: fake)
    await producer.start()

    producer.send_event("forgeai.rewards", _Event(value=3))
    await asyncio.sleep(0)

    assert fake.started is True
    assert len(fake.sent) == 1
    assert fake.sent[0][0] == "forgeai.rewards"
    assert fake.sent[0][1] == b'{"value":3}'


@pytest.mark.asyncio
async def test_serialization_error_increments_counter_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    fake = _FakeProducer()
    producer = ForgeKafkaProducer(producer_factory=lambda _servers: fake)
    await producer.start()
    before = KAFKA_SERIALIZATION_ERRORS_TOTAL._value.get()

    producer.send_event("forgeai.rewards", _BadEvent())
    await asyncio.sleep(0)

    assert KAFKA_SERIALIZATION_ERRORS_TOTAL._value.get() == before + 1


@pytest.mark.asyncio
async def test_send_error_increments_counter_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    fake = _FakeProducer(fail_send=True)
    producer = ForgeKafkaProducer(producer_factory=lambda _servers: fake)
    await producer.start()
    topic = "forgeai.rewards"
    before = KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic)._value.get()

    producer.send_event(topic, _Event(value=9))
    await asyncio.sleep(0)

    assert KAFKA_SEND_ERRORS_TOTAL.labels(topic=topic)._value.get() == before + 1


@pytest.mark.asyncio
async def test_graceful_shutdown_calls_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    fake = _FakeProducer()
    producer = ForgeKafkaProducer(producer_factory=lambda _servers: fake)
    await producer.start()
    await producer.stop()
    assert fake.stopped is True
