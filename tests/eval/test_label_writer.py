from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.eval.judge import JudgeScores
from forgeai.eval.label_writer import EvalLabelWriter
from forgeai.eval.response_store import StoredResponse
from forgeai.kafka.constants import TOPIC_EVAL_LABELS
from forgeai.observability.metrics import EVAL_JUDGE_FAILURES_TOTAL


def _stored_response(rid: uuid.UUID) -> StoredResponse:
    return StoredResponse(
        request_id=rid,
        query="query",
        response_text="answer",
        retrieved_context="ctx",
        stored_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_successful_label_write_end_to_end() -> None:
    rid = uuid.uuid4()
    consumer = MagicMock()
    producer = MagicMock()
    response_store = MagicMock()
    response_store.fetch = AsyncMock(return_value=_stored_response(rid))
    judge = MagicMock()
    judge.score = AsyncMock(
        return_value=JudgeScores(
            relevance=0.8,
            groundedness=0.7,
            completeness=0.6,
            conciseness=0.9,
            composite=0.75,
            raw_response="{}",
            judged_at=datetime.now(UTC),
            judge_model="m",
            parse_success=True,
        )
    )
    dlog = MagicMock()
    dlog.label_decision = AsyncMock(return_value=None)
    writer = EvalLabelWriter(
        consumer=consumer,
        producer=producer,
        response_store=response_store,
        judge=judge,
        decision_logger=dlog,
    )
    await writer._handle_decision_event({"request_id": str(rid)})
    dlog.label_decision.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_failure_skips_label_write_increments_counter() -> None:
    rid = uuid.uuid4()
    writer, dlog, _producer = _writer_with_mocks(rid)
    writer._judge.score = AsyncMock(return_value=None)
    before = EVAL_JUDGE_FAILURES_TOTAL._value.get()
    await writer._handle_decision_event({"request_id": str(rid)})
    dlog.label_decision.assert_not_called()
    assert EVAL_JUDGE_FAILURES_TOTAL._value.get() == before + 1


@pytest.mark.asyncio
async def test_response_store_miss_skips_processing() -> None:
    rid = uuid.uuid4()
    writer, dlog, producer = _writer_with_mocks(rid)
    writer._response_store.fetch = AsyncMock(return_value=None)
    await writer._handle_decision_event({"request_id": str(rid)})
    dlog.label_decision.assert_not_called()
    producer.send_event.assert_not_called()


@pytest.mark.asyncio
async def test_eval_label_event_published_after_write() -> None:
    rid = uuid.uuid4()
    writer, _dlog, producer = _writer_with_mocks(rid)
    await writer._handle_decision_event({"request_id": str(rid)})
    producer.send_event.assert_called_once()
    args = producer.send_event.call_args.args
    assert args[0] == TOPIC_EVAL_LABELS


def _writer_with_mocks(rid: uuid.UUID):
    consumer = MagicMock()
    producer = MagicMock()
    response_store = MagicMock()
    response_store.fetch = AsyncMock(return_value=_stored_response(rid))
    judge = MagicMock()
    judge.score = AsyncMock(
        return_value=JudgeScores(
            relevance=0.8,
            groundedness=0.7,
            completeness=0.6,
            conciseness=0.9,
            composite=0.75,
            raw_response="{}",
            judged_at=datetime.now(UTC),
            judge_model="m",
            parse_success=True,
        )
    )
    dlog = MagicMock()
    dlog.label_decision = AsyncMock(return_value=None)
    writer = EvalLabelWriter(
        consumer=consumer,
        producer=producer,
        response_store=response_store,
        judge=judge,
        decision_logger=dlog,
    )
    return writer, dlog, producer
