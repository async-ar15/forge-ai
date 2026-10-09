"""Asynchronous q_offline labeling worker driven by decision Kafka events."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from forgeai.eval.judge import RubricEvalJudge
from forgeai.eval.response_store import ResponseStore
from forgeai.kafka.constants import TOPIC_DECISIONS, TOPIC_EVAL_LABELS
from forgeai.kafka.consumer import ForgeKafkaConsumer
from forgeai.kafka.events import EvalLabelEvent
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.observability.metrics import (
    EVAL_JUDGE_FAILURES_TOTAL,
    EVAL_LABELS_WRITTEN_TOTAL,
)
from forgeai.policy.decision_logger import DecisionLogger
from forgeai.registry.constants import LabelSource

_LOG = logging.getLogger(__name__)


class EvalLabelWriter:
    """Consumes decisions and writes async q_offline labels without blocking traffic."""

    __slots__ = (
        "_consumer",
        "_decision_logger",
        "_judge",
        "_producer",
        "_response_store",
    )

    def __init__(
        self,
        *,
        consumer: ForgeKafkaConsumer,
        producer: ForgeKafkaProducer,
        response_store: ResponseStore,
        judge: RubricEvalJudge,
        decision_logger: DecisionLogger,
    ) -> None:
        self._consumer = consumer
        self._producer = producer
        self._response_store = response_store
        self._judge = judge
        self._decision_logger = decision_logger

    async def run(self) -> None:
        await self._consumer.consume(TOPIC_DECISIONS, self._handle_decision_event)

    async def _handle_decision_event(self, event: dict[str, Any]) -> None:
        request_id = _parse_request_id(event)
        if request_id is None:
            return
        stored = await self._response_store.fetch(request_id)
        if stored is None:
            _LOG.warning("eval_response_store_miss request_id=%s", request_id)
            return
        scores = await self._judge.score(
            query=stored.query,
            response_text=stored.response_text,
            retrieved_context=stored.retrieved_context,
        )
        if scores is None:
            EVAL_JUDGE_FAILURES_TOTAL.inc()
            return
        await self._decision_logger.label_decision(
            request_id=request_id,
            q_offline=float(scores.composite),
            judge_score=float(scores.composite),
            groundedness_score=float(scores.groundedness),
            label_source=LabelSource.JUDGE,
        )
        self._producer.send_event(
            TOPIC_EVAL_LABELS,
            EvalLabelEvent(
                request_id=request_id,
                timestamp=scores.judged_at,
                q_offline=float(scores.composite),
                judge_score=float(scores.composite),
                groundedness_score=float(scores.groundedness),
                label_source=LabelSource.JUDGE.value,
            ),
        )
        EVAL_LABELS_WRITTEN_TOTAL.inc()


def _parse_request_id(event: dict[str, Any]) -> uuid.UUID | None:
    raw = event.get("request_id")
    if raw is None:
        _LOG.warning("eval_decision_event_missing_request_id")
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        _LOG.warning("eval_decision_event_bad_request_id value=%s", raw)
        return None


__all__ = ["EvalLabelWriter"]
