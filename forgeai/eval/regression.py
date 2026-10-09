"""Regression alerts for policy quality and latency drift.

This detector alerts on regressions. It does not roll back policy versions
automatically. Automated rollback is a v2 feature requiring human-in-the-loop
validation of the rollback criteria.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.engine import Engine

from forgeai.eval.constants import (
    REGRESSION_CHECK_INTERVAL_SECONDS,
    REGRESSION_LATENCY_THRESHOLD_MS,
    REGRESSION_R_ONLINE_THRESHOLD,
    REGRESSION_SLO_THRESHOLD,
)
from forgeai.kafka.constants import TOPIC_REGRESSIONS
from forgeai.kafka.events import RegressionEvent
from forgeai.kafka.producer import ForgeKafkaProducer
from forgeai.observability.eval_queries import (
    get_policy_performance,
    get_recent_policy_versions,
)
from forgeai.observability.metrics import REGRESSION_DETECTED_TOTAL

_LOG = logging.getLogger(__name__)


class RegressionDetector:
    """Alert-only regression detector over exploitative offline eval metrics."""

    __slots__ = ("_engine", "_interval", "_producer")

    def __init__(
        self,
        *,
        engine: Engine,
        producer: ForgeKafkaProducer,
        interval_seconds: int = REGRESSION_CHECK_INTERVAL_SECONDS,
    ) -> None:
        self._engine = engine
        self._producer = producer
        self._interval = int(interval_seconds)

    async def run_forever(self) -> None:
        while True:
            await self.check_once()
            await asyncio.sleep(self._interval)

    async def check_once(self) -> None:
        versions = get_recent_policy_versions(self._engine, limit=2)
        if len(versions) < 2:
            return
        now = datetime.now(UTC)
        start = now - timedelta(seconds=self._interval)
        current, previous = versions[0], versions[1]
        cur = _perf_row(self._engine, current, start, now)
        prev = _perf_row(self._engine, previous, start, now)
        for event in _detect_events(previous, current, prev, cur):
            self._emit_event(event)

    def _emit_event(self, event: RegressionEvent) -> None:
        REGRESSION_DETECTED_TOTAL.labels(metric_name=event.metric_name).inc()
        self._producer.send_event(TOPIC_REGRESSIONS, event)
        _LOG.critical(
            "policy_regression metric=%s old=%s new=%s delta=%.6f",
            event.metric_name,
            event.policy_version_old,
            event.policy_version_new,
            event.delta,
        )


def _perf_row(
    engine: Engine,
    version: str,
    start: datetime,
    end: datetime,
) -> dict[str, float]:
    frame = get_policy_performance(engine, version, start, end)
    row = frame.iloc[0]
    return {
        "avg_r_online": _as_float(row.get("avg_r_online")),
        "p95_latency_ms": _as_float(row.get("p95_latency_ms")),
        "slo_violation_rate": _as_float(row.get("slo_violation_rate")),
    }


def _detect_events(
    previous_version: str,
    current_version: str,
    previous: dict[str, float],
    current: dict[str, float],
) -> list[RegressionEvent]:
    out: list[RegressionEvent] = []
    now = datetime.now(UTC)
    out.extend(
        _r_online_drop_events(now, previous_version, current_version, previous, current)
    )
    out.extend(
        _latency_rise_events(now, previous_version, current_version, previous, current)
    )
    out.extend(
        _slo_rise_events(now, previous_version, current_version, previous, current)
    )
    return out


def _r_online_drop_events(
    now: datetime,
    old: str,
    new: str,
    previous: dict[str, float],
    current: dict[str, float],
) -> list[RegressionEvent]:
    prev = previous["avg_r_online"]
    cur = current["avg_r_online"]
    if prev == 0.0:
        return []
    drop_ratio = (prev - cur) / abs(prev)
    if drop_ratio <= REGRESSION_R_ONLINE_THRESHOLD:
        return []
    return [_event(now, "avg_r_online", prev, cur, old, new)]


def _latency_rise_events(
    now: datetime,
    old: str,
    new: str,
    previous: dict[str, float],
    current: dict[str, float],
) -> list[RegressionEvent]:
    prev = previous["p95_latency_ms"]
    cur = current["p95_latency_ms"]
    if (cur - prev) <= REGRESSION_LATENCY_THRESHOLD_MS:
        return []
    return [_event(now, "p95_latency_ms", prev, cur, old, new)]


def _slo_rise_events(
    now: datetime,
    old: str,
    new: str,
    previous: dict[str, float],
    current: dict[str, float],
) -> list[RegressionEvent]:
    prev = previous["slo_violation_rate"]
    cur = current["slo_violation_rate"]
    if (cur - prev) <= REGRESSION_SLO_THRESHOLD:
        return []
    return [_event(now, "slo_violation_rate", prev, cur, old, new)]


def _event(
    now: datetime,
    metric_name: str,
    previous_value: float,
    current_value: float,
    old: str,
    new: str,
) -> RegressionEvent:
    return RegressionEvent(
        detected_at=now,
        metric_name=metric_name,
        previous_value=previous_value,
        current_value=current_value,
        delta=current_value - previous_value,
        policy_version_old=old,
        policy_version_new=new,
        severity="critical",
    )


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"cannot coerce {type(value)!r} to float")


__all__ = ["RegressionDetector"]
