"""Async persistence for ``routing_decisions`` with JSONL fallback (Section 4).

Does not own: gateway request lifecycle, bandit scoring, or offline judge jobs.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Final

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgeai.policy.constants import (
    DEFAULT_DECISION_FALLBACK_PATH,
    DecisionLoggerConfigKey,
)
from forgeai.policy.decision_log_orm import build_orm_record
from forgeai.policy.decision_log_record import DecisionLog
from forgeai.policy.decision_log_serialize import (
    decision_log_json_dumps,
    decision_log_to_jsonable,
)
from forgeai.policy.decision_log_validate import list_consistency_violations
from forgeai.policy.decision_logger_io import (
    JSONL_FALLBACK_REASON_KEY,
    try_append_fallback_jsonl,
)
from forgeai.policy.decision_logger_metrics import (
    DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL,
    DECISIONS_FALLBACK_TOTAL,
    DECISIONS_LABEL_MISSING_TOTAL,
    DECISIONS_LOGGED_TOTAL,
    DecisionFallbackReason,
)
from forgeai.registry.constants import LabelSource
from forgeai.registry.models.routing_decision import RoutingDecision

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def decision_fallback_path_from_env() -> str:
    """Resolve JSONL path from ``FORGEAI_DECISION_FALLBACK_PATH``."""

    return os.environ.get(
        DecisionLoggerConfigKey.DECISION_FALLBACK_PATH.value,
        DEFAULT_DECISION_FALLBACK_PATH,
    )


def _log_consistency_violations(decision: DecisionLog, codes: list[str]) -> None:
    """Emit CRITICAL structured logs for each invariant breach."""

    payload = decision_log_json_dumps(decision)
    for code in codes:
        _LOG.critical(
            "routing_decision_consistency_violation code=%s decision=%s",
            code,
            payload,
        )


def _write_fallback_payload(
    path: str,
    decision: DecisionLog,
    reason: str,
) -> None:
    """Append JSONL line; on failure emit CRITICAL with raw ``DecisionLog`` JSON."""

    body = dict(decision_log_to_jsonable(decision))
    body[JSONL_FALLBACK_REASON_KEY] = reason
    if not try_append_fallback_jsonl(path, body):
        _LOG.critical(
            "routing_decision_jsonl_unavailable decision=%s",
            decision_log_json_dumps(decision),
        )


async def _commit_routing_decision_row(
    session_factory: async_sessionmaker[AsyncSession],
    row: RoutingDecision,
) -> None:
    """Insert ``row`` in one transaction (raises on DB failure)."""

    async with session_factory() as session:
        session.add(row)
        await session.commit()


async def _update_reward_columns(
    session_factory: async_sessionmaker[AsyncSession],
    request_id: uuid.UUID,
    *,
    r_online: float,
    r_components: dict[str, float],
) -> bool:
    """Update online reward columns for an existing routing decision row."""

    async with session_factory() as session:
        row = await session.get(RoutingDecision, request_id)
        if row is None:
            return False
        row.r_online = float(r_online)
        row.r_components = {k: float(v) for k, v in r_components.items()}
        await session.commit()
    return True


def _record_consistency_violations(decision: DecisionLog) -> None:
    """Count and log structural invariant breaches before persistence."""

    violations = list_consistency_violations(decision)
    if not violations:
        return
    DECISIONS_CONSISTENCY_VIOLATIONS_TOTAL.inc(len(violations))
    _log_consistency_violations(decision, violations)


def _emit_serialization_failure(
    fallback_path: str,
    decision: DecisionLog,
) -> None:
    """CRITICAL log, fallback counter, JSONL append for ORM build failures."""

    _LOG.critical(
        "routing_decision_orm_build_failed decision=%s",
        decision_log_json_dumps(decision),
        exc_info=True,
    )
    DECISIONS_FALLBACK_TOTAL.labels(
        reason=DecisionFallbackReason.SERIALIZATION_ERROR,
    ).inc()
    _write_fallback_payload(
        fallback_path,
        decision,
        DecisionFallbackReason.SERIALIZATION_ERROR,
    )


def _emit_db_failure(fallback_path: str, decision: DecisionLog) -> None:
    """CRITICAL log, fallback counter, JSONL append for commit failures."""

    _LOG.critical(
        "routing_decision_db_commit_failed decision=%s",
        decision_log_json_dumps(decision),
        exc_info=True,
    )
    DECISIONS_FALLBACK_TOTAL.labels(reason=DecisionFallbackReason.DB_ERROR).inc()
    _write_fallback_payload(
        fallback_path,
        decision,
        DecisionFallbackReason.DB_ERROR,
    )


class DecisionLogger:
    """Writes phase-1 rows to Postgres; phase-2 labels; never fails the request path.

    ``log_decision`` and ``label_decision`` never raise: failures become CRITICAL logs,
    Prometheus counters, and optional JSONL append.
    """

    __slots__ = ("_deployment_version", "_fallback_path", "_session_factory")

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        deployment_version: str,
    ) -> None:
        """Wire async session factory and process deployment id for log correlation."""

        self._session_factory = session_factory
        self._deployment_version = deployment_version
        self._fallback_path = decision_fallback_path_from_env()

    async def log_decision(self, decision: DecisionLog) -> None:
        """Persist ``decision`` or JSONL fallback; never raises to callers.

        Guarantees: no exception propagates; violations still attempt insert;
        metrics updated.
        """

        _record_consistency_violations(decision)
        try:
            row = build_orm_record(decision)
        except Exception:
            _emit_serialization_failure(self._fallback_path, decision)
            return
        try:
            await _commit_routing_decision_row(self._session_factory, row)
        except Exception:
            _emit_db_failure(self._fallback_path, decision)
            return
        DECISIONS_LOGGED_TOTAL.inc()
        _LOG.info(
            "routing_decision_logged request_id=%s r_online=%s deployment=%s",
            decision.request_id,
            decision.r_online,
            self._deployment_version,
        )

    async def label_decision(
        self,
        request_id: uuid.UUID,
        q_offline: float,
        judge_score: float,
        groundedness_score: float,
        label_source: LabelSource,
        *,
        force: bool = False,
    ) -> None:
        """Phase-2 label columns; warns on missing/already-labeled rows; never raises.

        Guarantees: no exception propagates to integration callers.
        """

        try:
            await self._label_decision_impl(
                request_id,
                q_offline,
                judge_score,
                groundedness_score,
                label_source,
                force=force,
            )
        except Exception:
            _LOG.warning(
                "routing_decision_label_swallowed request_id=%s",
                request_id,
                exc_info=True,
            )

    async def _label_decision_impl(
        self,
        request_id: uuid.UUID,
        q_offline: float,
        judge_score: float,
        groundedness_score: float,
        label_source: LabelSource,
        *,
        force: bool,
    ) -> None:
        """Internal label path; SQLAlchemy errors swallowed by ``label_decision``."""

        async with self._session_factory() as session:
            row = await session.get(RoutingDecision, request_id)
            if row is None:
                DECISIONS_LABEL_MISSING_TOTAL.inc()
                _LOG.warning("routing_decision_label_missing request_id=%s", request_id)
                return
            if row.labeled_at is not None and not force:
                prior: dict[str, Any] = {
                    RoutingDecision.q_offline.key: row.q_offline,
                    RoutingDecision.judge_score.key: row.judge_score,
                    RoutingDecision.groundedness_score.key: row.groundedness_score,
                    RoutingDecision.label_source.key: row.label_source,
                }
                _LOG.warning(
                    "routing_decision_already_labeled request_id=%s existing=%s",
                    request_id,
                    prior,
                )
                return
            row.q_offline = q_offline
            row.judge_score = judge_score
            row.groundedness_score = groundedness_score
            row.label_source = label_source.value
            row.labeled_at = datetime.now(UTC)
            await session.commit()

    async def update_online_reward(
        self,
        request_id: uuid.UUID,
        *,
        r_online: float,
        r_components: dict[str, float],
    ) -> None:
        """Update r_online/r_components if row exists; swallow failures."""

        try:
            ok = await _update_reward_columns(
                self._session_factory,
                request_id,
                r_online=r_online,
                r_components=r_components,
            )
            if not ok:
                _LOG.warning(
                    "routing_decision_reward_missing request_id=%s", request_id
                )
        except Exception:
            _LOG.warning(
                "routing_decision_reward_update_swallowed request_id=%s",
                request_id,
                exc_info=True,
            )


__all__ = ["DecisionLogger", "decision_fallback_path_from_env"]
