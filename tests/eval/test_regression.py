from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from forgeai.eval import regression as regression_mod
from forgeai.eval.regression import RegressionDetector
from forgeai.kafka.constants import TOPIC_REGRESSIONS


@pytest.mark.asyncio
async def test_no_regression_when_metrics_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    monkeypatch.setattr(
        regression_mod,
        "get_recent_policy_versions",
        lambda _engine, limit=2: ["new", "old"],
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _engine, _version, _start, _end: _frame(1.0, 100.0, 0.01),
    )
    await detector.check_once()
    producer.send_event.assert_not_called()


@pytest.mark.asyncio
async def test_regression_detected_on_r_online_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    monkeypatch.setattr(
        regression_mod, "get_recent_policy_versions", lambda *_a, **_k: ["new", "old"]
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _e, v, _s, _n: (
            _frame(0.7, 100.0, 0.01) if v == "new" else _frame(1.0, 100.0, 0.01)
        ),
    )
    await detector.check_once()
    assert producer.send_event.called


@pytest.mark.asyncio
async def test_regression_detected_on_latency_increase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    monkeypatch.setattr(
        regression_mod, "get_recent_policy_versions", lambda *_a, **_k: ["new", "old"]
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _e, v, _s, _n: (
            _frame(1.0, 250.0, 0.01) if v == "new" else _frame(1.0, 100.0, 0.01)
        ),
    )
    await detector.check_once()
    assert producer.send_event.called


@pytest.mark.asyncio
async def test_regression_detected_on_slo_rate_increase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    monkeypatch.setattr(
        regression_mod, "get_recent_policy_versions", lambda *_a, **_k: ["new", "old"]
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _e, v, _s, _n: (
            _frame(1.0, 100.0, 0.20) if v == "new" else _frame(1.0, 100.0, 0.01)
        ),
    )
    await detector.check_once()
    assert producer.send_event.called


@pytest.mark.asyncio
async def test_regression_event_published_to_kafka(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    monkeypatch.setattr(
        regression_mod, "get_recent_policy_versions", lambda *_a, **_k: ["new", "old"]
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _e, v, _s, _n: (
            _frame(0.5, 100.0, 0.01) if v == "new" else _frame(1.0, 100.0, 0.01)
        ),
    )
    await detector.check_once()
    args = producer.send_event.call_args.args
    assert args[0] == TOPIC_REGRESSIONS


@pytest.mark.asyncio
async def test_detector_does_not_roll_back_only_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, producer = _detector()
    rollback = MagicMock()
    monkeypatch.setattr(
        regression_mod, "get_recent_policy_versions", lambda *_a, **_k: ["new", "old"]
    )
    monkeypatch.setattr(
        regression_mod,
        "get_policy_performance",
        lambda _e, v, _s, _n: (
            _frame(0.5, 100.0, 0.01) if v == "new" else _frame(1.0, 100.0, 0.01)
        ),
    )
    await detector.check_once()
    rollback.assert_not_called()
    assert producer.send_event.called


def _frame(
    avg_r_online: float, p95_latency_ms: float, slo_violation_rate: float
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "avg_r_online": avg_r_online,
                "p95_latency_ms": p95_latency_ms,
                "slo_violation_rate": slo_violation_rate,
            }
        ]
    )


def _detector() -> tuple[RegressionDetector, MagicMock]:
    producer = MagicMock()
    detector = RegressionDetector(
        engine=MagicMock(), producer=producer, interval_seconds=60
    )
    return detector, producer
