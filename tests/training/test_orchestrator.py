from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.registry.constants import TrainingRunStatus
from forgeai.training.orchestrator import (
    TrainingJobConfig,
    TrainingOrchestrator,
    TrainingState,
)


class _RayFake:
    def __init__(self) -> None:
        self._status = "RUNNING"
        self.cancel = MagicMock()

    def remote(self, fn):
        class _Remote:
            def remote(self, payload):  # noqa: ANN001
                return {"fn": fn, "payload": payload}

        return _Remote()

    def get_job_status(self, _ref):  # noqa: ANN001
        return self._status


def _cfg() -> TrainingJobConfig:
    return TrainingJobConfig(
        job_id=uuid.uuid4(),
        base_model_id=uuid.uuid4(),
        dataset_path="s3://bucket/data.jsonl",
        output_path="s3://bucket/out",
        seed=1234,
    )


@pytest.mark.asyncio
async def test_submit_training_job_creates_db_row_and_returns_handle() -> None:
    ray = _RayFake()
    orch = TrainingOrchestrator(session_factory=MagicMock(), ray_module=ray)
    orch._insert_training_run = AsyncMock(return_value=None)  # type: ignore[method-assign]
    handle = await orch.submit_training_job(_cfg())
    assert isinstance(handle.job_id, uuid.UUID)
    orch._insert_training_run.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_job_status_maps_ray_states_correctly() -> None:
    ray = _RayFake()
    orch = TrainingOrchestrator(session_factory=MagicMock(), ray_module=ray)
    cfg = _cfg()
    orch._insert_training_run = AsyncMock(return_value=None)  # type: ignore[method-assign]
    await orch.submit_training_job(cfg)
    orch._load_run = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            current_epoch=1, current_loss=0.1, error_message=None
        )
    )
    ray._status = "CHECKPOINTING"
    status = await orch.get_job_status(cfg.job_id)
    assert status.state == TrainingState.CHECKPOINTING


@pytest.mark.asyncio
async def test_cancel_job_updates_db_status() -> None:
    ray = _RayFake()
    orch = TrainingOrchestrator(session_factory=MagicMock(), ray_module=ray)
    cfg = _cfg()
    orch._insert_training_run = AsyncMock(return_value=None)  # type: ignore[method-assign]
    await orch.submit_training_job(cfg)
    orch._update_status = AsyncMock(return_value=None)  # type: ignore[method-assign]
    cancelled = await orch.cancel_job(cfg.job_id)
    assert cancelled is True
    orch._update_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        cfg.job_id,
        TrainingRunStatus.CANCELLED.value,
    )
