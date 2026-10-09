"""Ray orchestration and durable training-run status persistence."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgeai.registry.constants import TrainingRunStatus
from forgeai.registry.models.training_run import TrainingRun
from forgeai.training.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_STEPS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_SAVE_STEPS,
    DEFAULT_TARGET_MODULES,
    DEFAULT_USE_QLORA,
    DEFAULT_WARMUP_STEPS,
    TrainingConfigKey,
)


class _RayHandle(Protocol):
    def cancel(self) -> None: ...


class _RayRemoteFn(Protocol):
    def remote(self, config: dict[str, object]) -> _RayHandle: ...


class _RayModule(Protocol):
    def remote(
        self, fn: Callable[[dict[str, object]], dict[str, object]]
    ) -> _RayRemoteFn: ...
    def cancel(self, ref: _RayHandle) -> None: ...
    def get_job_status(self, ref: _RayHandle) -> object: ...


@dataclass(frozen=True, slots=True)
class TrainingJobConfig:
    """Training hyperparameters.

    seed is required to ensure reproducibility. there is no safe default - a
    caller that does not set a seed must do so explicitly.
    """

    job_id: uuid.UUID
    base_model_id: uuid.UUID
    dataset_path: str
    output_path: str
    seed: int
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: float = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_MODULES)
    )
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    use_qlora: bool = DEFAULT_USE_QLORA
    quantization_bits: int | None = None
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    save_steps: int = DEFAULT_SAVE_STEPS
    eval_steps: int = DEFAULT_EVAL_STEPS


@dataclass(frozen=True, slots=True)
class TrainingJobHandle:
    job_id: uuid.UUID
    submitted_at: datetime
    config_snapshot: dict[str, object]


class TrainingState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingJobStatus:
    job_id: uuid.UUID
    state: TrainingState
    progress_pct: float
    current_epoch: int
    current_loss: float
    best_checkpoint_path: str | None
    error_message: str | None


class TrainingOrchestrator:
    """Submit, observe, and cancel async training runs."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        ray_module: object,
    ) -> None:
        self._session_factory = session_factory
        self._ray = ray_module
        self._ray_address = _read_ray_address()
        self._job_refs: dict[uuid.UUID, _RayHandle] = {}

    async def submit_training_job(self, config: TrainingJobConfig) -> TrainingJobHandle:
        """Submit Ray job and persist config snapshot immediately."""

        submitted_at = datetime.now(UTC)
        ray = cast(_RayModule, self._ray)
        remote_fn = ray.remote(_ray_training_entrypoint)
        ref = remote_fn.remote(asdict(config))
        self._job_refs[config.job_id] = ref
        await self._insert_training_run(config, submitted_at)
        return TrainingJobHandle(
            job_id=config.job_id,
            submitted_at=submitted_at,
            config_snapshot=asdict(config),
        )

    async def get_job_status(self, job_id: uuid.UUID) -> TrainingJobStatus:
        """Poll Ray and map runtime status to ForgeAI status."""

        state, msg = _map_ray_state(self._ray, self._job_refs.get(job_id))
        row = await self._load_run(job_id)
        return TrainingJobStatus(
            job_id=job_id,
            state=state,
            progress_pct=float(row.current_epoch / max(row.current_epoch, 1)),
            current_epoch=int(row.current_epoch),
            current_loss=float(row.current_loss),
            best_checkpoint_path=None,
            error_message=msg or row.error_message,
        )

    async def cancel_job(self, job_id: uuid.UUID) -> bool:
        """Cancel running Ray job and mark durable status."""

        ref = self._job_refs.get(job_id)
        if ref is None:
            return False
        ray = cast(_RayModule, self._ray)
        ray.cancel(ref)
        await self._update_status(job_id, TrainingRunStatus.CANCELLED.value)
        return True

    async def _insert_training_run(
        self, config: TrainingJobConfig, submitted_at: datetime
    ) -> None:
        async with self._session_factory() as session:
            run = TrainingRun(
                job_id=config.job_id,
                base_model_id=config.base_model_id,
                config_snapshot=asdict(config),
                status=TrainingRunStatus.PENDING.value,
                current_epoch=0,
                current_loss=0.0,
                submitted_at=submitted_at,
                seed=config.seed,
            )
            session.add(run)
            await session.commit()

    async def _load_run(self, job_id: uuid.UUID) -> TrainingRun:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(TrainingRun).where(TrainingRun.job_id == job_id)
            )
            if row is None:
                raise KeyError(f"unknown training job_id={job_id}")
            return row

    async def _update_status(self, job_id: uuid.UUID, status: str) -> None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(TrainingRun).where(TrainingRun.job_id == job_id)
            )
            if row is None:
                return
            row.status = status
            await session.commit()


def _read_ray_address() -> str | None:
    """Read optional Ray cluster address from the environment."""

    raw = os.environ.get(TrainingConfigKey.RAY_ADDRESS.value)
    if raw is None or raw == "":
        return None
    return raw


def _map_ray_state(
    ray_module: object, ref: _RayHandle | None
) -> tuple[TrainingState, str | None]:
    if ref is None:
        return TrainingState.FAILED, "job_reference_missing"
    ray = cast(_RayModule, ray_module)
    status = "RUNNING"
    if hasattr(ray, "get_job_status"):
        status = str(ray.get_job_status(ref))
    if "." in status:
        status = status.rsplit(".", 1)[-1]
    status = status.upper()
    mapping = {
        "PENDING": TrainingState.PENDING,
        "RUNNING": TrainingState.RUNNING,
        "FINISHED": TrainingState.COMPLETED,
        "FAILED": TrainingState.FAILED,
        "STOPPED": TrainingState.CANCELLED,
        "CHECKPOINTING": TrainingState.CHECKPOINTING,
    }
    return mapping.get(status, TrainingState.RUNNING), None


def _ray_training_entrypoint(config_snapshot: dict[str, object]) -> dict[str, object]:
    return {"ok": True, "job_id": config_snapshot["job_id"]}


__all__ = [
    "TrainingJobConfig",
    "TrainingJobHandle",
    "TrainingJobStatus",
    "TrainingOrchestrator",
    "TrainingState",
]
