"""Checkpoint promotion into model registry as pending deployment artifacts."""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgeai.proto import registry_service_pb2
from forgeai.registry.models.artifact import Artifact
from forgeai.registry.models.checkpoint import Checkpoint
from forgeai.registry.models.training_run import TrainingRun
from forgeai.training.checkpoint import CheckpointManager
from forgeai.training.exceptions import CheckpointVerificationError

_LOG = logging.getLogger(__name__)


class ModelPromoter:
    """Verifies checkpoints and promotes into registry metadata."""

    def __init__(
        self,
        *,
        checkpoint_manager: CheckpointManager,
        registry_stub: object,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._checkpoint_manager = checkpoint_manager
        self._registry_stub = registry_stub
        self._session_factory = session_factory

    async def promote_checkpoint(
        self, checkpoint_id: uuid.UUID, target_environment: str
    ) -> uuid.UUID:
        verified = await self._checkpoint_manager.verify_checkpoint(checkpoint_id)
        row = await self._load_checkpoint(checkpoint_id)
        if row is None:
            raise KeyError(f"unknown checkpoint_id={checkpoint_id}")
        if not verified:
            raise CheckpointVerificationError(
                checkpoint_id=str(checkpoint_id),
                expected_sha256=row.sha256_hash,
                actual_sha256="mismatch",
            )
        artifact_id = await self._create_artifact(row.s3_path)
        model_version_id = await self._create_model_version(row, artifact_id)
        _LOG.info(
            "checkpoint_promoted checkpoint_id=%s "
            "model_version_id=%s target_environment=%s sha256_hash=%s",
            checkpoint_id,
            model_version_id,
            target_environment,
            row.sha256_hash,
        )
        return model_version_id

    async def _load_checkpoint(self, checkpoint_id: uuid.UUID) -> Checkpoint | None:
        async with self._session_factory() as session:
            return await session.get(Checkpoint, checkpoint_id)

    async def _create_artifact(self, s3_path: str) -> uuid.UUID:
        async with self._session_factory() as session:
            row = Artifact(storage_uri=s3_path)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.artifact_id

    async def _create_model_version(
        self, checkpoint: Checkpoint, artifact_id: uuid.UUID
    ) -> uuid.UUID:
        base_model_id = await self._load_base_model_id(checkpoint.training_run_id)
        registry_stub = cast(Any, self._registry_stub)
        request_cls = getattr(
            registry_stub,
            "RegistryCreateModelVersionRequest",
            registry_service_pb2.RegistryCreateModelVersionRequest,
        )
        req = request_cls(
            owning_model_id=str(base_model_id),
            semantic_version_tag=f"train-{checkpoint.training_run_id}-{checkpoint.step}",
            artifact_object_id=str(artifact_id),
            originating_training_run_id=str(checkpoint.training_run_id),
        )
        resp = await registry_stub.RegistryCreateModelVersion(req)
        return uuid.UUID(resp.created_model_version_id)

    async def _load_base_model_id(self, training_run_id: uuid.UUID) -> uuid.UUID:
        async with self._session_factory() as session:
            stmt = select(TrainingRun.base_model_id).where(
                TrainingRun.training_run_id == training_run_id
            )
            val = await session.scalar(stmt)
            if val is None:
                raise KeyError(
                    "missing training run for checkpoint "
                    f"training_run_id={training_run_id}"
                )
            return val


__all__ = ["ModelPromoter"]
