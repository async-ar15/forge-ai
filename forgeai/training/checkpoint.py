"""Deterministic checkpoint persistence and verification."""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from forgeai.registry.models.checkpoint import Checkpoint
from forgeai.training.orchestrator import TrainingJobConfig


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_id: uuid.UUID
    training_run_id: uuid.UUID
    step: int
    s3_path: str
    sha256_hash: str
    loss_at_step: float
    created_at: datetime
    verified: bool


class CheckpointManager:
    """Save, verify, and resume checkpoints with deterministic hashing."""

    def __init__(
        self, *, session_factory: async_sessionmaker[AsyncSession], boto3_module: object
    ) -> None:
        self._session_factory = session_factory
        self._boto3 = boto3_module

    async def save_checkpoint(
        self, trainer: object, step: int, config: TrainingJobConfig
    ) -> CheckpointRecord:
        local_dir = Path(tempfile.mkdtemp(prefix="forgeai-ckpt-"))
        trainer_any = cast(Any, trainer)
        trainer_any.save_model(str(local_dir))
        trainer_any.save_state()
        digest = _sha256_dir(local_dir)
        s3_path = await _upload_dir(self._boto3, local_dir, config.output_path, step)
        row = await self._insert_row(
            config.job_id,
            step,
            s3_path,
            digest,
            float(trainer_any.state.log_history[-1].get("loss", 0.0)),
        )
        return _row_to_record(row)

    async def verify_checkpoint(self, checkpoint_id: uuid.UUID) -> bool:
        row = await self._load_checkpoint(checkpoint_id)
        if row is None:
            return False
        local_dir = await _download_checkpoint_dir(self._boto3, row.s3_path)
        digest = _sha256_dir(local_dir)
        ok = digest == row.sha256_hash
        await self._set_verified(checkpoint_id, ok)
        return ok

    async def resume_from_checkpoint(
        self, training_run_id: uuid.UUID
    ) -> CheckpointRecord | None:
        """Never resume from unverified state; corruption risk is explicit."""

        async with self._session_factory() as session:
            stmt = (
                select(Checkpoint)
                .where(
                    Checkpoint.training_run_id == training_run_id,
                    Checkpoint.verified.is_(True),
                )
                .order_by(desc(Checkpoint.step))
                .limit(1)
            )
            row = await session.scalar(stmt)
            return None if row is None else _row_to_record(row)

    async def _insert_row(
        self,
        training_run_id: uuid.UUID,
        step: int,
        s3_path: str,
        sha256_hash: str,
        loss_at_step: float,
    ) -> Checkpoint:
        async with self._session_factory() as session:
            row = Checkpoint(
                training_run_id=training_run_id,
                step=step,
                s3_path=s3_path,
                sha256_hash=sha256_hash,
                loss_at_step=loss_at_step,
                verified=False,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def _load_checkpoint(self, checkpoint_id: uuid.UUID) -> Checkpoint | None:
        async with self._session_factory() as session:
            return await session.get(Checkpoint, checkpoint_id)

    async def _set_verified(self, checkpoint_id: uuid.UUID, verified: bool) -> None:
        async with self._session_factory() as session:
            row = await session.get(Checkpoint, checkpoint_id)
            if row is None:
                return
            row.verified = verified
            await session.commit()


def _sha256_dir(path: Path) -> str:
    """File system order is not guaranteed; sorting ensures deterministic hash."""

    hasher = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        hasher.update(str(file.relative_to(path)).encode("utf-8"))
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


async def _upload_dir(
    boto3_module: object, local_dir: Path, output_path: str, step: int
) -> str:
    bucket, prefix = _parse_s3_uri(output_path)
    client = cast(Any, boto3_module).client("s3")
    for file in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        key = f"{prefix}/step-{step}/{file.relative_to(local_dir)}"
        client.upload_file(str(file), bucket, key)
    return f"s3://{bucket}/{prefix}/step-{step}"


async def _download_checkpoint_dir(boto3_module: object, s3_path: str) -> Path:
    bucket, prefix = _parse_s3_uri(s3_path)
    local = Path(tempfile.mkdtemp(prefix="forgeai-verify-"))
    client = cast(Any, boto3_module).client("s3")
    listed = client.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    for obj in listed:
        key = obj["Key"]
        rel = key[len(prefix) :].lstrip("/")
        dest = local / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
    return local


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3 uri, got {uri}")
    payload = uri[5:]
    bucket, key = payload.split("/", 1)
    return bucket, key


def _row_to_record(row: Checkpoint) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=row.checkpoint_id,
        training_run_id=row.training_run_id,
        step=row.step,
        s3_path=row.s3_path,
        sha256_hash=row.sha256_hash,
        loss_at_step=row.loss_at_step,
        created_at=row.created_at,
        verified=row.verified,
    )


__all__ = ["CheckpointManager", "CheckpointRecord"]
