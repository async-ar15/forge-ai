from __future__ import annotations

import tempfile
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.training.checkpoint import CheckpointManager, CheckpointRecord, _sha256_dir
from forgeai.training.orchestrator import TrainingJobConfig


def _cfg() -> TrainingJobConfig:
    return TrainingJobConfig(
        job_id=uuid.uuid4(),
        base_model_id=uuid.uuid4(),
        dataset_path="s3://bucket/data.jsonl",
        output_path="s3://bucket/out",
        seed=1,
    )


def _record(verified: bool, step: int) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=uuid.uuid4(),
        training_run_id=uuid.uuid4(),
        step=step,
        s3_path="s3://bucket/out/step-1",
        sha256_hash="abc",
        loss_at_step=0.1,
        created_at=datetime.now(UTC),
        verified=verified,
    )


@pytest.mark.asyncio
async def test_save_checkpoint_computes_and_stores_sha256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = CheckpointManager(session_factory=MagicMock(), boto3_module=MagicMock())
    trainer = MagicMock()
    trainer.state = types.SimpleNamespace(log_history=[{"loss": 0.2}])
    monkeypatch.setattr(
        mgr,
        "_insert_row",
        AsyncMock(
            return_value=types.SimpleNamespace(
                checkpoint_id=uuid.uuid4(),
                training_run_id=uuid.uuid4(),
                step=1,
                s3_path="s3://bucket/out/step-1",
                sha256_hash="abc",
                loss_at_step=0.2,
                created_at=datetime.now(UTC),
                verified=False,
            )
        ),
    )
    monkeypatch.setattr(
        "forgeai.training.checkpoint._upload_dir",
        AsyncMock(return_value="s3://bucket/out/step-1"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(
            "forgeai.training.checkpoint.tempfile.mkdtemp", MagicMock(return_value=tmp)
        )
        Path(tmp, "model.bin").write_bytes(b"model")
        rec = await mgr.save_checkpoint(trainer, 1, _cfg())
    assert rec.s3_path.endswith("step-1")


@pytest.mark.asyncio
async def test_verify_checkpoint_returns_true_on_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = CheckpointManager(session_factory=MagicMock(), boto3_module=MagicMock())
    row = types.SimpleNamespace(
        checkpoint_id=uuid.uuid4(), s3_path="s3://b/k", sha256_hash="x"
    )
    monkeypatch.setattr(mgr, "_load_checkpoint", AsyncMock(return_value=row))
    monkeypatch.setattr(
        "forgeai.training.checkpoint._download_checkpoint_dir",
        AsyncMock(return_value=Path(tempfile.mkdtemp())),
    )
    monkeypatch.setattr(
        "forgeai.training.checkpoint._sha256_dir", MagicMock(return_value="x")
    )
    mgr._set_verified = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.verify_checkpoint(row.checkpoint_id) is True


@pytest.mark.asyncio
async def test_verify_checkpoint_returns_false_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = CheckpointManager(session_factory=MagicMock(), boto3_module=MagicMock())
    row = types.SimpleNamespace(
        checkpoint_id=uuid.uuid4(), s3_path="s3://b/k", sha256_hash="x"
    )
    monkeypatch.setattr(mgr, "_load_checkpoint", AsyncMock(return_value=row))
    monkeypatch.setattr(
        "forgeai.training.checkpoint._download_checkpoint_dir",
        AsyncMock(return_value=Path(tempfile.mkdtemp())),
    )
    monkeypatch.setattr(
        "forgeai.training.checkpoint._sha256_dir", MagicMock(return_value="y")
    )
    mgr._set_verified = AsyncMock(return_value=None)  # type: ignore[method-assign]
    assert await mgr.verify_checkpoint(row.checkpoint_id) is False


@pytest.mark.asyncio
async def test_resume_from_checkpoint_latest_verified_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = CheckpointManager(session_factory=MagicMock(), boto3_module=MagicMock())
    expected = _record(True, 20)
    monkeypatch.setattr(
        mgr,
        "resume_from_checkpoint",
        AsyncMock(return_value=expected),
    )
    got = await mgr.resume_from_checkpoint(expected.training_run_id)
    assert got is not None
    assert got.step == 20


def test_sha256_dir_sorted_files_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    digest1 = _sha256_dir(tmp_path)
    digest2 = _sha256_dir(tmp_path)
    assert digest1 == digest2
