from __future__ import annotations

import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from forgeai.training.exceptions import CheckpointVerificationError
from forgeai.training.promoter import ModelPromoter


@pytest.mark.asyncio
async def test_promote_checkpoint_calls_verify_first() -> None:
    checkpoint_manager = MagicMock()
    checkpoint_manager.verify_checkpoint = AsyncMock(return_value=True)
    registry_stub = MagicMock()
    session_factory = MagicMock()
    promoter = ModelPromoter(
        checkpoint_manager=checkpoint_manager,
        registry_stub=registry_stub,
        session_factory=session_factory,
    )
    checkpoint_id = uuid.uuid4()
    row = types.SimpleNamespace(
        training_run_id=uuid.uuid4(), step=1, sha256_hash="abc", s3_path="s3://b/k"
    )
    promoter._load_checkpoint = AsyncMock(return_value=row)  # type: ignore[method-assign]
    promoter._create_artifact = AsyncMock(return_value=uuid.uuid4())  # type: ignore[method-assign]
    promoter._create_model_version = AsyncMock(return_value=uuid.uuid4())  # type: ignore[method-assign]
    await promoter.promote_checkpoint(checkpoint_id, "staging")
    checkpoint_manager.verify_checkpoint.assert_awaited_once_with(checkpoint_id)


@pytest.mark.asyncio
async def test_checkpoint_verification_error_on_failed_verify() -> None:
    checkpoint_manager = MagicMock()
    checkpoint_manager.verify_checkpoint = AsyncMock(return_value=False)
    promoter = ModelPromoter(
        checkpoint_manager=checkpoint_manager,
        registry_stub=MagicMock(),
        session_factory=MagicMock(),
    )
    row = types.SimpleNamespace(
        training_run_id=uuid.uuid4(), step=1, sha256_hash="abc", s3_path="s3://b/k"
    )
    promoter._load_checkpoint = AsyncMock(return_value=row)  # type: ignore[method-assign]
    with pytest.raises(CheckpointVerificationError):
        await promoter.promote_checkpoint(uuid.uuid4(), "staging")


@pytest.mark.asyncio
async def test_successful_promotion_creates_model_version_row() -> None:
    checkpoint_manager = MagicMock()
    checkpoint_manager.verify_checkpoint = AsyncMock(return_value=True)
    promoter = ModelPromoter(
        checkpoint_manager=checkpoint_manager,
        registry_stub=MagicMock(),
        session_factory=MagicMock(),
    )
    row = types.SimpleNamespace(
        training_run_id=uuid.uuid4(), step=1, sha256_hash="abc", s3_path="s3://b/k"
    )
    model_version_id = uuid.uuid4()
    promoter._load_checkpoint = AsyncMock(return_value=row)  # type: ignore[method-assign]
    promoter._create_artifact = AsyncMock(return_value=uuid.uuid4())  # type: ignore[method-assign]
    promoter._create_model_version = AsyncMock(return_value=model_version_id)  # type: ignore[method-assign]
    out = await promoter.promote_checkpoint(uuid.uuid4(), "production")
    assert out == model_version_id
