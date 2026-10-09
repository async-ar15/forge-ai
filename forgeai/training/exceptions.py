"""Structured training exceptions for deterministic failure handling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NaNLossError(RuntimeError):
    step: int
    epoch: float
    loss_value: float

    def __str__(self) -> str:
        return (
            f"NaN loss detected at step={self.step} "
            f"epoch={self.epoch} value={self.loss_value}"
        )


@dataclass(frozen=True, slots=True)
class DatasetSchemaError(ValueError):
    dataset_path: str
    line_index: int
    missing_keys: tuple[str, ...]

    def __str__(self) -> str:
        keys = ",".join(self.missing_keys)
        return (
            "Dataset schema invalid "
            f"path={self.dataset_path} line={self.line_index} missing={keys}"
        )


@dataclass(frozen=True, slots=True)
class CheckpointVerificationError(RuntimeError):
    checkpoint_id: str
    expected_sha256: str
    actual_sha256: str

    def __str__(self) -> str:
        return (
            "Checkpoint verification failed "
            f"checkpoint_id={self.checkpoint_id} "
            f"expected={self.expected_sha256} actual={self.actual_sha256}"
        )


__all__ = ["CheckpointVerificationError", "DatasetSchemaError", "NaNLossError"]
