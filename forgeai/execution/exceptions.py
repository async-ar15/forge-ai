"""Typed execution failures (Section 2/5). Does not own: gateway retry policy."""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.enums import ModelTier, Precision


@dataclass(frozen=True, slots=True)
class PrecisionUnavailableError(Exception):
    """Registry cannot serve the requested precision for this model tier."""

    model_tier: ModelTier
    requested_precision: Precision
    available_precisions: tuple[Precision, ...]

    def __str__(self) -> str:
        """Human-readable summary; structured fields stay on the exception."""

        avail = ", ".join(p.value for p in self.available_precisions)
        return (
            f"precision_unavailable tier={self.model_tier.value} "
            f"requested={self.requested_precision.value} available=[{avail}]"
        )


@dataclass(frozen=True, slots=True)
class QuantizationMismatchError(Exception):
    """Artifact layout does not match declared quantization (e.g. missing AWQ)."""

    precision: Precision
    artifact_path: str
    detail: str

    def __str__(self) -> str:
        """Single-line message for operators."""

        return (
            f"quantization_mismatch precision={self.precision.value} "
            f"path={self.artifact_path!r} detail={self.detail}"
        )


__all__ = ["PrecisionUnavailableError", "QuantizationMismatchError"]
